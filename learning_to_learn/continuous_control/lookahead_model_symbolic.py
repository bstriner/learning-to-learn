import gc
from tqdm import tqdm
import csv
import itertools
import os
import theano.tensor as T
import theano
import numpy as np
from .dense_layer import DenseLayer
from .util import leaky_relu, accuracy, logit, logit_np, cast_updates
from .mlp import MLP
from theano.tensor.shared_randomstreams import RandomStreams
from keras.optimizers import Adam


class LookaheadModelSymbolic(object):
    def __init__(self,
                 inner_model,
                 loss_function,
                 lr_opt,
                 schedule,
                 input_type=T.ftensor3,
                 target_type=T.imatrix,
                 units=256):
        self.inner_model = inner_model
        self.loss_function = loss_function
        assert isinstance(inner_model, MLP)
        assert schedule.ndim == 1
        self.schedule = theano.shared(value=np.float32(schedule), name="schedule")
        depth = schedule.shape[0]
        self.depth = depth
        # input is batch
        # output is new lr
        batch = theano.shared(value=np.float32(1), name="batch")

        reset_vars = [
            (batch, np.float32(1))
        ]

        # Build LR prediction model
        lr_model = MLP([DenseLayer(1, units, activation=leaky_relu, name="outer_1"),
                        DenseLayer(units, units, activation=leaky_relu, name="outer_2"),
                        DenseLayer(units, units, activation=leaky_relu, name="outer_3"),
                        DenseLayer(units, 1, activation=T.nnet.sigmoid, name="outer_4")])

        offsets = T.arange(depth, dtype='float32')
        batch_idxs = offsets + batch
        lr_input = T.reshape(T.log(batch_idxs), (-1, 1))
        lr_output = lr_model.call(lr_input)[:, 0]  # (depth,)
        print "Lr input {}, output {}".format(lr_input.dtype, lr_output.dtype)

        input_x_train = input_type(name="input_x_train")  # (depth, n, units)
        target_y_train = target_type(name="target_y_train")  # (depth, n)
        input_x_val = input_type(name="input_x_val")  # (depth, n, units)
        target_y_val = target_type(name="target_y_val")  # (depth, n)

        outputs_info = ([None] * 5) + inner_model.weights
        sequences = [lr_output, input_x_train, target_y_train, input_x_val, target_y_val]
        ret, _ = theano.scan(self.scan_fun, sequences=sequences, outputs_info=outputs_info)
        idx = 0
        loss = ret[idx]
        idx += 1
        acc = ret[idx]
        idx += 1
        loss_val = ret[idx]
        idx += 1
        acc_val = ret[idx]
        idx += 1
        loss_next_val = ret[idx]
        idx += 1
        weights_next = ret[idx:(idx + len(self.inner_model.weights))]
        idx += len(self.inner_model.weights)
        assert len(ret) == idx

        weighted_loss = T.sum(loss_next_val * self.schedule)
        lr_updates = lr_opt.get_updates(lr_model.weights, {}, weighted_loss)
        inner_updates = [(w, wn[0, ...]) for w, wn in zip(inner_model.weights, weights_next)]
        other_updates = [
            (batch, batch + 1)
        ]
        updates = inner_updates + lr_updates + other_updates
        inputs = [input_x_train,
                  target_y_train,
                  input_x_val,
                  target_y_val]

        outputs = [loss[0], acc[0], loss_val[0], acc_val[0], lr_output[0]]

        self.train_function = theano.function(inputs,
                                              outputs,
                                              updates=cast_updates(updates))
        srng = RandomStreams(123)
        self.reset_function = theano.function([], [], updates=cast_updates(reset_vars +
                                                                           inner_model.reset_updates(srng)))

        # initialize the models before training
        print "Initializing model"
        reset_lr_function = theano.function([], [], updates=cast_updates(lr_model.reset_updates(srng)))
        reset_lr_function()
        self.reset_function()
        self.validation_function = theano.function(inputs,
                                                   outputs,
                                                   updates=cast_updates(inner_updates + other_updates))

    def scan_fun(self, *params):
        print "Params"
        for i, p in enumerate(params):
            print "{}: {}, {}, {}".format(i, p, p.ndim, p.dtype)
        # sequences, priors, non-sequences
        # sequences
        idx = 0
        lr_t = params[idx]
        idx += 1
        input_x_train = params[idx]
        idx += 1
        target_y_train = params[idx]
        idx += 1
        input_x_val = params[idx]
        idx += 1
        target_y_val = params[idx]
        idx += 1
        # priors
        inner_weights = params[idx:(idx + len(self.inner_model.weights))]
        idx += len(self.inner_model.weights)
        # no non-sequences
        assert len(params) == idx

        ypred = self.inner_model.call_on_weights(input_x_train, inner_weights)
        loss = self.loss_function(target_y_train, ypred)
        acc = accuracy(target_y_train, ypred)
        ypred_val = self.inner_model.call_on_weights(input_x_val, inner_weights)
        loss_val = self.loss_function(target_y_val, ypred_val)
        acc_val = accuracy(target_y_val, ypred_val)

        # update weights
        weights_t = [w - (lr_t * T.grad(loss, w)) for w in inner_weights]

        # next val loss
        ypred_next_val = self.inner_model.call_on_weights(input_x_val, weights_t)
        loss_next_val = self.loss_function(target_y_val, ypred_next_val)
        return [loss, acc, loss_val, acc_val, loss_next_val] + weights_t

    def validate(self, gen, batches, validation_epochs, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        epoch_data = []
        for epoch in tqdm(range(validation_epochs), desc="Validation"):
            data = []
            self.reset_function()
            for batch in tqdm(range(batches), desc="Validation Epoch {}".format(epoch)):
                inputs = next(gen)
                outputs = self.validation_function(*inputs)
                data.append(outputs)
            epoch_data.append(data)
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(['Batch'] +
                       ['Loss {}'.format(i) for i in range(validation_epochs)] +
                       ['Accuracy {}'.format(i) for i in range(validation_epochs)] +
                       ['Validation Loss {}'.format(i) for i in range(validation_epochs)] +
                       ['Validation Accuracy {}'.format(i) for i in range(validation_epochs)] +
                       ['LR {}'.format(i) for i in range(validation_epochs)] +
                       ["Average Loss", "Average Accuracy",
                        "Average Validation Loss", "Average Validation Accuracy",
                        "Average LR"])
            for batch in range(batches):
                avgs = [np.mean([epoch_data[e][batch][metric]
                                 for e in range(validation_epochs)])
                        for metric in range(5)]
                w.writerow([batch] +
                           [epoch_data[i][batch][j] for j in range(5) for i in range(validation_epochs)] +
                           avgs)

    def train(self, gen, epochs, batches, validation_epochs, frequency, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.validate(gen=gen, batches=batches, validation_epochs=validation_epochs,
                      output_path=os.path.join(output_path, "initial.csv"))

        with open(os.path.join(output_path, "history.csv"), 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Epoch",
                        "Final Train Loss",
                        "Final Train Acc",
                        "Final Validation Loss",
                        "Final Validation Acc",
                        "Final Learning Rate"])
            f.flush()
            for epoch in tqdm(range(epochs), desc="Training"):
                # Reset MLP weights
                self.reset_function()
                for batch in tqdm(range(batches), desc="Epoch {}".format(epoch)):
                    inputs = next(gen)
                    # print "Batch inputs: {}".format([z.shape for z in inputs])
                    losses = self.train_function(*inputs)
                    del inputs
                    # gc.collect()
                w.writerow([epoch] + losses)
                f.flush()

                if (epoch + 1) % frequency == 0:
                    self.validate(gen=gen, batches=batches, validation_epochs=validation_epochs,
                                  output_path=os.path.join(output_path, "epoch-{:09d}.csv".format(epoch)))
