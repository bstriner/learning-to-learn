import csv
import itertools
import os

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from learning_to_learn.continuous_control.networks.dense_layer import DenseLayer
from learning_to_learn.continuous_control.networks.mlp import MLP
from .util import leaky_relu, accuracy, cast_updates


class LookaheadRNNModel(object):
    def __init__(self,
                 inner_model,
                 loss_function,
                 target_type,
                 lr_opt,
                 depth,
                 decay,
                 units=256):
        self.inner_model = inner_model
        self.loss_function = loss_function
        assert isinstance(inner_model, MLP)
        self.depth = depth
        self.decay = theano.shared(value=np.float32(decay), name='decay')

        # input is batch
        # output is new lr
        batch = theano.shared(value=np.int32(1), name="batch")

        reset_vars = [
            (batch, np.int32(1))
        ]

        # Build LR prediction model
        lr_model = MLP([DenseLayer(1, units, activation=leaky_relu, name="outer_1"),
                        DenseLayer(units, units, activation=leaky_relu, name="outer_2"),
                        DenseLayer(units, units, activation=leaky_relu, name="outer_3"),
                        DenseLayer(units, 1, activation=T.nnet.sigmoid, name="outer_4")])

        offsets = T.arange(depth)
        batch_idxs = offsets + (batch.dimshuffle(('x',)))
        lr_input = T.reshape(T.log(batch_idxs), (-1, 1))
        lr_output = lr_model.call(lr_input)[:, 0]  # (depth,)

        # Model inputs
        input_x_train = T.ftensor3(name="input_x_train")
        target_y_train = T.imatrix(name="target_y_train")
        input_x_val = T.ftensor3(name="input_x_val")
        target_y_val = T.imatrix(name="target_y_val")
        inputs = [input_x_train, target_y_train, input_x_val, target_y_val]

        # Scan over iterations of SGD
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

        # Model to predict future discounted loss
        # Iterate over training loss and lr to predict validation loss
        rnn_model = MLP([DenseLayer(units, units, activation=leaky_relu, name="rnn_1"),
                        DenseLayer(units, units, activation=leaky_relu, name="rnn_2"),
                        DenseLayer(units, units, activation=leaky_relu, name="rnn_3"),
                        DenseLayer(units, 1, name="rnn_4")])
        loss_next_val_h0 = theano.shared(np.float32([0]), name='loss_next_val_h0')
        loss_next_val_shifted = T.concatenate((loss_next_val_h0, loss_next_val[:-1]),axis=0)
        loss_next_val_predicted = rnn_model.call(loss, lr_input, lr_output, loss_next_val_shifted)

        # r(t) = loss(t) + decay*r(t+1)
        rnn_loss = T.mean(T.abs_(loss_next_val_predicted-loss_next_val))


        weighted_loss = T.sum(loss_next_val * self.schedule)
        lr_updates = lr_opt.get_updates(lr_model.weights, {}, weighted_loss)
        inner_updates = [(w, wn[0, ...]) for w, wn in zip(inner_model.weights, weights_next)]
        other_updates = [
            (batch, batch + 1)
        ]
        updates = inner_updates + lr_updates + other_updates
            # current loss
            ypred = inner_model.call_on_weights(input_x_train, weights_t)
            loss = loss_function(target_y_train, ypred)
            ypred_val = inner_model.call_on_weights(input_x_val, weights_t)
            loss_val = loss_function(target_y_val, ypred_val)
            # update params
            lr_d = lr_output[d]
            weights_t = [w - (lr_d * T.grad(loss, w)) for w in weights_t]
            # next val loss
            ypred_next_val = inner_model.call_on_weights(input_x_val, weights_t)
            loss_next_val = loss_function(target_y_val, ypred_next_val)
            losses.append(loss_next_val)
            if d == 0:
                weights_next = weights_t
                acc = accuracy(target_y_train, ypred)
                acc_val = accuracy(target_y_val, ypred_val)
                outputs = [loss, acc, loss_val, acc_val, lr_d]

        lossv = T.stack(losses)
        assert lossv.ndim == 1
        assert self.schedule.ndim == 1
        weighted_loss = T.sum(lossv * self.schedule, axis=0)

        # train LR model to minimize validation loss
        lr_updates = lr_opt.get_updates(lr_model.weights, {}, weighted_loss)
        lr_updates = cast_updates(lr_updates)
        # lr_updates = list(downcast(lr_updates))

        inner_updates = [(w, nw) for w, nw in zip(inner_model.weights, weights_next)]
        inner_updates = cast_updates(inner_updates)
        other_updates = [
            (batch, batch + 1)
        ]
        other_updates = cast_updates(other_updates)
        updates = inner_updates + lr_updates + other_updates
        for u in updates:
            if u[0].ndim != u[1].ndim or u[0].dtype != u[1].dtype:
                print "Update {}: {}/{} -> {}/{}".format(u[0].name, u[0].ndim, u[0].dtype,
                                                         u[1].ndim, u[1].dtype)
        self.train_function = theano.function(inputs,
                                              outputs,
                                              updates=updates)
        srng = RandomStreams(123)
        self.reset_function = theano.function([], [], updates=reset_vars + inner_model.reset_updates(srng))

        # initialize the models before training
        print "Initializing model"
        reset_lr_function = theano.function([], [], updates=lr_model.reset_updates(srng))
        reset_lr_function()
        self.reset_function()

        self.validation_function = theano.function(inputs[:4],
                                                   outputs,
                                                   updates=inner_updates + other_updates)
        """
        lr_p = theano.shared(np.float32(0), name="initial_lr_p")
        lr_initial = T.nnet.sigmoid(lr_p)
        initial_weights = [p[1] for p in inner_model.reset_updates(srng)]
        initial_y = inner_model.call_on_weights(input_x_train, initial_weights)
        initial_loss = loss_function(target_y_train, initial_y)
        initial_y_val = inner_model.call_on_weights(input_x_val, initial_weights)
        initial_loss_val = loss_function(target_y_val, initial_y_val)
        initial_weights_next = [w - (lr_initial * T.grad(initial_loss, w)) for w in initial_weights]
        initial_y_next = inner_model.call_on_weights(input_x_train, initial_weights_next)
        initial_loss_next = loss_function(target_y_train, initial_y_next)
        initial_y_val_next = inner_model.call_on_weights(input_x_val, initial_weights_next)
        initial_loss_val_next = loss_function(target_y_val, initial_y_val_next)
        initial_opt = Adam(1e-3)
        initial_updates = initial_opt.get_updates([lr_p], {}, initial_loss_val_next)
        self.lr0_function = theano.function(inputs,
                                            [initial_loss, initial_loss_val,
                                             initial_loss_next, initial_loss_val_next,
                                             lr_initial],
                                            updates=initial_updates)
        """

    """
    def train_initial_lr(self, gen, epochs, output_path, verbose=True):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Epoch", "Loss", "Validation Loss", "Next Loss", "Next Validation Loss", "LR"])
            for epoch in tqdm(range(epochs), desc="Training initial LR"):
                inputs = next(gen)
                losses = self.lr0_function(*inputs)
                w.writerow([epoch] + losses)
                if verbose:
                    tqdm.write("Epoch {}: Loss: {}, Val Loss: {}, LR: {}".format(epoch,
                                                                                 losses[2], losses[3], losses[4]))
    """


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
            for epoch in tqdm(range(epochs), desc="Training"):
                # Reset MLP weights
                self.reset_function()
                for batch in tqdm(range(batches), desc="Epoch {}".format(epoch)):
                    inputs = list(itertools.chain.from_iterable(next(gen) for _ in range(self.depth)))
                    losses = self.train_function(*inputs)
                    del inputs
                    # gc.collect()
                w.writerow([epoch] + losses)

                if (epoch + 1) % frequency == 0:
                    self.validate(gen=gen, batches=batches, validation_epochs=validation_epochs,
                                  output_path=os.path.join(output_path, "epoch-{:09d}.csv".format(epoch)))
