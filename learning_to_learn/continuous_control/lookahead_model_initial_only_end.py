import csv
import os

import numpy as np
import theano
import theano.tensor as T
from keras.optimizers import Optimizer
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from .dense_layer import DenseLayer
from .mlp import MLP
from .util import leaky_relu, accuracy, cast_updates
from ..continuous_control.optimizers.optimizer import VariableOptimizer


class LookaheadModelInitialOnlyEnd(object):
    def __init__(self,
                 inner_model,
                 loss_function,
                 lr_opt,
                 inner_opt,
                 depth,
                 units=256):
        self.inner_model = inner_model
        self.loss_function = loss_function
        self.depth = depth
        assert isinstance(lr_opt, Optimizer)
        assert isinstance(inner_opt, VariableOptimizer)
        assert isinstance(inner_model, MLP)
        srng = RandomStreams(123)

        initial_weights = [u[1] for u in inner_model.reset_updates(srng=srng)]

        # Build LR prediction model
        lr_model = MLP([DenseLayer(1, units, activation=leaky_relu, name="outer_1"),
                        DenseLayer(units, units, activation=leaky_relu, name="outer_2"),
                        DenseLayer(units, units, activation=leaky_relu, name="outer_3"),
                        DenseLayer(units, 1, activation=T.nnet.sigmoid, name="outer_4")])

        lr_init = np.zeros((depth,), dtype='float32')
        lr_p = theano.shared(lr_init.astype(np.float32), name="lr_schedule")
        lr = T.nnet.sigmoid(lr_p)

        input_x_train = T.ftensor3(name="input_x_train")  # (depth, n, units)
        target_y_train = T.imatrix(name="target_y_train")  # (depth, n)
        input_x_val = T.fmatrix(name="input_x_val")  # (depth, n, units)
        target_y_val = T.ivector(name="target_y_val")  # (depth, n)

        outputs_info = ([None] * 5) + initial_weights
        sequences = [lr, input_x_train, target_y_train]
        non_sequences = [input_x_val, target_y_val]
        ret, _ = theano.scan(self.scan_fun,
                             sequences=sequences,
                             outputs_info=outputs_info,
                             non_sequences=non_sequences)
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

        final_loss = loss_next_val[-1]
        # weighted_loss = T.sum(loss_next_val * self.schedule)
        lr_updates = lr_opt.get_updates([lr_p], {}, final_loss)
        inputs = [input_x_train,
                  target_y_train,
                  input_x_val,
                  target_y_val]

        outputs = [loss, acc, loss_val, acc_val, lr]

        self.train_function = theano.function(inputs,
                                              final_loss,
                                              updates=cast_updates(lr_updates))

        # initialize the models before training
        print "Initializing model"
        reset_lr_function = theano.function([], [], updates=cast_updates(lr_model.reset_updates(srng)))
        reset_lr_function()
        self.validation_function = theano.function(inputs,
                                                   outputs)

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
        # priors
        inner_weights = params[idx:(idx + len(self.inner_model.weights))]
        idx += len(self.inner_model.weights)
        # non-sequences
        input_x_val = params[idx]
        idx += 1
        target_y_val = params[idx]
        idx += 1
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

    def validate(self, gen, validation_epochs, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        epoch_data = []
        for _ in tqdm(range(validation_epochs), desc="Validation"):
            inputs = next(gen)
            outputs = self.validation_function(*inputs)
            epoch_data.append(outputs)

        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(['Batch'] +
                       ["Average Loss", "Average Accuracy",
                        "Average Validation Loss", "Average Validation Accuracy",
                        "Average LR"] +
                       ['Loss {}'.format(i) for i in range(validation_epochs)] +
                       ['Accuracy {}'.format(i) for i in range(validation_epochs)] +
                       ['Validation Loss {}'.format(i) for i in range(validation_epochs)] +
                       ['Validation Accuracy {}'.format(i) for i in range(validation_epochs)] +
                       ['LR {}'.format(i) for i in range(validation_epochs)])
            for batch in range(self.depth):
                avgs = [np.mean([epoch_data[e][metric][batch]
                                 for e in range(validation_epochs)])
                        for metric in range(5)]
                data = [epoch_data[e][metric][batch] for metric in range(5)
                        for e in range(validation_epochs)]
                w.writerow([batch] + avgs + data)

    def train(self, gen, epochs, batches, validation_epochs, frequency, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.validate(gen=gen, validation_epochs=validation_epochs,
                      output_path=os.path.join(output_path, "initial.csv"))

        with open(os.path.join(output_path, "history.csv"), 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Epoch",
                        "Loss"])
            f.flush()
            it1 = tqdm(range(epochs), desc="Training")
            for epoch in it1:
                losses = []
                it2 = tqdm(range(batches), desc="Epoch {}".format(epoch))
                for batch in it2:
                    inputs = next(gen)
                    # print "Batch inputs: {}".format([z.shape for z in inputs])
                    l = self.train_function(*inputs)
                    del inputs
                    losses.append(l)
                    it2.desc = "Epoch {}, Loss {}".format(epoch, np.mean(losses))
                w.writerow([epoch, np.mean(losses)])
                f.flush()

                if (epoch + 1) % frequency == 0:
                    self.validate(gen=gen, validation_epochs=validation_epochs,
                                  output_path=os.path.join(output_path, "epoch-{:09d}.csv".format(epoch)))
