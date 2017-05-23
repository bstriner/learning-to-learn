import csv
import itertools
import os

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from learning_to_learn.continuous_control.util import accuracy
from .model import ControlModel
from .util import get_tensors
from ..optimizers.optimizer import VariableOptimizer

"""
Baseline model with constant LR
"""


class BaselineModel(ControlModel):
    def __init__(self,
                 inner_model,
                 loss_function,
                 target_type,
                 opt_params,
                 inner_opt):
        assert isinstance(inner_opt, VariableOptimizer)
        assert len(opt_params) == len(inner_opt.opt_params)

        srng = RandomStreams(123)
        input_x_train = T.fmatrix(name="input_x_train")
        target_y_train = target_type(name="target_y_train")
        input_x_val = T.fmatrix(name="input_x_val")
        target_y_val = target_type(name="target_y_val")

        # current loss
        ypred = inner_model.call(input_x_train)
        loss = loss_function(target_y_train, ypred)
        acc = accuracy(target_y_train, ypred)

        # current val loss
        ypred_val = inner_model.call(input_x_val)
        loss_val = loss_function(target_y_val, ypred_val)
        acc_val = accuracy(target_y_val, ypred_val)

        # update params
        opt_weights_initial = inner_opt.get_opt_weights_initial(srng, inner_model.weights)
        opt_weights = [theano.shared(w) for w in get_tensors(opt_weights_initial)]
        inner_params_t, opt_weights_t = inner_opt.get_updates(loss=loss,
                                                              params=inner_model.weights,
                                                              opt_params=opt_params,
                                                              opt_weights=opt_weights)

        inputs = [input_x_train, target_y_train, input_x_val, target_y_val]
        outputs = [loss, acc, loss_val, acc_val]
        inner_updates = [(p, p_t) for p, p_t in zip(inner_model.weights, inner_params_t)]
        opt_weight_updates = [(w, w_t) for w, w_t in zip(opt_weights, opt_weights_t)]
        self.train_function = theano.function(inputs,
                                              outputs,
                                              updates=inner_updates + opt_weight_updates)

        reset_weights = [(w, w0) for w, w0 in zip(opt_weights, opt_weights_initial)]
        reset_inner = inner_model.reset_updates(srng)
        self.reset_function = theano.function([], [],
                                              updates=reset_inner + reset_weights)

        # initialize the models before training
        print "Initializing model"
        self.reset_function()
        super(BaselineModel, self).__init__()

    def train(self, gen, batches, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Batch",
                        "Train Loss",
                        "Train Acc",
                        "Validation Loss",
                        "Validation Acc"])
            for batch in tqdm(range(batches), desc="Training"):
                # Reset MLP weights
                inputs = next(gen)
                losses = self.train_function(*inputs)
                w.writerow([batch, losses[0], losses[1], losses[2], losses[3]])

    def train_several(self, count, gen, batches, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        all_losses = []
        for _ in tqdm(range(count), "Baseline training"):
            self.reset_function()
            losses = []
            for _ in tqdm(range(batches), desc="Training"):
                inputs = next(gen)
                loss = self.train_function(*inputs)
                losses.append(loss)
            all_losses.append(losses)
        avgs = []
        with open(output_path, 'wb') as f:
            w = csv.writer(f)
            w.writerow(["Batch"] +
                       ["Avg Loss",
                        "Avg Acc",
                        "Avg Val Loss",
                        "Avg Val Acc"] +
                       ["Train Loss {}".format(i) for i in range(count)] +
                       ["Train Acc {}".format(i) for i in range(count)] +
                       ["Val Loss {}".format(i) for i in range(count)] +
                       ["Val Acc {}".format(i) for i in range(count)])
            for batch in range(batches):
                avg = [np.mean([all_losses[k][batch][j]
                                for k in range(count)])
                       for j in range(4)]
                data = list(itertools.chain.from_iterable([
                    [all_losses[k][batch][j] for k in range(count)]
                    for j in range(4)]))
                w.writerow([batch] + avg + data)
                avgs.append(avg)
        return avgs
