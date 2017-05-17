from tqdm import tqdm
import csv
import os
import theano.tensor as T
import theano
import numpy as np
from .dense_layer import DenseLayer
from .util import leaky_relu, accuracy
from .mlp import MLP
from theano.tensor.shared_randomstreams import RandomStreams

"""
Baseline model with constant LR
"""

class BaselineModel(object):
    def __init__(self,
                 inner_model,
                 loss_function,
                 target_type,
                 innner_lr0=1e-3):
        input_x_train = T.fmatrix(name="input_x_train")
        target_y_train = target_type(name="target_y_train")
        input_x_val = T.fmatrix(name="input_x_val")
        target_y_val = target_type(name="target_y_val")

        lr = theano.shared(np.float32(innner_lr0), name="lr")
        # current loss
        ypred = inner_model.call(input_x_train)
        loss = loss_function(target_y_train, ypred)
        acc = accuracy(target_y_train, ypred)

        # current val loss
        ypred_val = inner_model.call(input_x_val)
        loss_val = loss_function(target_y_val, ypred_val)
        acc_val = accuracy(target_y_val, ypred_val)

        # update params
        inner_weights_next = [w - (lr * T.grad(loss, w)) for w in inner_model.weights]
        inner_updates = [(w, T.cast(nw, 'float32')) for w, nw in zip(inner_model.weights, inner_weights_next)]
        inputs = [input_x_train, target_y_train, input_x_val, target_y_val]
        outputs = [loss, acc, loss_val, acc_val]
        self.train_function = theano.function(inputs,
                                              outputs,
                                              updates=inner_updates)
        srng = RandomStreams(123)
        self.reset_function = theano.function([], [], updates=inner_model.reset_updates(srng))

        # initialize the models before training
        print "Initializing model"
        self.reset_function()


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
            for epoch in tqdm(range(batches), desc="Training"):
                # Reset MLP weights
                inputs = next(gen)
                losses = self.train_function(*inputs)
                w.writerow([epoch, losses[0], losses[1], losses[2], losses[3]])


    def train_several(self, count, gen, batches, output_path):
        for i in range(count):
            self.reset_function()
            self.train(gen, batches, output_path="{}/test-{}.csv".format(output_path, i))