from keras.datasets import mnist
import theano.tensor as T
import theano
from learning_to_learn.continuous_control.dense_layer import DenseLayer
import itertools
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np


def leaky_relu(x):
    return T.nnet.relu(x, 0.2)


def logit_np(x):
    return np.log(x / (1 - x))


def logit(x):
    # inverse of T.nnet.sigmoid
    return T.log(x / (1 - x))


def nll_loss(ytrue, ypred):
    assert ytrue.ndim == 1
    assert ypred.ndim == 2
    eps = np.float32(1e-6)
    return T.mean(-T.log(eps + ypred[T.arange(ypred.shape[0]), T.flatten(ytrue)]), axis=None)


def accuracy(ytrue, ypred):
    assert ytrue.ndim == 1
    assert ypred.ndim == 2
    return T.mean(T.eq(ytrue, T.argmax(ypred, axis=1)), axis=None)
