import warnings

import numpy as np
import theano
import theano.tensor as T


def get_tensors(tensors):
    assert isinstance(tensors, (list, tuple))
    f = theano.function([], tensors)
    return f()


def leaky_relu(x):
    return T.nnet.relu(x, 0.2)


def logit_np(x):
    return np.log(x / (1 - x))


def logit(x):
    # inverse of T.nnet.sigmoid
    return T.log(x / (1 - x))


def random_uniform_init_T(w, srng):
    scale = 0.05
    print "Init w: {}, {}, {}".format(w, w.dtype, w.ndim)
    return srng.uniform(low=-scale, high=scale, size=w.shape, dtype=w.dtype)


def nll_loss(ytrue, ypred):
    assert ytrue.ndim == 1
    assert ypred.ndim == 2
    eps = np.float32(1e-6)
    return T.mean(-T.log(eps + ypred[T.arange(ypred.shape[0]), T.flatten(ytrue)]), axis=None)


def accuracy(ytrue, ypred):
    assert ytrue.ndim == 1
    assert ypred.ndim == 2
    return T.mean(T.eq(ytrue, T.argmax(ypred, axis=1)), axis=None)


def cast_updates_gen(updates):
    for k, v in updates:
        if k.ndim != v.ndim:
            raise ValueError("Incorrect ndim {}. {} -> {}".format(k, k.ndim, v.ndim))
        if k.dtype != v.dtype:
            warnings.warn("Warning: Casting {} update from {} to {}".format(k, v.dtype, k.dtype))
            yield k, T.cast(v, k.dtype)
        else:
            yield k, v


def cast_updates(updates):
    return list(cast_updates_gen(updates))
