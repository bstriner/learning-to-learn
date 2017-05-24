import numpy as np
import theano
import theano.tensor as T

from .util import logit_np


def sigmoid_parameterization(depth, initial):
    params0 = np.repeat(np.array(initial).reshape((1, -1)), depth, axis=0)
    init = logit_np(params0).astype(np.float32)
    params = theano.shared(value=init, name="opt_params")
    opt_params = T.nnet.sigmoid(params)
    opt_params_array = [opt_params[:, i] for i in range(params0.shape[1])]
    return [params], opt_params_array


def exponential_parameterization(depth, initial):
    params0 = np.array(initial).reshape((1, -1)).astype(np.float32)
    params_t = theano.shared(value=logit_np(params0), name="opt_params")
    decay_t = theano.shared(value=np.zeros(params0.shape, dtype=np.float32), name="opt_decay")
    params_s = T.nnet.sigmoid(params_t)
    decay_s = T.nnet.sigmoid(decay_t)
    offsets = T.arange(depth)
    # Need to test these shaped
    opt_params = T.power(decay_s, offsets) * params_s
    opt_params_array = [opt_params[:, i] for i in range(params0.shape[1])]
    return [params_t, decay_t], opt_params_array
