import theano.tensor as T
import theano
import numpy as np


def random_uniform_init_T(w, srng):
    scale = 0.05
    print "Init w: {}, {}, {}".format(w, w.dtype, w.ndim)
    return srng.uniform(low=-scale, high=scale, size=w.shape, dtype=w.dtype)


class DenseLayer(object):
    def __init__(self, input_units, units, name, activation=None, init=random_uniform_init_T):
        self.input_units = input_units
        self.units = units
        self.activation = activation
        self.init = init

        self.kernel = theano.shared(value=np.zeros((input_units, units), dtype=np.float32),
                                    name="{}_kernel".format(name))
        self.bias = theano.shared(value=np.zeros((units,), dtype=np.float32),
                                  name="{}_bias".format(name))
        self.weights = [self.kernel, self.bias]

    def call(self, x):
        return self.call_on_weights(x, self.weights)

    def call_on_weights(self, x, weights):
        assert len(weights) == 2
        kernel, bias = weights
        y = T.dot(x, kernel) + bias
        if self.activation:
            y = self.activation(y)
        return y

    def reset_updates(self, srng):
        updates = [(w, random_uniform_init_T(w, srng)) for w in self.weights]
        return updates
