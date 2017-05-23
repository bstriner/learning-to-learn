import itertools


class MLP(object):
    def __init__(self, layers):
        self.layers = layers
        self.weights = list(itertools.chain.from_iterable(layer.weights for layer in layers))

    def call(self, x):
        return self.call_on_weights(x, self.weights)

    def call_on_weights(self, x, weights):
        idx = 0
        h = x
        for layer in self.layers:
            w = weights[idx:(idx + len(layer.weights))]
            idx += len(layer.weights)
            h = layer.call_on_weights(h, w)
        return h

    def reset_updates(self, srng):
        return list(itertools.chain.from_iterable(layer.reset_updates(srng) for layer in self.layers))
