import numpy as np
import theano
import theano.tensor as T

from .optimizer import VariableOptimizer
from ..util import logit_np


class VariableSGD(VariableOptimizer):
    def __init__(self, initial_lr=1e-3):
        self.initial_lr = np.float32(initial_lr)
        self.lr = theano.shared(value=self.initial_lr, name='sgd_lr')
        self.opt_params = [self.lr]
        self.opt_param_count = len(self.opt_params)
        self.opt_param_names = ["lr"]
        super(VariableSGD, self).__init__("SGD")

    def get_updates(self, loss, params, opt_params, opt_weights):
        assert len(opt_params) == 1
        assert len(opt_weights) == 0
        lr = opt_params[0]
        param_updates = [p - (lr * T.grad(loss, p)) for p in params]
        return param_updates, []

    def get_opt_weights_initial(self, srng, params):
        return []

    def get_opt_params_initial(self):
        params0 = np.array([0.01]).reshape(1, -1)
        init = logit_np(params0).astype(np.float32)
        return init
