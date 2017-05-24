import numpy as np
import theano
import theano.tensor as T

from .optimizer import VariableOptimizer
from ..util import logit_np


class VariableAdam(VariableOptimizer):
    def __init__(self,
                 initial_lr=1e-3,
                 initial_beta_1=0.9,
                 initial_beta_2=0.999,
                 epsilon=1e-8):
        self.initial_lr = np.float32(initial_lr)
        self.initial_beta_1 = np.float32(initial_beta_1)
        self.initial_beta_2 = np.float32(initial_beta_2)
        self.epsilon = epsilon
        self.lr = theano.shared(value=self.initial_lr, name='adam_lr')
        self.beta_1 = theano.shared(value=self.initial_beta_1, name='adam_beta_1')
        self.beta_2 = theano.shared(value=self.initial_beta_2, name='adam_beta_2')
        self.opt_params = [self.lr, self.beta_1, self.beta_2]
        self.opt_param_count = len(self.opt_params)
        self.opt_param_names = ["lr", "beta_1", "beta_2"]
        super(VariableAdam, self).__init__("Adam")

    def get_updates(self, loss, params, opt_params, opt_weights):
        assert len(opt_params) == 3
        assert len(opt_weights) == len(params) * 2
        lr = opt_params[0]
        beta_1 = opt_params[1]
        beta_2 = opt_params[2]
        ms = opt_weights[:len(params)]
        vs = opt_weights[len(params):]

        param_updates = []
        m_updates = []
        v_updates = []
        for p, m, v in zip(params, ms, vs):
            g = T.grad(loss, p)
            m_t = (beta_1 * m) + ((1. - beta_1) * g)
            v_t = (beta_2 * v) + ((1. - beta_2) * T.square(g))
            p_t = p - (lr * m_t / (T.sqrt(v_t + self.epsilon) + self.epsilon))
            param_updates.append(p_t)
            m_updates.append(m_t)
            v_updates.append(v_t)
        opt_weight_updates = m_updates + v_updates
        return param_updates, opt_weight_updates

    def get_opt_weights_initial(self, srng, params):
        ms = [T.zeros_like(p) for p in params]
        vs = [T.zeros_like(p) for p in params]
        return ms + vs

    def get_opt_params_initial(self):
        return [0.001, 0.9, 0.999]
