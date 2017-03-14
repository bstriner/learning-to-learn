
from keras.optimizers import SGD
from .polynomial_env import PolynomialEnv
from ..utils.action_mapping import ActionMappingDiscrete


class PolynomialSgdDiscreteEnv(PolynomialEnv):
    def __init__(self):
        action_mapping = ActionMappingDiscrete(1, lambda opt: (opt.lr,), limits=[[1e-6, 1e-1]], scale=0.05)
        PolynomialEnv.__init__(self, action_mapping=action_mapping)

    def create_optimizer(self):
        lr = 1e-2
        self.optimizer = SGD(lr=lr)
