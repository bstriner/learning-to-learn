
from keras.optimizers import SGD
from .mnist_env import MnistEnv
from ..utils.action_mapping import ActionMappingContinuousLogarithmic


class MnistSgdContinuousEnv(MnistEnv):
    def __init__(self):
        action_mapping = ActionMappingContinuousLogarithmic(1, lambda opt: (opt.lr,), limits=[[1e-9, 1e-1]])
        MnistEnv.__init__(self, action_mapping=action_mapping)

    def create_optimizer(self):
        lr = 1e-3
        self.optimizer = SGD(lr=lr)
