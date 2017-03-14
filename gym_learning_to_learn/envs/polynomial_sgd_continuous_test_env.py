from keras.optimizers import SGD
from .polynomial_env import PolynomialEnv
from .polynomial_sgd_continuous_env import PolynomialSgdContinuousEnv
from gym_learning_to_learn.utils.np_utils import load_dataset


class PolynomialSgdContinuousTestEnv(PolynomialSgdContinuousEnv):
    def load_data(self):
        if (self.data_train is None):
            test_set = "output/polynomial/test-set.npz"
            self.data_train, self.data_val, self.data_test = load_dataset(test_set)
