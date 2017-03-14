from keras.models import Model
from keras.layers import Input, Dense, Flatten, LeakyReLU
from ..datasets import polynomial
from .base_env import BaseEnv
from keras.regularizers import l1l2


class PolynomialEnv(BaseEnv):
    def __init__(self, action_mapping):
        self.output_dim = 1
        self.batch_size = 32
        self.max_steps = 500
        self.data_train, self.data_val, self.data_test = None, None, None
        self.regenerate = True
        BaseEnv.__init__(self, action_mapping=action_mapping)

    def load_data(self):
        if (self.data_train is None) or self.regenerate:
            self.data_train, self.data_val, self.data_test = polynomial.load_data()

    def create_model(self):
        self.load_data()
        input_dim = self.data_train[0].shape[1]
        x = Input((input_dim,))
        #reg = lambda: l1l2(1e-7, 1e-7)
        reg = lambda: None
        nch = 256
        h = Dense(nch, W_regularizer=reg())(x)
        h = LeakyReLU(0.2)(h)
        h = Dense(nch, W_regularizer=reg())(h)
        h = LeakyReLU(0.2)(h)
        h = Dense(nch, W_regularizer=reg())(h)
        h = LeakyReLU(0.2)(h)
        y = Dense(self.output_dim, W_regularizer=reg())(h)
        self.model = Model(x, y)
        self.create_optimizer()
        self.model.compile(self.optimizer, 'mean_squared_error')
