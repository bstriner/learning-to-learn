from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.optimizers import SGD
from ..datasets import mnist
from .base_env import BaseEnv


class MnistEnv(BaseEnv):
    def __init__(self, action_mapping):
        self.input_shape = (28, 28)
        self.output_dim = 10
        self.batch_size = 32
        self.data_train, self.data_val, self.data_test = mnist.load_data()
        self.max_steps = 200
        BaseEnv.__init__(self, action_mapping=action_mapping)

    def create_model(self):
        x = Input(self.input_shape)
        h = Flatten()(x)
        nch = 256
        h = Dense(nch, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(nch/2, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(nch/4, activation='relu')(h)
        h = Dropout(0.5)(h)
        y = Dense(self.output_dim, activation='softmax')(h)
        self.model = Model(x, y)
        self.create_optimizer()
        self.model.compile(self.optimizer, 'categorical_crossentropy')
        self.current_step = 0
