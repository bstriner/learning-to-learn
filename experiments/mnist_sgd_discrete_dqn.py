
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.regularizers import l1l2
from learning_to_learn import train

ENV_NAME = 'SGD-MNIST-Discrete-v0'


def create_model(window_length, env):
    nb_actions = env.action_space.n
    reg = lambda: l1l2(1e-7, 1e-7)
    dropout = 0.5
    model = Sequential()
    model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    model.add(Dense(128, W_regularizer=reg()))
    model.add(Dropout(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Dense(64, W_regularizer=reg()))
    model.add(Dropout(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Dense(32, W_regularizer=reg()))
    model.add(Dropout(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Dense(nb_actions, W_regularizer=reg()))
    return model


def main(argv):
    train.main(argv,
             "output/mnist-sgd-discrete/dqn.h5",
               ENV_NAME,
               create_model)


if __name__ == '__main__':
    main(sys.argv[1:])
