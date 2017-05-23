import theano.tensor as T

from learning_to_learn.continuous_control.networks.dense_layer import DenseLayer
from learning_to_learn.continuous_control.networks.mlp import MLP
from learning_to_learn.continuous_control.util import leaky_relu


def create_model(input_dim=28 * 28,
                 output_dim=10,
                 units=256,
                 internal_activation=leaky_relu):
    model = MLP([DenseLayer(input_dim, units, activation=internal_activation, name="inner_1"),
                 DenseLayer(units, units, activation=internal_activation, name="inner_2"),
                 DenseLayer(units, units, activation=internal_activation, name="inner_3"),
                 DenseLayer(units, output_dim, activation=T.nnet.softmax, name="inner_4")])
    return model
