from learning_to_learn.continuous_control.util import leaky_relu, nll_loss
from learning_to_learn.continuous_control.mlp import MLP
from learning_to_learn.continuous_control.dense_layer import DenseLayer
import theano.tensor as T


def create_model(input_dim=28 * 28,
                 output_dim=10,
                 units=256,
                 internal_activation=leaky_relu):
    model = MLP([DenseLayer(input_dim, units, activation=internal_activation, name="inner_1"),
                 DenseLayer(units, units, activation=internal_activation, name="inner_2"),
                 DenseLayer(units, units, activation=internal_activation, name="inner_3"),
                 DenseLayer(units, output_dim, activation=T.nnet.softmax, name="inner_4")])
    return model
