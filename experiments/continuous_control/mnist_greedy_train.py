import theano.tensor as T
from keras.optimizers import Adam

from learning_to_learn.continuous_control.greedy_model import GreedyModel
from learning_to_learn.continuous_control.mnist import mnist_generator
from learning_to_learn.continuous_control.networks.dense_layer import DenseLayer
from learning_to_learn.continuous_control.networks.mlp import MLP
from learning_to_learn.continuous_control.util import leaky_relu, nll_loss


def create_model(input_dim, output_dim, units, internal_activation):
    model = MLP([DenseLayer(input_dim, units, activation=internal_activation, name="inner_1"),
                 DenseLayer(units, units, activation=internal_activation, name="inner_2"),
                 DenseLayer(units, units, activation=internal_activation, name="inner_3"),
                 DenseLayer(units, output_dim, activation=T.nnet.softmax, name="inner_4")])
    return model


def main():
    batch_size = 128
    lr_opt = Adam(1e-5)
    epochs = 10000
    batches = 500
    validation_epochs = 10
    lr_units = 256
    inner_units = 256
    frequency = 200
    gen = mnist_generator(batch_size)
    inner_model = create_model(input_dim=28 * 28,
                               output_dim=10,
                               units=inner_units,
                               internal_activation=leaky_relu)
    lr_model = GreedyModel(inner_model=inner_model,
                          loss_function=nll_loss,
                          target_type=T.ivector,
                          lr_opt=lr_opt,
                          units=lr_units)
    output_path = "output/mnist_greedy"
    lr_model.train(gen=gen,
                   epochs=epochs,
                   batches=batches,
                   validation_epochs=validation_epochs,
                   frequency=frequency,
                   output_path=output_path)


if __name__ == "__main__":
    main()
