import theano.tensor as T
from learning_to_learn.continuous_control.dense_layer import DenseLayer
from learning_to_learn.continuous_control.util import leaky_relu, nll_loss
from learning_to_learn.continuous_control.mnist import mnist_generator
from learning_to_learn.continuous_control.mlp import MLP
from learning_to_learn.continuous_control.baseline_model import BaselineModel


def create_model(input_dim, output_dim, units, internal_activation):
    model = MLP([DenseLayer(input_dim, units, activation=internal_activation, name="inner_1"),
                 DenseLayer(units, units, activation=internal_activation, name="inner_2"),
                 DenseLayer(units, units, activation=internal_activation, name="inner_3"),
                 DenseLayer(units, output_dim, activation=T.nnet.softmax, name="inner_4")])
    return model


def main():
    batch_size = 128
    innner_lr0 = 1e-1
    batches = 1000
    inner_units = 256
    gen = mnist_generator(batch_size)
    inner_model = create_model(input_dim=28 * 28,
                               output_dim=10,
                               units=inner_units,
                               internal_activation=leaky_relu)
    lr_model = BaselineModel(inner_model=inner_model,
                             loss_function=nll_loss,
                             target_type=T.ivector,
                             innner_lr0=innner_lr0)
    output_path = "output/mnist_baseline"
    lr_model.train_several(count=5,
                           gen=gen,
                           batches=batches,
                           output_path=output_path)


if __name__ == "__main__":
    main()
