import theano.tensor as T
from learning_to_learn.continuous_control.dense_layer import DenseLayer
from learning_to_learn.continuous_control.util import leaky_relu, nll_loss
from learning_to_learn.continuous_control.mnist import mnist_batch_generator, mnist_generator
from learning_to_learn.continuous_control.internal_model import create_model
from learning_to_learn.continuous_control.mlp import MLP
from learning_to_learn.continuous_control.lookahead_model_initial import LookaheadModelInitial
from keras.optimizers import Adam
import numpy as np


def main():
    batch_size = 64
    lr_opt = Adam(1e-4)
    epochs = 10000
    validation_epochs = 64
    lr_units = 256
    inner_units = 256
    batches = 512
    frequency = 50
    depth = 100
    decay = 1.1
    schedule = np.power(decay, np.arange(depth))
    gen = mnist_batch_generator(batch_size=batch_size, depth=depth)
    inner_model = create_model(input_dim=28 * 28,
                               output_dim=10,
                               units=inner_units,
                               internal_activation=leaky_relu)
    lr_model = LookaheadModelInitial(inner_model=inner_model,
                                      loss_function=nll_loss,
                                      input_type=T.ftensor3,
                                      target_type=T.imatrix,
                                      lr_opt=lr_opt,
                                      schedule=schedule,
                                      units=lr_units)
    output_path = "output/mnist_lookahead_initial_50"
    lr_model.train(gen=gen,
                   epochs=epochs,
                   validation_epochs=validation_epochs,
                   frequency=frequency,
                   batches=batches,
                   output_path=output_path)


if __name__ == "__main__":
    main()
