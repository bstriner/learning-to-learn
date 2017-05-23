import numpy as np
import theano.tensor as T
from keras.optimizers import Adam

from learning_to_learn.continuous_control.internal_model import create_model
from learning_to_learn.continuous_control.lookahead_model_symbolic import LookaheadModelSymbolic
from learning_to_learn.continuous_control.mnist import mnist_batch_generator
from learning_to_learn.continuous_control.util import leaky_relu, nll_loss


def main():
    batch_size = 64
    lr_opt = Adam(1e-5)
    epochs = 10000
    batches = 50

    validation_epochs = 10
    lr_units = 256
    inner_units = 256
    frequency = 200
    depth = 10
    decay = 1.1
    schedule = np.power(decay, np.arange(depth))
    gen = mnist_batch_generator(batch_size=batch_size, depth=depth)
    inner_model = create_model(input_dim=28 * 28,
                               output_dim=10,
                               units=inner_units,
                               internal_activation=leaky_relu)
    lr_model = LookaheadModelSymbolic(inner_model=inner_model,
                                      loss_function=nll_loss,
                                      input_type=T.ftensor3,
                                      target_type=T.imatrix,
                                      lr_opt=lr_opt,
                                      schedule=schedule,
                                      units=lr_units)
    output_path = "output/mnist_lookahead_symbolic"
    lr_model.train(gen=gen,
                   epochs=epochs,
                   batches=batches,
                   validation_epochs=validation_epochs,
                   frequency=frequency,
                   output_path=output_path)


if __name__ == "__main__":
    main()
