from keras.optimizers import Adam

from learning_to_learn.continuous_control.internal_model import create_model
from learning_to_learn.continuous_control.lookahead_model_initial_only_end import LookaheadModelInitialOnlyEnd
from learning_to_learn.continuous_control.mnist import mnist_multiple_batch_generator
from learning_to_learn.continuous_control.util import leaky_relu, nll_loss
from learning_to_learn.continuous_control.optimizers.sgd import VariableSGD

def main():
    batch_size = 32
    lr_opt = Adam(1e-4)
    inner_opt = VariableSGD()
    epochs = 10000
    validation_epochs = 64
    lr_units = 256
    inner_units = 256
    batches = 512
    frequency = 20
    depth = 50
    validation_batch_size = 128
    gen = mnist_multiple_batch_generator(batch_size=batch_size,
                                         depth=depth,
                                         validation_batch_size=validation_batch_size)
    inner_model = create_model(input_dim=28 * 28,
                               output_dim=10,
                               units=inner_units,
                               internal_activation=leaky_relu)
    lr_model = LookaheadModelInitialOnlyEnd(inner_model=inner_model,
                                            loss_function=nll_loss,
                                            lr_opt=lr_opt,
                                            inner_opt=inner_opt,
                                            depth=depth,
                                            units=lr_units)
    output_path = "output/mnist_lookahead_initial_50_only_end"
    lr_model.train(gen=gen,
                   epochs=epochs,
                   validation_epochs=validation_epochs,
                   frequency=frequency,
                   batches=batches,
                   output_path=output_path)


if __name__ == "__main__":
    main()
