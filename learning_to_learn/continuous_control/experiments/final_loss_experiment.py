from learning_to_learn.continuous_control.internal_model import create_model
from learning_to_learn.continuous_control.models.final_loss_model import FinalLossModel
from learning_to_learn.continuous_control.mnist import mnist_multiple_batch_generator
from learning_to_learn.continuous_control.util import leaky_relu, nll_loss


def final_loss_experiment(output_path,
                          inner_opt,
                          lr_opt,
                          depth,
                          batch_size=32,
                          epochs=1000,
                          validation_epochs=64,
                          inner_units=256,
                          batches=512,
                          validation_batch_size=512,
                          frequency=20):
    gen = mnist_multiple_batch_generator(batch_size=batch_size,
                                         depth=depth,
                                         validation_batch_size=validation_batch_size)
    inner_model = create_model(input_dim=28 * 28,
                               output_dim=10,
                               units=inner_units,
                               internal_activation=leaky_relu)
    lr_model = FinalLossModel(inner_model=inner_model,
                                            loss_function=nll_loss,
                                            lr_opt=lr_opt,
                                            inner_opt=inner_opt,
                                            depth=depth,)
    lr_model.train(gen=gen,
                   epochs=epochs,
                   validation_epochs=validation_epochs,
                   frequency=frequency,
                   batches=batches,
                   output_path=output_path)
