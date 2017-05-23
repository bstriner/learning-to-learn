import theano.tensor as T
from keras.optimizers import Adam

from learning_to_learn.continuous_control.greedy_model import BatchModel
from learning_to_learn.continuous_control.internal_model import create_model
from learning_to_learn.continuous_control.mnist import mnist_generator
from learning_to_learn.continuous_control.util import nll_loss


def main():
    batch_size = 128
    innner_lr0 = 1e-2
    lr_opt = Adam(1e-4)
    epochs = 10000
    lr_units = 256
    gen = mnist_generator(batch_size)
    inner_model = create_model()
    lr_model = BatchModel(inner_model=inner_model,
                          loss_function=nll_loss,
                          target_type=T.ivector,
                          innner_lr0=innner_lr0,
                          lr_opt=lr_opt,
                          units=lr_units)
    output_path = "output/mnist_initial_lr.csv"
    lr_model.train_initial_lr(gen=gen,
                   epochs=epochs,
                   output_path=output_path)


if __name__ == "__main__":
    main()
