from keras.optimizers import Adam

from learning_to_learn.continuous_control.experiments.final_loss_experiment import final_loss_experiment
from learning_to_learn.continuous_control.optimizers.sgd import VariableSGD


def main():
    lr_opt = Adam(1e-4)
    inner_opt = VariableSGD()
    depth = 50
    output_path = "output/mnist_sgd_final_loss_50"
    final_loss_experiment(lr_opt=lr_opt,
                          inner_opt=inner_opt,
                          output_path=output_path,
                          depth=depth)


if __name__ == "__main__":
    main()
