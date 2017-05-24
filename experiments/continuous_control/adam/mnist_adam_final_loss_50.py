from keras.optimizers import Adam

from learning_to_learn.continuous_control.experiments.final_loss_experiment import final_loss_experiment
from learning_to_learn.continuous_control.optimizers.adam import VariableAdam
from learning_to_learn.continuous_control.parameterizations import sigmoid_parameterization


def main():
    lr_opt = Adam(1e-4)
    inner_opt = VariableAdam()
    depth = 50
    output_path = "output/mnist_adam_final_loss_50"
    final_loss_experiment(lr_opt=lr_opt,
                          inner_opt=inner_opt,
                          output_path=output_path,
                          parameterization=sigmoid_parameterization,
                          depth=depth)


if __name__ == "__main__":
    main()
