from learning_to_learn.continuous_control.experiments.baseline_experiment import baseline_experiment
from learning_to_learn.continuous_control.optimizers.sgd import VariableSGD


def main():
    inner_opt = VariableSGD()
    tasks = [
        ("sgd-lr-0.1", [0.1]),
        ("sgd-lr-0.01", [0.01]),
        ("sgd-lr-0.001", [0.001]),
        ("sgd-lr-0.5", [0.5]),
        ("sgd-lr-0.9", [0.9])
    ]
    output_path = "output/mnist_sgd_baseline"
    baseline_experiment(tasks=tasks,
                        output_path=output_path,
                        inner_opt=inner_opt)


if __name__ == "__main__":
    main()
