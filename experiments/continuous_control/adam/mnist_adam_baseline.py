from learning_to_learn.continuous_control.experiments.baseline_experiment import baseline_experiment
from learning_to_learn.continuous_control.optimizers.adam import VariableAdam


def main():
    inner_opt = VariableAdam()
    defaults = [0.9, 0.999]
    tasks = [
        ("adam-lr-0.1", [0.1] + defaults),
        ("adam-lr-0.01", [0.01] + defaults),
        ("adam-lr-0.001", [0.001] + defaults),
        ("adam-lr-0.5", [0.5] + defaults),
        ("adam-lr-0.9", [0.9] + defaults)
    ]
    output_path = "output/mnist_adam_baseline"
    baseline_experiment(tasks=tasks,
                        output_path=output_path,
                        inner_opt=inner_opt)


if __name__ == "__main__":
    main()
