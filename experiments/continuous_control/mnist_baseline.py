import theano.tensor as T
from learning_to_learn.continuous_control.dense_layer import DenseLayer
from learning_to_learn.continuous_control.util import leaky_relu, nll_loss
from learning_to_learn.continuous_control.mnist import mnist_generator
from learning_to_learn.continuous_control.mlp import MLP
from learning_to_learn.continuous_control.baseline_model import BaselineModel
from learning_to_learn.continuous_control.internal_model import create_model
import os
import csv


def main():
    batch_size = 32
    batches = 500
    inner_units = 256
    count = 50
    gen = mnist_generator(batch_size)
    inner_model = create_model(input_dim=28 * 28,
                               output_dim=10,
                               units=inner_units,
                               internal_activation=leaky_relu)

    tasks = [
        ("0.1", 0.1),
        ("0.01", 0.01),
        ("0.001", 0.001),
        ("0.5", 0.5),
    ]
    output_path = "output/mnist_baseline"
    avgs = []
    for name, lr in tasks:
        lr_model = BaselineModel(inner_model=inner_model,
                                 loss_function=nll_loss,
                                 target_type=T.ivector,
                                 innner_lr0=lr)
        avg = lr_model.train_several(count=count,
                                     gen=gen,
                                     batches=batches,
                                     output_path=os.path.join(output_path,
                                                              "lr-{}.csv".format(name)))
        avgs.append(avg)

    with open(os.path.join(output_path, 'summary.csv'), 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Batch'] +
                   ['Loss (lr={})'.format(name) for name, lr in tasks] +
                   ['Acc (lr={})'.format(name) for name, lr in tasks] +
                   ['Val Loss (lr={})'.format(name) for name, lr in tasks] +
                   ['Val Acc (lr={})'.format(name) for name, lr in tasks])
        for batch in range(batches):
            row = [avgs[t][batch][c] for c in range(4) for t in range(len(tasks))]
            w.writerow([batch] + row)


if __name__ == "__main__":
    main()
