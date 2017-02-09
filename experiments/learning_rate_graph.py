import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def main(argv):
    parser = argparse.ArgumentParser(description='Graph learning rates over time.')
    fn_default = 'output\polynomial-sgd-discrete\dqn.h5-test.csv'
    parser.add_argument('--input', action="store", help='input filename (default: {})'.format(fn_default))
    parser.add_argument('--lr', action="store", help='learning rate graph filename (default: [input]-lr.png)')
    parser.add_argument('--loss', action="store", help='loss graph filename (default: [input]-lr.png)')
    args = parser.parse_args(argv)
    fn = args.input or fn_default
    lrpath = args.lr or "{}-lr.png".format(fn)
    losspath = args.loss or "{}-loss.png".format(fn)

    df = pd.read_csv(fn)
    epoch = df["epoch"].as_matrix()
    iteration = df["iteration"].as_matrix()
    # loss_train
    # loss_val
    nl_lr = df["nl_lr"].as_matrix()
    lr = np.exp(-nl_lr)
    loss = df["reward"].as_matrix()
    #loss = np.exp(-loss)
    min_epoch = np.min(epoch, axis=None)
    max_epoch = np.max(epoch, axis=None)
    iters = np.max(iteration, axis=None)
    data = np.zeros((iters + 1, max_epoch - min_epoch + 1, 2))
    for e in range(min_epoch, max_epoch + 1):
        d1 = lr[np.where(epoch == e)]
        d2 = loss[np.where(epoch == e)]
        data[:, e - min_epoch, 0] = d1
        data[:, e - min_epoch, 1] = d2

    avg = np.mean(data, axis=1)
    idx = np.arange(0, iters + 1)

    # Plot each trial
    fig = plt.figure()
    plt.plot(idx, data[:, :, 0])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.xlim(0, iters)
    fig.savefig(lrpath)
    plt.close(fig)

    # Plot average
    fig = plt.figure()
    plt.plot(idx, avg[:, 0])
    plt.xlabel("Epoch")
    plt.ylabel("Average Learning Rate")
    plt.xlim(0, iters)
    fig.savefig("{}-avg.png".format(os.path.join(os.path.dirname(lrpath), os.path.basename(lrpath))))
    plt.close(fig)

    # Plot each trial
    fig = plt.figure()
    plt.plot(idx, data[:, :, 1])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(0, iters)
    fig.savefig(losspath)
    plt.close(fig)

    # Plot average
    fig = plt.figure()
    plt.plot(idx, avg[:, 1])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(0, iters)
    fig.savefig("{}-avg.png".format(os.path.join(os.path.dirname(losspath), os.path.basename(losspath))))
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
