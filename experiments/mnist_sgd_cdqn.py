import sys
from learning_to_learn import train
from learning_to_learn.cdqn import create_agent_cdqn

ENV_NAME = 'SGD-MNIST-Continuous-v0'


def main(argv):
    train.main(argv,
               "output/mnist-sgd-cdqn/cdqn.h5",
               ENV_NAME,
               create_agent_cdqn)


if __name__ == '__main__':
    main(sys.argv[1:])
