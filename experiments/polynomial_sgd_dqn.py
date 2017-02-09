import sys
from learning_to_learn import train
from learning_to_learn.dqn import create_agent_dqn

ENV_NAME = 'SGD-Polynomial-Discrete-v0'


def main(argv):
    train.main(argv,
               "output/polynomial-sgd-dqn/dqn.h5",
               ENV_NAME,
               create_agent_dqn)


if __name__ == '__main__':
    main(sys.argv[1:])
