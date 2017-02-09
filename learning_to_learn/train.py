import argparse
import numpy as np
import gym
from gym import wrappers
import pandas as pd
import sys
import gym_learning_to_learn
import pandas as pd
from gym_learning_to_learn.wrappers import DataRecorder
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.regularizers import l1l2
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import os


def main(argv, default_path, env_name, create_agent):
    parser = argparse.ArgumentParser(description='Train a DQN to control hyperparameters.')
    parser.add_argument('--create', action="store_true", help='create the model')
    parser.add_argument('--load', action="store_true", help='load the model')
    parser.add_argument('--train', action="store_true", help='train the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--steps', nargs="?", default=100000, type=int, action='store',
                        help='number of steps to train (default=100000)')
    parser.add_argument('--memory', nargs="?", default=5000, type=int, action='store',
                        help='memory size (default=5000)')
    parser.add_argument('--window', nargs="?", default=5, type=int, action='store',
                        help='window size (default=5)')
    parser.add_argument('--src', nargs="?", default=default_path, action='store',
                        help='source file (default: {})'.format(default_path))
    parser.add_argument('--dst', nargs="?", default=default_path, action='store',
                        help='destination file (default: {})'.format(default_path))
    default_test_path = "{}-test.csv".format(default_path)
    parser.add_argument('--test-dst', nargs="?", default=default_test_path, action='store',
                        help='destination file (default: {})'.format(default_test_path))
    args = parser.parse_args(argv)
    if not (args.create or args.load):
        print("Must choose one of --create or --load")
        parser.print_help()
        # parser.print_usage() # for just the usage line
        parser.exit()
    path = os.path.dirname(os.path.abspath(args.dst))
    if not os.path.exists(path):
        os.makedirs(path)
    csvpath = "{}.csv".format(args.dst)
    h5path = args.dst
    # Get the environment and extract the number of actions.
    env = gym.make(env_name)
    np.random.seed(123)
    env.seed(123)
    #nb_actions = env.action_space.n
    verbose = 1 if args.verbose else 0
    env.env.verbose = verbose

    # Next, we build a very simple model.
    dqn = create_agent(env, args)

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.

    visualize = False

    if args.create:
        dqn.save_weights(h5path, overwrite=True)

    if args.load:
        dqn.load_weights(args.src)

    if args.train:
        history = dqn.fit(env, nb_steps=args.steps, visualize=visualize, verbose=2)
        pd.DataFrame(history.history).to_csv(csvpath)
        dqn.save_weights(h5path, overwrite=True)

    if args.test:
        # Finally, evaluate our algorithm for 5 episodes.
        # env = wrappers.Monitor(env, 'output/polynomial-sgd-discrete-dqn', force=True)
        dr = DataRecorder(env)
        dqn.test(dr, nb_episodes=5, visualize=visualize)
        names = ["epoch", "iteration"] + env.env.observation_names() + ["reward", "done"]
        df = pd.DataFrame(dr.data_frame(names, lambda x: [x[0], x[1]] + list(x[2]) + [x[3], x[4]]))
        path = os.path.dirname(os.path.abspath(args.test_dst))
        if not os.path.exists(path):
            os.makedirs(path)
        df.to_csv(args.test_dst)
