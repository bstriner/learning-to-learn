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

ENV_NAME = 'SGD-Polynomial-Discrete-v0'


def main(argv):
    parser = argparse.ArgumentParser(description='Train a DQN to control hyperparameters.')
    parser.add_argument('--train', action="store_true", help='train the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--retrain', action='store_true', help='retrain the model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--steps', nargs="?", default=100000, type=int, action='store',
                        help='number of steps to train')
    parser.add_argument('--memory', nargs="?", default=5000, type=int, action='store',
                        help='memory size (default=5000)')
    parser.add_argument('--dest', nargs="?", default='dqn_weights.h5', action='store',
                        help='destination file')
    args = parser.parse_args(argv)

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    verbose = 1 if args.verbose else 0
    env.env.verbose=verbose
    print("options: {}".format(args))
    memory_limit = args.memory
    window_length = 2
    # Next, we build a very simple model.
    model = Sequential()
    reg = lambda: l1l2(1e-7, 1e-7)
    model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    model.add(Dense(128, W_regularizer=reg()))
    model.add(Activation('relu'))
    model.add(Dense(64, W_regularizer=reg()))
    model.add(Activation('relu'))
    model.add(Dense(32, W_regularizer=reg()))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, W_regularizer=reg()))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=memory_limit, window_length=window_length)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=memory_limit,
                   target_model_update=1e-3, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.

    visualize = False
    if args.train:
        history = dqn.fit(env, nb_steps=args.steps, visualize=visualize, verbose=2)
        pd.DataFrame(history.history).to_csv("dqn_history.csv")
        dqn.save_weights(args.dest, overwrite=True)

    if args.retrain:
        dqn.load_weights(args.dest)
        history = dqn.fit(env, nb_steps=args.steps, visualize=visualize, verbose=2)
        pd.DataFrame(history.history).to_csv("dqn_history.csv")
        dqn.save_weights(args.dest, overwrite=True)

    if args.test:
        # Finally, evaluate our algorithm for 5 episodes.
        # env = wrappers.Monitor(env, 'output/polynomial-sgd-discrete-dqn', force=True)
        dr = DataRecorder(env)
        dqn.load_weights(args.dest)
        dqn.test(dr, nb_episodes=5, visualize=True)
        names = ["epoch", "iteration"] + env.observation_names() + ["reward", "done"]
        df = pd.DataFrame(dr.data_frame(names, lambda x: [x[0], x[1]] + list(x[2]) + [x[3], x[4]]))
        df.to_csv("polynomial-sgd-discrete-dqn.csv")


if __name__ == '__main__':
    main(sys.argv[1:])
