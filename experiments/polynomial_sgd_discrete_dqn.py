import argparse
import numpy as np
import gym
import pandas as pd
import sys
import gym_learning_to_learn
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'SGD-Polynomial-Discrete-v0'


def main(argv):
    parser = argparse.ArgumentParser(description='Train a DQN to control hyperparameters.')
    parser.add_argument('--train', action="store_true", help='train the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    args = parser.parse_args(argv)

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    memory_limit = 5000
    window_length = 2
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=memory_limit, window_length=window_length)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=memory_limit,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
    if args.train:
        visualize = False
        history = dqn.fit(env, nb_steps=50000, visualize=visualize, verbose=2)
        pd.DataFrame(history.history).to_csv("dqn_history.csv")
        # After training is done, we save the final weights.
        dqn.save_weights(filename, overwrite=True)

    if args.test:
        # Finally, evaluate our algorithm for 5 episodes.
        dqn.load_weights(filename)
        dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main(sys.argv[1:])
