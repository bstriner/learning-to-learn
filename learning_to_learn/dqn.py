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

from keras.layers import Dense, Flatten, Dropout, LeakyReLU
from keras.models import Sequential
from keras.regularizers import l1l2
from learning_to_learn import train
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import os


def create_model(env, args):
    nb_actions = env.action_space.n
    reg = lambda: l1l2(1e-7, 1e-7)
    dropout = 0.5
    model = Sequential()
    model.add(Flatten(input_shape=(args.window,) + env.observation_space.shape))
    model.add(Dense(128, W_regularizer=reg()))
    model.add(Dropout(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Dense(64, W_regularizer=reg()))
    model.add(Dropout(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Dense(32, W_regularizer=reg()))
    model.add(Dropout(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Dense(nb_actions, W_regularizer=reg()))
    return model


def create_agent_dqn(env, args):
    model = create_model(env, args)
    memory = SequentialMemory(limit=args.memory, window_length=args.window)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=args.memory,
                   target_model_update=1e-3, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn
