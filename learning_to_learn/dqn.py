from keras.optimizers import Adam, RMSprop

from keras.layers import Dense, Flatten, Dropout, LeakyReLU
from keras.models import Sequential
from keras.regularizers import L1L2
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def create_model(env, args):

    nb_actions = env.action_space.n
    reg = lambda: l1l2(1e-7, 1e-7)
    dropout = 0.5
    nch = 512
    model = Sequential()
    model.add(Flatten(input_shape=(args.window,) + env.observation_space.shape))
    model.add(Dense(nch, W_regularizer=reg()))
    #model.add(LayerNormalization())
    #model.add(Dropout(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Dense(nch, W_regularizer=reg()))
    #model.add(LayerNormalization())
    #model.add(Dropout(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Dense(nch, W_regularizer=reg()))
    #model.add(LayerNormalization())
    #model.add(Dropout(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Dense(nb_actions, W_regularizer=reg()))
    return model


def create_agent_dqn(env, args):
    model = create_model(env, args)
    memory = SequentialMemory(limit=args.memory, window_length=args.window)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=args.memory,
                   target_model_update=1e-4, policy=policy)
    dqn.compile(RMSprop(lr=1e-4), metrics=['mae'])
    return dqn
