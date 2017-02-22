from keras.layers import Dense, Activation, Flatten, Dropout
from rl.agents import ContinuousDQNAgent
from rl.random import OrnsteinUhlenbeckProcess
from keras.layers import Dense, Flatten, Dropout, LeakyReLU, merge, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from keras.regularizers import l1l2


def create_V(env, args):
    V_model = Sequential()
    nch = 256
    reg = lambda: l1l2(1e-7, 1e-7)
    V_model.add(Flatten(input_shape=(args.window,) + env.observation_space.shape))
    V_model.add(Dense(nch, W_regularizer=reg()))
    V_model.add(LeakyReLU(0.2))
    V_model.add(Dense(nch / 2, W_regularizer=reg()))
    V_model.add(LeakyReLU(0.2))
    V_model.add(Dense(nch / 4, W_regularizer=reg()))
    V_model.add(LeakyReLU(0.2))
    V_model.add(Dense(1, W_regularizer=reg()))
    V_model.add(Activation('linear'))
    return V_model


def create_mu(env, args):
    mu_model = Sequential()
    nch = 256
    reg = lambda: l1l2(1e-7, 1e-7)
    mu_model.add(Flatten(input_shape=(args.window,) + env.observation_space.shape))
    mu_model.add(Dense(nch, W_regularizer=reg()))
    mu_model.add(LeakyReLU(0.2))
    mu_model.add(Dense(nch / 2, W_regularizer=reg()))
    mu_model.add(LeakyReLU(0.2))
    mu_model.add(Dense(nch / 4, W_regularizer=reg()))
    mu_model.add(LeakyReLU(0.2))
    mu_model.add(Dense(env.action_space.shape[0], W_regularizer=reg()))
    mu_model.add(Activation('linear'))
    return mu_model


def create_L(env, args):
    nb_actions = env.action_space.shape[0]
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(args.window,) + env.observation_space.shape, name='observation_input')
    nch = 256
    reg = lambda: l1l2(1e-7, 1e-7)
    x = merge([action_input, Flatten()(observation_input)], mode='concat')
    x = Dense(nch, W_regularizer=reg())(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(nch / 2, W_regularizer=reg())(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(nch / 4, W_regularizer=reg())(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(((nb_actions * nb_actions + nb_actions) / 2), W_regularizer=reg())(x)
    x = Activation('linear')(x)
    L_model = Model(input=[action_input, observation_input], output=x)
    return L_model


def create_agent_cdqn(env, args):
    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    # processor = PendulumProcessor()
    nb_actions = env.action_space.shape[0]
    V_model = create_V(env, args)
    mu_model = create_mu(env, args)
    L_model = create_L(env, args)
    memory = SequentialMemory(limit=args.memory, window_length=args.window)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
    cdqn = ContinuousDQNAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                              memory=memory, nb_steps_warmup=args.memory, random_process=random_process,
                              gamma=.99, target_model_update=1e-3)

    # , processor=processor)
    cdqn.compile(Adam(lr=1e-4, clipnorm=1.), metrics=['mae'])
    return cdqn
