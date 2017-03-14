from gym.envs.registration import register

register(
    id='SGD-MNIST-Discrete-v0',
    entry_point='gym_learning_to_learn.envs:MnistSgdDiscreteEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    nondeterministic=True
)

register(
    id='SGD-MNIST-Continuous-v0',
    entry_point='gym_learning_to_learn.envs:MnistSgdContinuousEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    nondeterministic=True
)

register(
    id='SGD-Polynomial-Discrete-v0',
    entry_point='gym_learning_to_learn.envs:PolynomialSgdDiscreteEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    nondeterministic=True
)

register(
    id='SGD-Polynomial-Continuous-v0',
    entry_point='gym_learning_to_learn.envs:PolynomialSgdContinuousEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    nondeterministic=True
)
register(
    id='SGD-Polynomial-Continuous-Test-v0',
    entry_point='gym_learning_to_learn.envs:PolynomialSgdContinuousTestEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    nondeterministic=True
)

