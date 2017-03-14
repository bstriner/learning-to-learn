import gym
import gym_learning_to_learn
from gym.wrappers import Monitor
import gym
import time
from tqdm import tqdm

monitor = False
env = gym.make('MNIST-SGD-v0')
if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)
for i_episode in tqdm(range(500)):
    observation = env.reset()
    for t in tqdm(range(env.max_steps)):
        # env.render()
        # print(observation)
        #action = env.action_space.sample()
        action = 0
        # time.sleep(1)
        observation, reward, done, info = env.step(action)
        print "Reward: {}".format(reward)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
