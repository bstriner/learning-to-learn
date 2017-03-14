from gym import Env
from gym.utils import seeding
from gym import spaces
import numpy as np
import keras.backend as K


class BaseEnv(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, action_mapping):
        self._seed()
        self.verbose = 0
        self.viewer = None
        self.batch_size = 32
        self.optimizer = None
        self.model = None
        self.current_step = 0
        self.action_mapping = action_mapping
        self.action_space = action_mapping.action_space
        bounds = float('inf')
        self.observation_space = spaces.Box(-bounds, bounds, (4,))
        self.viewer = None
        self.best = None
        self.evaluate_test = False
        Env.__init__(self)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def create_model(self):
        pass

    def create_optimizer(self):
        pass

    def loss_scale(self, loss):
        return -np.log(loss)

    def _step(self, action):
        reward = self.action_mapping.step(self.optimizer, action)
        loss_before = self.losses(self.data_val)
        if self.best is None:
            self.best = loss_before
        self.model.fit(self.data_train[0], self.data_train[1],
                       validation_data=(self.data_val[0], self.data_val[1]),
                       nb_epoch=1, verbose=self.verbose, batch_size=self.batch_size)
        loss_after = self.losses(self.data_val)
        self.current_step += 1
        observation = self._observation()
        if (loss_after > 1e10) or (not np.all(np.isfinite(observation))):
            print("Episode terminated due to NaN loss. Loss: {}, Obs: {}, Lr: {}".format(loss_after, observation,
                                                                                         K.get_value(
                                                                                             self.optimizer.lr)))
            observation[0] = -1
            observation[1] = -1
            reward = np.float32(-10000)
            return observation, reward, True, {}
        # reward = (self.best - loss_after)
        # eps = 1e-8
        # reward = np.float32((1.0 / (eps + loss_after)))
        reward += self.loss_scale(loss_after)
        if self.verbose:
            print("LR: {}, Reward: {}, Loss: {}".format(K.get_value(self.optimizer.lr), reward, loss_after))
        # reward = -loss_after
        assert np.all(np.isfinite(reward))
        if loss_after < self.best:
            self.best = loss_after
        done = self.current_step > self.max_steps
        # print("Step: {}".format(observation))
        info = {}
        if self.evaluate_test:
            info["test_loss"] = self.losses(self.data_test)
        return observation, reward, done, info

    def set_evaluate_test(self, evaluate_test):
        self.evaluate_test = evaluate_test

    def losses(self, data):
        loss = self.model.evaluate(data[0], data[1], verbose=self.verbose, batch_size=self.batch_size)
        return loss

    def _observation(self):
        # eps = 1e-8
        loss_train = self.loss_scale(self.losses(self.data_train))
        loss_val = self.loss_scale(self.losses(self.data_val))
        lr = K.get_value(self.optimizer.lr)
        nllr = -np.log(lr)
        ret = np.array([loss_train, loss_val, nllr, self.current_step])
        # assert np.all(np.isfinite(ret)), "Lr: {}, Inf: {}".format(lr, ret)
        return ret

    def observation_names(self):
        return ["loss_train", "loss_val", "nl_lr", "step"]

    def _reset(self):
        self.create_model()
        self.current_step = 0
        self.best = None
        observation = self._observation()
        return observation

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'human':
            print(self._observation())
        elif mode == "ansi":
            return "Observation: {}\n".format(self._observation())
        else:
            raise NotImplementedError("mode not supported: {}".format(mode))
