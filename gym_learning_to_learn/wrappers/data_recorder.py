from gym import Wrapper


class DataRecorder(Wrapper):
    def __init__(self, env):
        self.data = []
        self.epoch = 0
        self.iteration = 0
        super(DataRecorder, self).__init__(env)

    def _reset(self):
        self.epoch += 1
        self.iteration = 0
        observation = self.env.reset()
        self.data.append([self.epoch, self.iteration, observation, 0, False, {}])
        return observation

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.iteration += 1
        self.data.append([self.epoch, self.iteration, observation, reward, done, info])
        return observation, reward, done, info

    def data_frame(self, names, f):
        #ret = {k: [] for k in names}
        ret = []
        for datum in self.data:
            vals = f(datum)
            #for k, v in zip(names, vals):
            #    ret[k] = v
            ret.append({k:v for k,v in zip(names, vals)})
        return ret
