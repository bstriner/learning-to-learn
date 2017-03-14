import numpy as np
import keras.backend as K
from gym.spaces import Box, MultiDiscrete #, DiscreteToMultiDiscrete


class ActionMapping(object):
    def __init__(self, k, get_params, action_space):
        self.k = k
        self.get_params = get_params
        self.action_space = action_space

    def step(self, optimizer, action):
        pass


class ActionMappingContinuousLogarithmic(ActionMapping):
    def __init__(self, k, get_params, limits, scale=1e-1):
        self.limits = limits
        self.scale = scale
        bounds = 50.0
        action_space = Box(-bounds, bounds, (k,))
        ActionMapping.__init__(self, k, get_params, action_space)

    def step(self, optimizer, action):
        reward = 0
        params = self.get_params(optimizer)
        for param, act, limit in zip(params, action, self.limits):
            p = K.get_value(param)
            pnext = np.log(p) + (act * self.scale)
            pclip = np.clip(pnext, np.log(limit[0]), np.log(limit[1]))
            reward -= np.abs(pnext - pclip)
            pvalue = np.float32(np.exp(pclip))
            if not np.all(np.isfinite(pvalue)):
                raise ValueError("Not finite param: {}->{}, action: {}".format(p, pvalue, act))
            K.set_value(param, pvalue)
        return reward


class ActionMappingDiscrete(ActionMapping):
    def __init__(self, k, get_params, limits, scale=1e-1):
        self.scale = scale
        space = MultiDiscrete([[0, 2] for _ in range(k)])
        action_space = DiscreteToMultiDiscrete(space, 'all')
        self.limits = limits
        ActionMapping.__init__(self, k, get_params, action_space)

    def step(self, optimizer, action):
        action = self.action_space(action)
        params = self.get_params(optimizer)
        reward = 0
        for param, act, limit in zip(params, action, self.limits):
            mul = 1.0 + self.scale
            if act == 0:
                scale = 1.0 / mul
            elif act == 1:
                scale = 1.0
            elif act == 2:
                scale = mul
            else:
                raise ValueError("Invalid action: {}".format(act))
            p = K.get_value(param)
            pnext = p*scale
            if pnext < limit[0] or pnext > limit[1]:
                reward -= 5
            pnext = np.float32(np.clip(pnext, limit[0], limit[1]))
            if not np.all(np.isfinite(pnext)):
                raise ValueError("Not finite param: {}->{}, action: {}".format(p, pnext, act))
            K.set_value(param, pnext)
        return reward
