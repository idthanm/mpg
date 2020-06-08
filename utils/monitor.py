from gym.core import Wrapper
import time


class Monitor(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        self.total_steps = 0
        self.rewards = []
        self.needs_reset = False

    def reset(self, **kwargs):
        self.reset_state()
        return self.env.reset(**kwargs)

    def reset_state(self):
        if self.needs_reset:
            self.rewards = []
            self.needs_reset = False

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return ob, rew, done, info

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen}
            assert isinstance(info, dict)
            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1
