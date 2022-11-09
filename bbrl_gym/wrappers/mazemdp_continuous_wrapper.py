import gym
import random
import numpy as np


class MazeMDPContinuousWrapper(gym.Wrapper):
    """
    Specific wrapper to shape the reward of the rocket lander environment
    """

    def __init__(self, env):
        super(MazeMDPContinuousWrapper, self).__init__(env)
        high = np.array(
            [
                env.coord_x.max()[0],
                env.coord_y.max()[0],
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                env.coord_x.min()[0],
                env.coord_y.min()[0],
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low, high)

    def is_continuous_state():
        return True

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        x = self.env.coord_x[next_state]
        y = self.env.coord_x[next_state]
        xc = x + random.random()
        yc = y + random.random()
        next_continuous = [xc, yc]
        return next_continuous, reward, done, info
