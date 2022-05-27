"""
Simple Maze MDP
"""

import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from mazemdp import create_random_maze
from mazemdp.maze import build_maze

from mazemdp.maze_plotter import show_videos
from mazemdp.mdp import Mdp

logger = logging.getLogger(__name__)


class MazeMDPEnv(gym.Env):
    def __init__(self, **kwargs):
        if kwargs == {}:
            self.mdp, nb_states = create_random_maze(10, 10, 0.2)
        else:
            kwargs = kwargs['kwargs']
            width = kwargs['width']
            height = kwargs['height']
            if 'hit' not in kwargs.keys():
                hit = False
            else:
                hit = kwargs['hit']
            if 'walls' not in kwargs.keys():
                ratio = kwargs['ratio']
                self.mdp, nb_states = create_random_maze(width, height, ratio, hit)
            else:
                self.mdp, nb_states = build_maze(width, height, kwargs['walls'], hit)
            
        self.observation_space = spaces.Discrete(nb_states)
        self.action_space = spaces.Discrete(4)

        self.seed()
        self.np_random = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        return self.mdp.step(action)

    def reset(self):
        self.mdp.reset()

    def render(self, mode="human", close=False):
        self.mdp.render()
