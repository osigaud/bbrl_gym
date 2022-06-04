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

from mazemdp.mdp import SimpleActionSpace

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

        self.nb_states = nb_states
        self.observation_space = spaces.Discrete(nb_states)
        self.action_space = SimpleActionSpace(nactions=4)
        self.terminal_states = [nb_states]
        self.P = self.mdp.P
        self.gamma = self.mdp.gamma
        self.r = self.mdp.r

        self.seed()
        self.np_random = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        return self.mdp.step(action)

    def reset(self):
        self.mdp.reset()

    def render(self, v=None, policy=None, agent_pos=-1, title="No Title"):
        self.mdp.render(v, policy, agent_pos, title)

    def new_render(self, title):
        self.mdp.new_render(title)
