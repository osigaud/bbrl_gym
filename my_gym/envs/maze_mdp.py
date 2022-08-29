"""
Simple Maze MDP
"""

import logging

import gym
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
            kwargs = kwargs["kwargs"]
            width = kwargs["width"]
            height = kwargs["height"]
            if "hit" not in kwargs.keys():
                hit = False
            else:
                hit = kwargs["hit"]
            if "walls" not in kwargs.keys():
                ratio = kwargs["ratio"]
                self.mdp, nb_states = create_random_maze(width, height, ratio, hit)
            else:
                self.mdp, nb_states = build_maze(width, height, kwargs["walls"], hit)

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

    def draw_v_pi_a(self, v, policy, agent_pos, title="MDP studies"):
        self.mdp.render(v, policy, agent_pos, title)

    def draw_v_pi(self, v, policy, title="MDP studies"):
        agent_pos = None
        self.mdp.render(v, policy, agent_pos, title)

    def draw_v(self, v, title="MDP studies"):
        policy = None
        agent_pos = None
        self.mdp.render(v, policy, agent_pos, title)

    def draw_pi(self, policy, title="MDP studies"):
        v = None
        agent_pos = None
        self.mdp.render(v, policy, agent_pos, title)

    def init_draw(self, title):
        self.mdp.new_render(title)

    def render(self, mode="human"):
        pass

    def set_no_agent(self):
        self.mdp.has_state = False
