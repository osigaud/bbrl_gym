The my_gym library is the place where I put additional gym-like environments.

So far, it contains the following environments:
- CartPoleContinuous-v0 (with timit limit = 200 steps)
- CartPoleContinuous-v1 (with timit limit = 500 steps)
- LineMDP-v0, a simple discrete state and action MDP
- LineMDPContinuous-v0, a simple discrete action MDP
- 2DMDPContinuous-v0, a discrete action MDP with 2D state
- RocketLander-v0, a rocket landing simulation adapted from [this repository](https://github.com/sdsubhajitdas/Rocket_Lander_Gym)

Besides, the gym version is forced to 0.21.0 to avoid the large changes that have appeared after version 0.22


## Installation

```
pip install -e .
```

## Use it

```
import gym
import my_gym

env = gym.make("CartPoleContinuous-v0")  # or -v1 or any other and then use your environment as usual
```
