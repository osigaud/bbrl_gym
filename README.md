The my_gym library is the place where I put additional gym-like environments.

So far, it contains the following environments:
- CartPoleContinuous-v0 (with timit limit = 200 steps)
- CartPoleContinuous-v1 (with timit limit = 500 steps)
- LineMDP-v0

Besides, the gym version is forced to 0.21.0 to avoid the large changes that have appeared after version 0.22


## Installation

```
pip install -e .
```

## Use it

```
import gym
import my_gym

env = gym.make("CartPoleContinuous-v1")  # or -v0 and then use your environment as usual
```
