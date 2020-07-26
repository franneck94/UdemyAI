import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.utils.play import play

if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    play(env, zoom=3)