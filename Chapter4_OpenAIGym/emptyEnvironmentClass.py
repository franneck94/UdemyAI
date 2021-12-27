from typing import Any

import gym
import numpy as np
from gym import spaces


class CustomEnv(gym.Env):
    N_DISCRETE_ACTIONS = 10
    SHAPE = (1,)
    metadata = {'render.modes': ['human']}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.SHAPE, dtype=np.uint8)

    def step(self, action: int) -> Any:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def render(self, mode: str = 'human', close: bool = False) -> Any:
        raise NotImplementedError
