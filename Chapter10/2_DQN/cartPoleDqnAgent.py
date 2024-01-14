import collections  # noqa: F401, RUF100
import random  # noqa: F401, RUF100
from typing import Any

import gym  # noqa: F401, RUF100
import numpy as np  # noqa: F401, RUF100

from cartPoleDqn import DQN  # noqa: F401


class Agent:
    def __init__(self):
        pass

    def get_action(self) -> Any:
        pass

    def train(self):
        pass

    def remember(self):
        pass

    def replay(self) -> None:
        pass

    def play(self):
        pass
