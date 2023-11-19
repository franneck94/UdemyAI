from typing import Any

import gym
import numpy as np

from environment import GOAL
from environment import Env
from environment import GraphicDisplay


class Agent:
    def __init__(self, env: Env) -> None:
        self.env = env
        self.rows, self.cols = self.env.width, self.env.height
        self.S = self.env.all_state
        self.A = self.env.possible_actions
        self.num_states = len(self.S)
        self.num_actions = len(self.A)
        self.gamma = 0.9
        self.init_prob = 1.0 / self.num_actions
        self.policy = np.full(
            shape=(self.rows, self.cols, self.num_actions),
            fill_value=self.init_prob,
        )
        self.v_values = np.zeros(shape=(self.rows, self.cols))

    def get_value(self, state: tuple[int, int]) -> Any:
        pass

    def get_action(self, state: tuple[int, int]) -> Any:
        pass

    def policy_evaluation(self) -> None:
        pass

    def policy_improvement(self) -> None:
        pass


def main() -> None:
    env = Env()
    agent = Agent(env)
    grid_world = GraphicDisplay(agent)
    grid_world.mainloop()


if __name__ == "__main__":
    main()
