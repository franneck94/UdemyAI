from typing import Any, Tuple

import gym
import numpy as np

from environment import Env
from environment import GraphicDisplay
from environment import GOAL


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.cols, self.rows = self.env.width, self.env.height
        self.S = self.env.all_state
        self.A = self.env.possible_actions
        self.num_actions = len(self.A)
        self.num_states = len(self.S)
        self.gamma = 0.9
        self.init_prob = 1.0 / self.num_actions
        self.policy = np.full(
            shape=(self.rows, self.cols, self.num_actions),
            fill_value=self.init_prob,
        )
        self.v_values = np.zeros(shape=(self.rows, self.cols))
        self.policy[GOAL[0], GOAL[1]] = [0.0 for _ in range(self.num_actions)]

    def get_value(self, state: Tuple[int, int]) -> Any:
        return self.v_values[state[0]][state[1]]

    def get_action(self, state: np.ndarray) -> Any:
        if state == GOAL:
            return
        policy_in_state = self.policy[state[0]][state[1]]
        action = np.random.choice(self.A, p=policy_in_state)
        return action

    def policy_evaluation(self) -> None:
        next_v_values = np.zeros(shape=(self.rows, self.cols))

        for state in self.S:
            value = 0.0
            if state == GOAL:
                continue

            for action in self.A:
                policy_in_state = self.policy[state[0]][state[1]]
                next_state, reward = self.env.step(state, action)
                next_value = self.v_values[next_state[0]][next_state[1]]
                value += policy_in_state[action] * (
                    reward + self.gamma * next_value
                )

            next_v_values[state[0]][state[1]] = round(value, 2)

        self.v_values = next_v_values

    def policy_improvement(self) -> None:
        next_policy = self.policy

        for state in self.S:
            if state == GOAL:
                continue

            temp_vals = np.zeros(shape=(self.num_actions))
            policy_update = np.zeros(shape=(self.num_actions))

            for index, action in enumerate(self.A):
                next_state, reward = self.env.step(state, action)
                next_value = self.v_values[next_state[0]][next_state[1]]
                temp_vals[index] = reward + self.gamma * next_value

            max_indicies = np.argwhere(temp_vals == np.max(temp_vals)).ravel()
            prob = 1.0 / len(max_indicies)
            for index in max_indicies:
                policy_update[index] = round(prob, 2)
            next_policy[state[0]][state[1]] = policy_update

        self.policy = next_policy


def main() -> None:
    env = Env()
    agent = Agent(env)
    grid_world = GraphicDisplay(agent)
    grid_world.mainloop()


if __name__ == "__main__":
    main()
