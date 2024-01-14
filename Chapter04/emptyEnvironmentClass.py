from typing import Any

import gym
import numpy as np
from gym import spaces


class CustomEnv(gym.Env):  # type: ignore
    STATES = [0, 1, 2]  # noqa: RUF012
    REWARDS = {  # noqa: RUF012
        0: {0: 0, 1: 1, 2: 1},
        1: {0: 0, 1: 1, 2: 0},
        2: {0: 0, 1: 1, 2: 0},
    }

    N_DISCRETE_ACTIONS = len(STATES)
    SHAPE = (1,)
    metadata = {"render.modes": ["human"]}  # noqa: RUF012

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=self.SHAPE, dtype=np.uint8
        )
        self.state = self.STATES[0]

    def step(self, action: int) -> Any:
        if action == 0:
            reward = self.REWARDS[self.state][action]
            self.state = max(self.state - 1, 0)
        elif action == 1:
            reward = self.REWARDS[self.state][action]
            self.state = min(self.state + 1, len(self.STATES) - 1)
        else:
            reward = 0

        done = self.state == 2
        return self.state, reward, done, {}

    def reset(self) -> None:
        self.state = self.STATES[0]

    def render(self, mode: str = "human", close: bool = False) -> Any:  # noqa: ARG002
        if self.state == 0:
            print("|X _ _|")
        elif self.state == 1:
            print("|_ X _|")
        elif self.state == 2:
            print("|_ _ X|")
        else:
            print("|_ _ _|")


def main() -> None:
    env = CustomEnv()
    env.reset()

    act_space = env.action_space
    obs_space = env.observation_space

    print("Action Space: ", act_space)
    print("Observation Space: ", obs_space)

    act_space_n = env.action_space.n

    obs_space_low = env.observation_space.low
    obs_space_high = env.observation_space.high
    obs_space_shape = env.observation_space.shape

    print("Action Space N: ", act_space_n)
    print("Observation Space Low: ", obs_space_low)
    print("Observation Space High: ", obs_space_high)
    print("Observation Space Shape: ", obs_space_shape)

    while True:
        act_sample = env.action_space.sample()
        print("Sample: ", act_sample)
        _, _, done, _ = env.step(act_sample)
        env.render()
        if done:
            break


if __name__ == "__main__":
    main()
