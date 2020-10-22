import numpy as np

from environment import Env
from environment import GraphicDisplay


class Agent:
    def __init__(self, env):
        self.env = env
        self.S = self.env.all_state
        self.A = self.env.possible_actions
        self.cols, self.rows = self.env.width, self.env.height
        self.num_actions = len(self.A)
        self.num_states = len(self.S)
        self.gamma = 0.9
        self.init_prob = 1.0 / self.num_actions
        self.policy = [[[self.init_prob for _ in range(self.num_actions)]
                        for _ in range(self.cols)] for _ in range(self.rows)]
        self.v_values = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.policy[2][2] = [0.0 for _ in range(self.num_actions)]

    def policy_evaluation(self):
        next_v_values = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]

        for state in self.S:
            value = 0.0
            if state == [2, 2]:
                next_v_values[state[0]][state[1]] = 0.0
                continue

            for action in self.A:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.v_values[next_state[0]][next_state[1]]
                value += self.policy[state[0]][state[1]][action] * (reward + self.gamma * next_value)

            next_v_values[state[0]][state[1]] = round(value, 2)

        self.v_values = next_v_values

    def policy_improvement(self):
        next_policy = self.policy

        for state in self.S:
            if state == [2, 2]:
                continue

            temp_vals = [0.0 for _ in range(self.num_actions)]
            policy_update = [0.0 for _ in range(self.num_actions)]

            for index, action in enumerate(self.A):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.v_values[next_state[0]][next_state[1]]
                temp_vals[index] = reward + self.gamma * next_value

            max_indicies = np.argwhere(temp_vals == np.max(temp_vals)).ravel()
            prob = 1.0 / len(max_indicies)
            for index in max_indicies:
                policy_update[index] = round(prob, 2)
            next_policy[state[0]][state[1]] = policy_update

        self.policy = next_policy

    def get_value(self, state):
        return self.v_values[state[0]][state[1]]

    def get_action(self, state: np.ndarray):
        if state != [2, 2]:
            action = np.random.choice(self.A, p=self.policy[state[0]][state[1]])
            return action


if __name__ == "__main__":
    env = Env()
    agent = Agent(env)
    grid_world = GraphicDisplay(agent)
    grid_world.mainloop()
