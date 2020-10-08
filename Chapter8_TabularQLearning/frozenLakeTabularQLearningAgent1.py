import gym
import matplotlib.pyplot as plt
import numpy as np

from plotting import plotting_fn


class Agent:
    def __init__(self, env):
        self.env = env
        self.observations = self.env.observation_space.n
        self.actions = self.env.action_space.n
        self.S = range(self.observations)
        self.A = range(self.actions)
        self.gamma = 0.9
        self.values = {s: {a: 0.0 for a in self.A} for s in self.S}
        self.state = self.env.reset()

    def get_action(self, s_next):
        act = np.argmax(list(self.values[s_next].values()))
        return act

    def get_value(self, s_next):
        act = np.max(list(self.values[s_next].values()))
        return act

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def get_sample(self, num_episodes):
        pass

    def compute_q_values(self):
        pass

    def train(self, num_iterations):
        pass

    def test(self, num_episodes):
        self.env = gym.make("FrozenLake-v0")
        sum_rewards = 0.0
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    sum_rewards += total_reward
                    break
        return sum_rewards / num_episodes

    def play(self, num_episodes, render=True):
        fig, ax = plt.subplots()
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                if render:
                    plotting_fn(state, ax)
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print("Episode: ", episode, " - Reward: ", total_reward)
                    break


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    agent.train(num_iterations=10000)
    agent.play(num_episodes=10)
