import gym
import matplotlib.pyplot as plt
import numpy as np

from plotting import plotting_fn


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observations = self.env.observation_space.n
        self.actions = self.env.action_space.n
        self.S = range(self.observations)
        self.A = range(self.actions)
        self.gamma = 0.9
        self.rewards = {s: {a: {s_next: 0 for s_next in self.S} for a in self.A} for s in self.S}
        self.transitions = {
            s: {a: {s_next: 0 for s_next in self.S} for a in self.A} for s in self.S
        }
        self.values = {s: {a: 0.0 for a in self.A} for s in self.S}

    def get_action(self, s_next):
        pass

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def get_samples(self, num_episodes: int):
        state = self.env.reset()
        for episode in range(num_episodes):
            action = self.get_random_action()
            new_state, reward, done, _ = self.env.step(action)
            self.rewards[state][action][new_state] = reward
            self.transitions[state][action][new_state] += 1
            if done:
                state = self.env.reset()
            else:
                state = new_state

    def compute_q_values(self):
        pass

    def train(self, num_iterations, num_episodes):
        self.get_samples(num_episodes=1000)
        for iteration in range(num_iterations):
            self.get_samples(num_episodes=num_episodes)
            self.compute_q_values()
            reward_mean = self.test(num_episodes=20) / 20
            # print(reward_mean)
            if reward_mean >= 0.9:
                break

    def test(self, num_episodes: int):
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
        return sum_rewards

    def play(self, num_episodes: int, render: bool = True):
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
    agent.train(num_iterations=50, num_episodes=100)
    agent.play(num_episodes=10)
