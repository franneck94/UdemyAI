import gym
import matplotlib.pyplot as plt
import numpy as np

from plotting import *


class Agent:
    def __init__(self, env):
        self.env = env
        self.observations = self.env.observation_space.n
        self.actions = self.env.action_space.n
        self.S = range(self.observations)
        self.A = range(self.actions)
        self.gamma = 0.9
        self.rewards = {s: {a: {s_next: 0 for s_next in self.S} for a in self.A} for s in self.S}
        self.transitions = {s: {a: {s_next: 0 for s_next in self.S} for a in self.A} for s in self.S}
        self.values = {s: {a: 0.0 for a in self.A} for s in self.S}
        self.state = self.env.reset()

    def get_action(self, s_next):
        act = np.argmax(list(self.values[s_next].values()))
        return act

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def get_samples(self, num_episodes):
        for _ in range(num_episodes):
            action = self.get_random_action()
            new_state, reward, done, _ = self.env.step(action)
            self.rewards[self.state][action][new_state] = reward
            self.transitions[self.state][action][new_state] += 1
            if done:
                self.state = self.env.reset()
            else:
                self.state = new_state

    def compute_q_values(self):
        for s in self.S:
            for a in self.A:
                q_value = 0.0
                transitions_s = self.transitions[s][a] # s=0,  {1: 3, 2: 4}
                total_counts = sum(transitions_s.values()) # sum([3, 4]) = 7
                if total_counts > 0:
                    for s_next, count in transitions_s.items():
                        reward = self.rewards[s][a][s_next]
                        best_action = self.get_action(s_next)
                        q_value += (count / total_counts) * (reward + self.gamma * self.values[s_next][best_action])
                    self.values[s][a] = q_value

    def train(self, num_iterations, num_episodes):
        self.get_samples(num_episodes=1000)
        for _ in range(num_iterations):
            self.get_samples(num_episodes=num_episodes)
            self.compute_q_values()
            reward_mean = self.test(num_episodes=20) / 20
            # print(reward_mean)
            if reward_mean >= 0.9:
                break

    def test(self, num_episodes):
        sum_rewards = 0.0
        for _ in range(num_episodes):
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

    def play(self, num_episodes, render=True):
        fig, ax = plt.subplots(figsize=(10, 10))
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                action = self.get_action(state)
                if render:
                    plotting_q_values(state, action, self.values, ax)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print("Episode: ", episode, " - Reward: ", total_reward)
                    break


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    agent.train(num_iterations=10000, num_episodes=1000)
    agent.play(num_episodes=20)
    save_map(agent.values, name="viaq.png")
