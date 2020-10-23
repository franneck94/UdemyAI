import collections
import os
import random
from typing import Deque

import gym
import numpy as np

from mountainCarDqn import DQN


PROJECT_PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyAI")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "dqn_mountain_car.h5")


def reward_func(next_state, done, current_best_position):
    position, velocity = next_state
    if done and position >= 0.5: # Winning
        reward = 100
        print("REACHED GOAL!")
    elif not done and abs(position) > current_best_position: # "Good"
        reward = 10
        current_best_position = position
    elif not done: # "Okay"
        reward = abs(velocity)
    else: # "Bad"
        reward = -100.0
    return reward


class Agent:
    def __init__(self, env: gym.Env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.shape
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 75_000
        self.train_start = 5_000
        self.memory: Deque = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-3
        self.dqn = DQN(
            self.state_shape,
            self.actions,
            self.learning_rate
        )
        self.target_dqn = DQN(
            self.state_shape,
            self.actions,
            self.learning_rate
        )
        self.target_dqn.update_model(self.dqn)
        self.batch_size = 32

    def get_action(self, state: np.ndarray):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.dqn(state))

    def train(self, num_episodes: int):
        last_rewards: Deque = collections.deque(maxlen=10)
        best_reward_mean = 0.0
        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            current_best_position = -np.inf
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            while True:
                action = self.get_action(state)
                next_state, _, done, _ = self.env.step(action)
                reward = reward_func(next_state, done, current_best_position)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state
                if done:
                    self.target_dqn.update_model(self.dqn)
                    print(f"Episode: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")
                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)
                    if current_reward_mean > best_reward_mean:
                        best_reward_mean = current_reward_mean
                        self.dqn.save_model(MODEL_PATH)
                    break

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = np.concatenate(states).astype(np.float32)
        states_next = np.concatenate(states_next).astype(np.float32)

        q_values = self.dqn(states)
        q_values_next = self.target_dqn(states_next)

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i])

        self.dqn.fit(states, q_values)

    def play(self, num_episodes: int, render: bool = True):
        self.dqn.load_model(MODEL_PATH)
        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                total_reward += reward
                state = next_state
                if done:
                    print(f"Episode: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")
                    break


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    agent = Agent(env)
    agent.train(num_episodes=100)
    input("Play?")
    agent.play(num_episodes=10, render=True)
