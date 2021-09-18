import collections
import os
import random
from typing import Deque

import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical

from frozenLakeDqn import DQN
from plotting import plotting_q_values


PROJECT_PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Coding/UdemyAI")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "dqn_frozen_lake.h5")
TARGET_MODEL_PATH = os.path.join(MODELS_PATH, "target_dqn_frozen_lake.h5")


class Agent:
    def __init__(self, env: gym.Env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.n
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 50_000
        self.train_start = 1_000
        self.memory: Deque = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-3
        self.dqn = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_dqn = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_dqn.update_model(self.dqn)
        self.batch_size = 32

    def get_action(self, state: np.ndarray):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.dqn(state))

    def train(self, num_episodes: int):
        last_rewards: Deque = collections.deque([0.0 for _ in range(5)], maxlen=5)
        best_reward_mean = 0.0

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = to_categorical(state, num_classes=self.observations)
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = to_categorical(next_state, num_classes=self.observations)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state

                if done:
                    print(f"Episode: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")
                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)

                    if current_reward_mean > best_reward_mean:
                        self.target_dqn.update_model(self.dqn)
                        best_reward_mean = current_reward_mean
                        self.dqn.save_model(MODEL_PATH)
                        self.target_dqn.save_model(TARGET_MODEL_PATH)
                        print(f"New best mean: {best_reward_mean}")

                        if current_reward_mean > 0.9:
                            return
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
        self.target_dqn.load_model(TARGET_MODEL_PATH)

        fig, ax = plt.subplots(figsize=(10, 10))
        states = np.array(
            [to_categorical(i, num_classes=self.observations) for i in range(self.observations)]
        )
        values = np.array([self.dqn(state.reshape(1, -1)) for state in states])
        values = np.squeeze(values)

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                state = to_categorical(state, num_classes=self.observations)
                state = np.reshape(state, (1, state.shape[0])).astype(np.float32)
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                state_ = state
                total_reward += reward

                if render:
                    plotting_q_values(state_, action, values, ax)
                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    agent.train(num_episodes=600)
    input("Play?")
    agent.play(num_episodes=3, render=True)
