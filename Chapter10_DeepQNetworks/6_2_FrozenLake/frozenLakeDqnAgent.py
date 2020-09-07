import collections
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

from plotting import plotting_q_values, save_map
from frozenLakeDqn import DQN


class Agent:
    def __init__(self, env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.n
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 2000
        self.train_start = 2000
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-3
        self.model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_model.update_model(self.model)
        self.batch_size = 16

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.model.predict(state))

    def train(self, num_episodes):
        last_rewards = collections.deque(maxlen=10)
        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()
            state = to_categorical(state, num_classes=self.observations)
            state = np.reshape(state, (1, state.shape[0]))
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = to_categorical(next_state, num_classes=self.observations)
                next_state = np.reshape(next_state, (1, next_state.shape[0]))
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state
                if done:
                    self.target_model.update_model(self.model)
                    print("Episode: ", episode+1, 
                          " Total Reward: ", total_reward, 
                          " Epsilon: ", round(self.epsilon, 3))
                    last_rewards.append(total_reward)
                    last_rewards_mean = np.mean(last_rewards)
                    if last_rewards_mean == 0.9:
                        self.model.save_model("C:/Users/Jan/Dropbox/_Programmieren/UdemyAI/Data/dqn_frozenlake.h5")
                        return
                    break

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min and len(self.memory) >= self.train_start:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = np.concatenate(states)
        states_next = np.concatenate(states_next)

        q_values = self.model.predict(states)
        q_values_next = self.target_model.predict(states_next)

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i])

        self.model.train(states, q_values)

    def play(self, num_episodes, render=True):
        self.model.load_model("C:/Users/Jan/Dropbox/_Programmieren/UdemyAI/Data/dqn_frozenlake.h5")
        fig, ax = plt.subplots(figsize=(10, 10))
        states = np.array(
            [to_categorical(i, num_classes=self.observations).reshape(1, -1) 
            for i in range(self.observations)])
        values = np.array([self.model.predict(state) for state in states])
        values = np.squeeze(values)
        save_map(values, name="dqn_frozenlake.png")
        for episode in range(num_episodes):
            state = self.env.reset()
            while True:
                state_ = state
                state = to_categorical(state, num_classes=self.observations)
                state = np.reshape(state, (1, state.shape[0]))
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                if render:
                    plotting_q_values(state_, action, values, ax)
                if done:
                    break

if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    agent.train(num_episodes=5000)
    input("Play?")
    agent.play(num_episodes=3, render=True)
