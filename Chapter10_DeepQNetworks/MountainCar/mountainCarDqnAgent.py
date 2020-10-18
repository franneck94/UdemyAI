import collections
import random

import gym
import numpy as np

from mountainCarDqn import DQN


class Agent:
    def __init__(self, env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.shape
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 5000
        self.train_start = 5000
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-4
        self.model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_model.update_model(self.model)
        self.batch_size = 32

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.model(state))

    def train(self, num_episodes):
        best_total_reward = 0.0
        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, (1, state.shape[0]))
            current_best_position = state[0][0]
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, next_state.shape[0]))
                position = next_state[0][0]
                if done and position >= 0.5: # Winning
                    reward = 100
                    print("\nREACHED GOAL!\n")
                elif not done and position > current_best_position: # "Good"
                    reward = 10
                    current_best_position = position
                else: # "Bad"
                    reward = -1
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state
                if episode == 0:
                    best_total_reward = total_reward
                if done:
                    self.target_model.update_model(self.model)
                    print("Episode: ", episode + 1,
                          " Total Reward: ", total_reward,
                          " Epsilon: ", round(self.epsilon, 3))
                    if total_reward > best_total_reward:
                        self.model.save_model("C:/Users/Jan/Dropbox/_Programmieren/UdemyAI/data/dqn_mountaincar.h5")
                        best_total_reward = total_reward
                        print("NEW BEST REWARD: ", best_total_reward)
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

        q_values = self.model(states)
        q_values_next = self.target_model(states_next)

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i])

        self.model.train(states, q_values)

    def play(self, num_episodes, render=True):
        self.model.load_model("C:/Users/Jan/Dropbox/_Programmieren/UdemyAI/data/dqn_mountaincar.h5")
        for episode in range(num_episodes):
            state = self.env.reset()
            state = np.reshape(state, (1, state.shape[0]))
            while True:
                action = self.get_action(state)
                next_state, _, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, next_state.shape[0]))
                state = next_state
                if render:
                    self.env.render()
                if done:
                    break


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    agent = Agent(env)
    agent.train(num_episodes=500)
    input("Play?")
    agent.play(num_episodes=10, render=True)
