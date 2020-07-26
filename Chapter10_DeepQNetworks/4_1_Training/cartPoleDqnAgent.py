import random
import collections

import gym
import numpy as np

from cartPoleDqn import *

class Agent:
    def __init__(self, env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.shape
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 50000
        self.train_start = 1000
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-3
        self.model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_model.update_model(self.model)
        self.batch_size = 32

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.model.predict(state))

    def train(self, num_episodes):
        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, (1, state.shape[0]))
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if done and total_reward < 499:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state
                if done:
                    self.target_model.update_model(self.model)
                    print("Episode: ", episode+1, " Total Reward: ", total_reward, " Epsilon: ", self.epsilon)
                    break

    def remember(self, state, action, reward, next_state, done):
        pass

    def replay(self):
        pass

    def play(self):
        pass

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=1000)