import collections
import random

import gym
import numpy as np

from cartPoleDqn import DQN


class Agent:
    def __init__(self, env: gym.Env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.shape
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 50_000
        self.train_start = 1_000
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
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
        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                if done and total_reward < 499:
                    reward = -100.0
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state
                if done:
                    if total_reward < 500:
                        total_reward += 100.0
                    self.target_dqn.update_model(self.dqn)
                    print(f"Episode: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")
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
        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                total_reward += reward
                state = next_state
                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=200)
    input("Play?")
    agent.play(num_episodes=10, render=True)
