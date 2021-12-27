import tensorflow as tf


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import collections
import os
import random
from typing import Any
from typing import Deque

import numpy as np

from pongDqn import DQN
from pongDqnWrappers import make_env


PROJECT_PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Coding/UdemyAI")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "dqn_pong.h5")
TARGET_MODEL_PATH = os.path.join(MODELS_PATH, "target_dqn_pong.h5")


class Agent:
    def __init__(self, env_name: str):
        # DQN Env Variables
        self.env_name = env_name
        self.num_buffer_frames = 4
        self.env = make_env(self.env_name, self.num_buffer_frames)
        self.img_size = 84
        self.img_shape = (self.img_size, self.img_size, self.num_buffer_frames)
        self.observations = self.env.observation_space.shape
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 100_000
        self.train_start = 10_000
        self.memory: Deque = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_steps = 100_000
        self.epsilon_step = (self.epsilon - self.epsilon_min) / self.epsilon_steps
        # DQN Network Variables
        self.learning_rate = 1e-3
        self.dqn = DQN(self.img_shape, self.actions, self.learning_rate)
        self.target_dqn = DQN(self.img_shape, self.actions, self.learning_rate)
        self.target_dqn.update_model(self.dqn)
        self.batch_size = 32
        self.sync_models = 1_000

    def get_action(self, state: np.ndarray) -> Any:
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.dqn(state))

    def train(self, num_episodes: int) -> None:
        last_rewards: Deque = collections.deque(maxlen=10)
        best_reward_mean = 0.0
        frame_it = 0

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()

            while True:
                frame_it += 1
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.epsilon_anneal()
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state

                if frame_it % self.sync_models == 0:
                    self.target_dqn.update_model(self.dqn)

                if done:
                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)
                    print(
                        f"Episode: {episode} Reward: {total_reward} MeanReward: {round(current_reward_mean, 2)} "
                        f"Epsilon: {round(self.epsilon, 8)} MemSize: {len(self.memory)}"
                    )

                    if current_reward_mean > best_reward_mean:
                        best_reward_mean = current_reward_mean
                        self.dqn.save_model(MODEL_PATH)
                        self.target_dqn.save_model(TARGET_MODEL_PATH)
                        print(f"New best mean: {best_reward_mean}")
                    break

    def epsilon_anneal(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_step

    def remember(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> None:
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = np.concatenate(states)
        states_next = np.concatenate(states_next)

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

    def play(self, num_episodes: int, render: bool = True) -> None:
        self.dqn.load_model(MODEL_PATH)
        self.target_dqn.load_model(TARGET_MODEL_PATH)
        self.epsilon = 0.0

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break


if __name__ == "__main__":
    env_name = "PongNoFrameskip-v4"
    agent = Agent(env_name)
    agent.train(num_episodes=3_000)
    input("Play?")
    agent.play(num_episodes=30, render=True)
