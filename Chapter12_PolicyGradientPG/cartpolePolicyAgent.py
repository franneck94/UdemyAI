import collections
import os
from typing import Deque
from typing import List

import gym
import numpy as np

from cartpolePolicyNN import NN


PROJECT_PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyAI")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "pg_cartpole.h5")


class Agent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.num_observations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.gamma = 0.99
        self.learning_rate = 1e-3
        self.nn = NN(
            state_shape=self.num_observations,
            num_actions=self.num_actions,
            learning_rate=self.learning_rate
        )
        self.actions: List[int] = []
        self.states: List[np.ndarray] = []
        self.rewards: List[float] = []

    def get_action(self, state: np.ndarray):
        policy = self.nn(state)[0]
        action = np.random.choice(self.num_actions, p=policy)
        return action

    def discounted_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards)
        sum_t = 0.0
        for t in reversed(range(len(self.rewards))):
            sum_t = sum_t * self.gamma + self.rewards[t]
            discounted_rewards[t] = sum_t
        return discounted_rewards

    def normalize_discounted(self, discounted_rewards):
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def update_policy(self):
        episode_length = len(self.states)

        discounted_rewards = self.discounted_rewards()
        discounted_rewards = self.normalize_discounted(discounted_rewards)

        states = np.zeros(shape=(episode_length, self.num_observations))
        q_values = np.zeros(shape=(episode_length, self.num_actions))

        for i in range(episode_length):
            states[i] = self.states[i]
            q_values[i][self.actions[i]] = discounted_rewards[i]

        self.nn.fit(states, q_values)
        self.actions = []
        self.states = []
        self.rewards = []

    def train(self, num_episodes: int):
        last_rewards: Deque = collections.deque(maxlen=5)

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
                self.append_sample(state, action, reward)
                total_reward += reward
                state = next_state

                if done:
                    if total_reward < 500:
                        total_reward += 100.0
                    self.update_policy()
                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)
                    print(f"Episode: {episode} Reward: {total_reward} Mean Reward: {current_reward_mean}")
                    if current_reward_mean > 400:
                        self.nn.save_model(MODEL_PATH)
                        return
                    break
        self.nn.save_model(MODEL_PATH)

    def play(self, num_episodes: int, render: bool = True):
        self.nn.load_model(MODEL_PATH)

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
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=2_000)
    input("Play?")
    agent.play(num_episodes=15, render=True)
