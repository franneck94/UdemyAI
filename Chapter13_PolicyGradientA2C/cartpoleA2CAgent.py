import collections
import os
from typing import Deque

import gym
import numpy as np

from cartpoleA2CNN import Actor
from cartpoleA2CNN import Critic


PROJECT_PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyAI")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
ACTOR_PATH = os.path.join(MODELS_PATH, "critic_cartpole.h5")
CRITIC_PATH = os.path.join(MODELS_PATH, "actor_cartpole.h5")


class Agent:
    def __init__(self, env):
        self.env = env
        self.num_observations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.num_values = 1
        self.gamma = 0.99
        self.learning_rate_actor = 1e-3
        self.learning_rate_critic = 5e-3
        self.actor = Actor(
            self.num_observations,
            self.num_actions,
            self.num_values,
            self.learning_rate_actor
        )
        self.critic = Critic(
            self.num_observations,
            self.num_actions,
            self.num_values,
            self.learning_rate_critic
        )

    def get_action(self, state: np.ndarray):
        policy = self.actor(state)[0]
        action = np.random.choice(self.num_actions, p=policy)
        return action

    def update_policy(self, state, action, reward, next_state, done):
        values = np.zeros((1, self.num_values))
        advantages = np.zeros((1, self.num_actions))

        value = self.critic(state)[0]
        next_value = self.critic(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            values[0][0] = reward
        else:
            advantages[0][action] = (reward + self.gamma * next_value) - value
            values[0][0] = reward + self.gamma * next_value

        self.actor.fit(state, advantages)
        self.critic.fit(state, values)

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
                self.update_policy(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if done:
                    if total_reward < 500:
                        total_reward += 100.0
                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)
                    print(f"Episode: {episode} Reward: {total_reward} Mean Reward: {current_reward_mean}")
                    if current_reward_mean > 400:
                        self.actor.save_model(ACTOR_PATH)
                        self.critic.save_model(CRITIC_PATH)
                        return
                    break
        self.actor.save_model(ACTOR_PATH)
        self.critic.save_model(CRITIC_PATH)

    def play(self, num_episodes: int, render: bool = True):
        self.actor.load_model(ACTOR_PATH)
        self.critic.load_model(CRITIC_PATH)

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
    agent.train(num_episodes=5000)
    input("Play?")
    agent.play(num_episodes=15, render=True)
