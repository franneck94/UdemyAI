import collections
import os
import random
from typing import Any
from typing import Deque

import gym
import numpy as np

from cartpoleA2CNN import Actor
from cartpoleA2CNN import Critic


PROJECT_PATH = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyAI")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
ACTOR_PATH = os.path.join(MODELS_PATH, "actor_cartpole.h5")
CRITIC_PATH = os.path.join(MODELS_PATH, "critic_cartpole.h5")


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.num_observations = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.num_values = 1
        self.gamma = 0.95
        self.learning_rate_actor = 1e-3  # 0.001
        self.learning_rate_critic = 5e-3  # 0.005
        self.actor = Actor(
            self.num_observations, self.num_actions, self.learning_rate_actor
        )
        self.critic = Critic(
            self.num_observations, self.num_values, self.learning_rate_critic
        )

    def get_action(self, state: np.ndarray) -> Any:
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.dqn(state))

    def train(self, num_episodes: int) -> None:
        last_rewards: Deque = collections.deque(maxlen=5)
        best_reward_mean = 0.0

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(
                    np.float32
                )
                if done and total_reward < 499:
                    reward = -100.0
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state

                if done:
                    if total_reward < 500:
                        total_reward += 100.0
                    print(
                        f"Episode: {episode} "
                        f"Reward: {total_reward} "
                        f"Epsilon: {self.epsilon}"
                    )
                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)

                    if current_reward_mean > best_reward_mean:
                        best_reward_mean = current_reward_mean
                        self.actor.save_model(ACTOR_PATH)
                        self.critic.save_model(CRITIC_PATH)
                        print(f"New best mean: {best_reward_mean}")

                        if best_reward_mean > 400:
                            return
                    break

    def remember(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self) -> None:
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
                q_values[i][a] = rewards[i] + self.gamma * np.max(
                    q_values_next[i]
                )

        self.dqn.fit(states, q_values)

    def play(self, num_episodes: int, render: bool = True) -> None:
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
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(
                    np.float32
                )
                total_reward += reward
                state = next_state

                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=250)
    input("Play?")
    agent.play(num_episodes=20, render=True)
