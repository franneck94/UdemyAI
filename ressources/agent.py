import random
import collections

import gym
import numpy as np
import matplotlib.pyplot as plt
  
from nn import *
from wrappers import *
 
class Agent:
    def __init__(self, game):
        self.game = game
        self.no_ops_steps = 30
        self.buffer_frames = 4
        self.img_shape = (84, 84, self.buffer_frames)
        self.env = make_atari(self.game, max_episode_steps=4000)
        self.env = make_env(self.env, frame_stack=True, episode_life=True, clip_rewards=True)
        self.num_actions = self.env.action_space.n
        self.num_values = 1
        self.gamma = 0.99
        self.lr_actor = 1e-4
        self.lr_critic = 2e-4
        self.model = NN(self.img_shape, self.num_actions, self.num_values, self.lr_actor, self.lr_critic)
        self.states, self.actions, self.rewards = [], [], []
    
    def get_action(self, state):
        policy = self.model.predict_actor(state)[0]
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

        states = np.asarray(self.states).squeeze()
        values = self.model.predict_critic(states)
        advantages = np.zeros((episode_length,))

        for i in range(episode_length):
            advantages[i] = discounted_rewards[i] - values[i] # A = V - Q
        actions = to_categorical(self.actions, num_classes=self.num_actions)

        self.model.train_actor(states, actions, advantages)
        self.model.train_critic(states, discounted_rewards)
        self.states, self.actions, self.rewards = [], [], []

    def train(self, num_episodes):
        total_rewards = []
        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.append_sample(state, action, reward)
                total_reward += reward
                state = next_state

                if done:
                    self.update_policy()
                    total_rewards.append(total_reward)
                    if len(total_rewards) >= 10:
                        mean_total_rewards = np.mean(total_rewards[-10:])
                    else:
                        mean_total_rewards = np.mean(total_rewards)
                    print("Episode: ", episode+1,
                        " Total Reward: ", total_reward,
                        " Mean: ", mean_total_rewards)
                    if mean_total_rewards > 490:
                        return total_rewards
                    break 
        return total_rewards

    def play(self, num_episodes, render=True):
        for episode in range(num_episodes):
            state = self.env.reset()
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    break

if __name__ == "__main__":
    game = "PongNoFrameskip-v4"
    agent = Agent(game)
    total_rewards = agent.train(num_episodes=5000)

    plt.plot(range(len(total_rewards)), total_rewards, color="blue")
    plt.savefig('./agent A2C Pong.png')

    input("Play?")
    agent.play(num_episodes=15, render=True)