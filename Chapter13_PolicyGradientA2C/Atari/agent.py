import time
import random
import collections

import gym
import numpy as np
import matplotlib.pyplot as plt
        
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from nn import *
from wrappers import *

class Agent:
    def __init__(self, game):
        self.game = game
        self.env = make_env(game)
        self.img_shape = (84, 84, 4)
        self.num_actions = 3
        self.num_values = 1
        self.gamma = 0.99
        self.lr_actor = 1e-3
        self.lr_critic = 1e-4
        self.model = NN(self.img_shape, self.num_actions, self.num_values, self.lr_actor, self.lr_critic)

    def get_action(self, state):
        policy = self.model.predict_actor(state)[0]
        action = np.random.choice(self.num_actions, p=policy)
        return action

    def update_policy(self, state, action, reward, next_state, done):
        values = np.zeros((1, self.num_values))
        advantages = np.zeros((1, self.num_actions))

        value = self.model.predict_critic(state)[0]
        next_value = self.model.predict_critic(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            values[0][0] = reward
        else:
            advantages[0][action] = (reward + self.gamma * next_value) - value
            values[0][0] = reward + self.gamma * next_value

        self.model.train_actor(state, advantages)
        self.model.train_critic(state, values)

    def train(self, num_episodes):
        print("Start training for: ", self.game, 
            " With ", self.num_actions, 
            " Actions and ", self.img_shape, " Imgs.")
        print("Actions: ", self.env.unwrapped.get_action_meanings()[1:self.num_actions+1])

        it = 0
        for episode in range(num_episodes):

            done = False
            episode_it = 0
            total_reward = 0.0
            state = self.env.reset()
            start_time = time.time()

            state = np.concatenate((state, state, state, state), axis=3)

            while True:
                action = self.get_action(state)

                it += 1
                episode_it += 1

                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_state = np.append(next_state, state[:, :, :, :3], axis=3)

                self.update_policy(state, action, reward, next_state, done)

                state = next_state

                if done:
                    current_time = time.time()
                    fps = episode_it / (current_time - start_time)
                    episode_it = 0
                    
                    print("Episode: ", episode+1,
                            "\tSteps: ", it,
                            "\tReward: ", total_reward,
                            "\tFPS: ", round(fps))
                    break


    def play(self, num_episodes, render=True):
        for episode in range(1, num_episodes+1):
            done = False
            total_reward = 0.0
            state = self.env.reset()

            state = np.concatenate((state, state, state, state), axis=3)

            while True:
                action = self.get_action(state)

                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_state = np.append(next_state, state[:, :, :, :3], axis=3)
                state = next_state

                if done:
                    break

if __name__ == "__main__":
    game = "PongNoFrameskip-v4"
    agent = Agent(game)
    agent.train(num_episodes=5000)
    input("Play?")
    agent.play(num_episodes=15, render=True)