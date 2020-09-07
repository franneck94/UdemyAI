import os
import time
import random
import collections

import numpy as np

from pongDqn import DQN
from wrappers import make_env


class Agent:
    def __init__(self, game):
        # DQN Env Variables
        self.game = game
        self.buffer_frames = 4
        self.img_shape = (84, 84, self.buffer_frames)
        self.env = make_env(self.game, self.buffer_frames)
        self.observations = self.env.observation_space.shape
        self.actions = 4
        # DQN Agent Variables
        self.replay_buffer_size = 50000
        self.train_start = 1000
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_steps = 100000
        self.epsilon_step = (self.epsilon - self.epsilon_min) / self.epsilon_steps
        # DQN Network Variables
        self.learning_rate = 1e-3
        self.model = DQN(self.img_shape, self.actions, self.learning_rate)
        self.target_model = DQN(self.img_shape, self.actions, self.learning_rate)
        self.target_model.update_model(self.model)
        self.batch_size = 32
        self.sync_models = 10000
        self.save_models = 100
        self.path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyAIKurs/data/")
        self.path_model = os.path.join(self.path, "dqn_pong.h5")
        self.path_target_model = os.path.join(self.path, "dqn_target_pong.h5")
        self.load = False
        if self.load:
            self.model.load_model(self.path_model)
            self.target_model.load_model(self.path_target_model)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.model.predict(state))

    def train(self, num_episodes):
        print("Start training for: ", self.game, " With ", self.actions,
              " Actions and ", self.img_shape, " Imgs.")
        print("Actions: ", self.env.unwrapped.get_action_meanings()[:self.actions])

        it = 0
        epsiode_it = 0
        total_rewards = []

        for episode in range(num_episodes):
            if episode % self.save_models == 0:
                self.model.save_model(self.path_model)
                self.target_model.save_model(self.path_target_model)

            total_reward = 0.0
            state = self.env.reset()
            state = np.concatenate((state, state, state, state), axis=3)
            start_time = time.time()

            while True:
                while state.shape[3] < 4:
                    state = np.concatenate((state, np.zeros((1, 84, 84, 1))), axis=3)
                action = self.get_action(state)
                it += 1
                epsiode_it += 1
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done and total_reward < 499:
                    reward = -100

                if state.shape == (1, 84, 84, 4) and next_state.shape == (1, 84, 84, 4):
                    self.remember(state, action, reward, next_state, done)
                self.epsilon_anneal()
                self.replay()
                state = next_state

                if it % self.sync_models == 0:
                    self.target_model.update_model(self.model)
                if done:
                    current_time = time.time()
                    fps = epsiode_it / (current_time - start_time)
                    epsiode_it = 0
                    total_rewards.append(total_reward)
                    mean_reward = np.mean(total_rewards[-5:])

                    print("Episode: ", episode + 1,
                          "\tSteps: ", it,
                          "\tMemSize: ", len(self.memory),
                          "\tEps: ", round(self.epsilon, 8),
                          "\tReward: ", total_reward,
                          "\tMean: ", mean_reward,
                          "\tFPS: ", round(fps))
                    self.model.save_model(self.path_model)
                    self.target_model.save_model(self.path_target_model)
                    break

    def epsilon_anneal(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_step

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

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
        self.model.load_model(self.path_model)
        self.target_model.load_model(self.path_target_model)
        for episode in range(num_episodes):
            state = self.env.reset()
            state = np.concatenate((state, state, state, state), axis=3)

            while True:
                action = self.get_action(state)
                next_state, _, done, _ = self.env.step(action)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                state = np.append(next_state, state[:, :, :, :3], axis=3)
                if render:
                    self.env.render()
                if done:
                    break


if __name__ == "__main__":
    game = "PongNoFrameskip-v4"
    agent = Agent(game)
    agent.train(num_episodes=1000)
    input("Play?")
    agent.play(num_episodes=15, render=True)
