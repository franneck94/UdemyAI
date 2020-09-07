import gym
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


class Agent:
    # Constructor: Env, NN, Obs, Action
    def __init__(self, env):
        self.env = env
        self.observations = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n
        self.model = self.get_model()

    # Keras NN Model
    def get_model(self):
        m = Sequential()
        m.add(Dense(100, input_dim=self.observations)) # Input: State s
        m.add(Activation("relu"))
        m.add(Dense(self.actions)) # Output: Action [L, R]
        m.add(Activation("softmax"))
        m.summary()
        m.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        return m

    # Based on the state/observation, get the action
    def get_action(self, observation):
        # get an action from our policy
        observation = observation.reshape(1, -1)
        action = self.model.predict(observation)[0] # [0.9, 0.1]
        action = np.random.choice(self.actions, p=action) # [L=0, R=1], p[0.9, 0.1]
        return action

    # Sample "random" games
    def get_samples(self):
        # play some "random" games with our current policy
        pass

    # Helper function for the train function
    def filter_episodes(self):
        # filter an episode by the reward
        # x, y trainset for the NN
        pass

    # Sample random games and train the NN
    def train(self):
        # for iteration time:
        #     x,y = get_smaples()
        #     nn train(x, y)
        #     monitor performance
        pass

    # "Testing" of the Agent
    def play(self, num_episodes, render=True):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print("Episode: ", episode, " - Reward: ", total_reward)
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train()
    agent.play(num_episodes=10)
