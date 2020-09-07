import gym
import numpy as np

from dqn import DQN


class Agent:
    def __init__(self, env):
        self.env = env
        self.num_observations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.gamma = 0.99
        self.learning_rate = 1e-2
        self.model = DQN(self.num_observations, self.num_actions, self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        policy = self.model.predict(state)[0]
        action = np.random.choice(self.num_actions, p=policy)
        return action

    def discounted_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards)
        val = 0.0
        for t in reversed(range(len(self.rewards))):
            val = val * self.gamma + self.rewards[t]
            discounted_rewards[t] = val
        return discounted_rewards

    def normalize_discounted(self, discounted_rewards):
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def train(self, num_episodes):

        for episode in range(num_episodes):

            while True:

                if done:

    def play(self, num_episodes):

        for episode in range(num_episodes):

            while True:

                if done:


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=1000)
    input("Play?")
    agent.play(num_episodes=15, render=True)
