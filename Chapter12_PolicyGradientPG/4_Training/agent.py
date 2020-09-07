import gym
import numpy as np

from dqn import DQN


class Agent:
    def __init__(self, env):
        self.env = env
        self.num_observations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.gamma = 0.99
        self.learning_rate = 1e-3
        self.model = DQN(self.num_observations, self.num_actions, self.learning_rate)
        self.actions, self.states, self.rewards = [], [], []

    def get_action(self, state):
        policy = self.model.predict(state)[0]
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

        states = np.zeros((episode_length, self.num_observations))
        q_values = np.zeros((episode_length, self.num_actions))

        for i in range(episode_length):
            states[i] = self.states[i]
            q_values[i][self.actions[i]] = discounted_rewards[i]

        self.model.train(states, q_values)
        self.states, self.actions, self.rewards = [], [], []

    def train(self, num_episodes):
        total_rewards = []
        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()

            while True:
                state = np.reshape(state, (1, self.num_observations))
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward = reward if not done or total_reward == 499 else -100
                self.append_sample(state, action, reward)
                total_reward += reward
                state = next_state

                if done:
                    self.update_policy()
                    total_reward = total_reward if total_reward == 500 else total_reward + 100
                    total_rewards.append(total_reward)
                    if len(total_rewards) > 10:
                        mean_total_rewards = np.mean(total_rewards[-10])
                    else:
                        mean_total_rewards = np.mean(total_rewards)
                    print("Episode: ", episode + 1,
                          " Total Reward: ", total_reward,
                          " Mean: ", mean_total_rewards)
                    if mean_total_rewards > 490:
                        return
                    break

    def play(self, num_episodes, render):
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
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=1000)
    input("Play?")
    agent.play(num_episodes=15, render=True)
