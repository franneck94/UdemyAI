import gym

env = gym.make("CartPole-v1")
episodes = 100

for episode in range(episodes):
    state = env.reset()
    total_reward = 0.0

    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done == True:
            break

    print("Episode: ", episode+1, " Total Reward: ", total_reward)