import gym


def main() -> None:
    episodes = 100

    env = gym.make("CartPole-v1")

    for episode in range(episodes):
        env.reset()
        total_reward = 0.0

        while True:
            env.render()
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break

        print(f"Episode: {episode} Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
