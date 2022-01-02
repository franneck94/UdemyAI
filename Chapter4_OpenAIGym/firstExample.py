import gym


def main() -> None:
    env = gym.make("CartPole-v0")
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
    env.close()


if __name__ == "__main__":
    main()
