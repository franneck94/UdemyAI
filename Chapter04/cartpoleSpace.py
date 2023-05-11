import gym


def main() -> None:
    env = gym.make("CartPole-v1")
    env.reset()

    act_space = env.action_space
    obs_space = env.observation_space

    print(f"act_space: {act_space}")
    print(f"obs_space: {obs_space}")

    act_space_n = env.action_space.n
    obs_space_low = env.observation_space.low
    obs_space_high = env.observation_space.high
    obs_space_shape = env.observation_space.shape

    print(f"act_space_n: {act_space_n}")
    print(f"obs_space_low: {obs_space_low}")
    print(f"obs_space_high: {obs_space_high}")
    print(f"obs_space_shape: {obs_space_shape}")

    for _ in range(10):
        act_sampled = env.action_space.sample()
        print(f"act_sampled: {act_sampled}")


if __name__ == "__main__":
    main()
