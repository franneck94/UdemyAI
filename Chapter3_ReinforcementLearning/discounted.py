def discounted_reward(rewards: list, gamma: float) -> float:
    val = 0.0
    episode_length = len(rewards)
    print(f"Length: {episode_length}")

    for t in range(episode_length):
        val += gamma ** (t) * rewards[t]
        print(f"Val: {gamma ** (t) * rewards[t]}")

    return val


def main() -> None:
    gamma = 0.99
    #         s0  s1  s2
    #       t: 0   1   2
    rewards = [1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    # 1 * gamma^0 + 0 * gamma^1 + 1 * gamma^2
    # 1 + 0 + 0.25 = 1.25

    discounted_reward_value = discounted_reward(rewards, gamma)
    print(f"Rewards: {rewards}")
    print(f"Discounted Reward: {discounted_reward_value}")


if __name__ == "__main__":
    main()
