def discounted_reward(rewards: list, gamma: float) -> float:
    result = 0.0
    for t in range(len(rewards)):
        result += gamma**t * rewards[t]
    return result


def main() -> None:
    gamma = 0.5  # [0, 1]
    rewards = [1, 1, 1, -1, -1, 1, -1]

    discounted_reward_value = discounted_reward(rewards, gamma)
    print(f"gamma: {gamma}")
    print(f"rewards: {rewards}")
    print(f"discounted_reward_value: {discounted_reward_value}")


if __name__ == "__main__":
    main()
