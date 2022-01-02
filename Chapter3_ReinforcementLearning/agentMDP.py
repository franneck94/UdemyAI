STATES = ["a", "b"]
ACTIONS = ["a", "b"]
REWARDS = {"a": {"a": 0, "b": 7}, "b": {"a": -5, "b": 0}}


def main() -> None:
    state = "a"
    reward = 0
    total_reward = 0

    print(f"Start-State: {state} Start-Reward: {reward}\n\n")

    for i in range(1, 11):
        print(f"State: {state} - Iteration: {i}")
        action = input("Action: ")
        if action in ACTIONS:
            reward = REWARDS[state][action]
            total_reward += reward
            state = action
            print(f"New State: {state} Reward: {reward} Total-Reward: {total_reward}")
            i += 1


if __name__ == "__main__":
    main()
