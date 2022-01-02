import numpy as np


STATES = ["a", "b"]
ACTIONS = ["a", "b"]
REWARDS = {"a": {"a": 0, "b": 7}, "b": {"a": -5, "b": 0}}
TRANSITIONS = {"a": {"a": 0.1, "b": 0.9}, "b": {"a": 0.9, "b": 0.1}}


def main() -> None:
    state = "a"
    all_states = state
    reward = 0
    total_reward = 0

    print(f"Start-State: {state} Start-Reward: {reward}\n\n")

    for i in range(1, 11):
        print(f"State: {state} - Iteration: {i}")
        action = input("Action: ")
        if action in ACTIONS:
            t = np.random.choice(len(STATES), p=list(TRANSITIONS[state].values()))
            transition = STATES[t]
            reward = REWARDS[state][transition]
            total_reward += reward
            state = transition
            all_states += " -> " + state
            print(f"New State: {state} Reward: {reward} Total-Reward: {total_reward}")


if __name__ == "__main__":
    main()
