import numpy as np


STATES = ["a", "b"]
ACTIONS = ["a", "b"]
REWARDS = {
    "a": {"a": 0, "b": 7},
    "b": {"a": -5, "b": 0}
}
TRANSITIONS = {
    "a": {"a": 0.1, "b": 0.9},
    "b": {"a": 0.9, "b": 0.1}
}


def main() -> None:
    state = "a"
    all_states = state
    reward = 0
    total_reward = 0

    print("\n\nStart-State: ", state, " Start-Reward: ", reward, "\n\n")

    for i in range(1, 11):
        print("State:", state, " - Iteration: ", i)
        action = input("Action: ")
        if action in ACTIONS:
            t = np.random.choice(len(STATES), p=list(TRANSITIONS[state].values()))
            transition = STATES[t]
            reward = REWARDS[state][transition]
            total_reward += reward
            state = transition
            all_states += " -> " + state
            print(
                all_states, "\nReward: ", reward, " Gesamt-Reward: ", total_reward, "\n"
            )


if __name__ == "__main__":
    main()
