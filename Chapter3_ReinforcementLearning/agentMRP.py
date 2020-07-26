import numpy as np

states = ["a", "b"]
rewards = {"a": {"a": 0, "b": 7}, "b": {"a": -5, "b": 0}}
transitions = {"a": {"a": 0.1, "b": 0.9}, "b": {"a": 0.9, "b": 0.1}}

state = "a"
all_states = state
reward = 0
total_reward = 0
i = 0

print("\n\nStart-State: ", state, " Start-Reward: ", reward, "\n\n")

while True and i < 10:
    print("State:", state, " - Iteration: ", i)
    t = np.random.choice(len(states), p=list(transitions[state].values()))
    transition = states[t]
    reward = rewards[state][transition]
    total_reward += reward
    state = transition
    all_states += " -> " + state 
    print(all_states , "\nReward: ", reward, " Gesamt-Reward: ", total_reward, "\n")
    i += 1
    if input() != " ":
        break