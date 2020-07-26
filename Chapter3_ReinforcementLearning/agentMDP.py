import numpy as np

states = ["a", "b"]
actions = ["a", "b"]
rewards = {"a": {"a": 0, "b": 7}, "b": {"a": -5, "b": 0}}

state = "a"
reward = 0
total_reward = 0

print("\n\nStart-State: ", state, " Start-Reward: ", reward, "\n\n")

for i in range(1, 11):
    print("State:", state, " - Iteration: ", i)
    action = input("Action: ")
    if action in actions:
        reward = rewards[state][action]
        total_reward += reward
        state = action
        print("Neuer State: ", state, " Reward: ", reward, " Gesamt-Reward: ", total_reward, "\n")
        i += 1