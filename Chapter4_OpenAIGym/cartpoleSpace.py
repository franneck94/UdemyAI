import gym


env = gym.make("CartPole-v1")
env.reset()

act_space = env.action_space
obs_space = env.observation_space

print("Action Space: ", act_space)
print("Observation Space: ", obs_space)

act_space_n = env.action_space.n

obs_space_low = env.observation_space.low
obs_space_high = env.observation_space.high
obs_space_shape = env.observation_space.shape

print("Action Space N: ", act_space_n)
print("Observation Space Low: ", obs_space_low)
print("Observation Space High: ", obs_space_high)
print("Observation Space Shape: ", obs_space_shape)

for i in range(10):
    act_sample = env.action_space.sample()
    print("Sample: ", act_sample)
