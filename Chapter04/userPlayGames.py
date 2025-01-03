import gym
from gym.utils.play import play
from gym import envs


if __name__ == "__main__":
    # Get a list of all installed envs (games)
    all_envs = envs.registry.all()
    for env in all_envs:
        print(env)

    # env = gym.make("Breakout-v0")
    env = gym.make("Pong-v4")
    play(env, zoom=3)
