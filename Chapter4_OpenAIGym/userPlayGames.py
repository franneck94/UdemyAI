import gym
from gym.utils.play import play


if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    # env = gym.make('Pong-v4')
    play(env, zoom=3)
