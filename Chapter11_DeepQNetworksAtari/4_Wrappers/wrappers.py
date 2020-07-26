import collections

import numpy as np

import gym

class StartGameWrapper(gym.Wrapper):
    def __init__(self, env):
        super(StartGameWrapper, self).__init__(env)
        self.env.reset()

    def reset(self, **kwargs):
        self.env.reset()
        observation, _, _, _ = self.env.step(1) # FIRE
        return observation

class FrameStack(gym.Wrapper):
    def __init__(self, env, buffer_frames):
        super(FrameStack, self).__init__(env)
        self.buffer_frames = buffer_frames
        self.frames = collections.deque(maxlen=self.buffer_frames)
        low = np.repeat(self.observation_space.low[np.newaxis, ...], self.buffer_frames, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], self.buffer_frames, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        frame_stack = np.asarray(self.frames, dtype=np.float32)
        frame_stack = np.moveaxis(frame_stack, 0, -1).reshape(1, 84, 84, -1)
        return frame_stack, reward, done, info

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        self.frames = collections.deque(maxlen=self.buffer_frames)
        frame_stack = np.zeros(shape=(1, 84, 84, 4), dtype=np.float32)
        return frame_stack

def make_env(game, buffer_frames):
    env = gym.make(game)
    env = gym.wrappers.AtariPreprocessing(
        env=env, 
        noop_max=30, 
        frame_skip=4, 
        screen_size=84, 
        terminal_on_life_loss=False, 
        grayscale_obs=True,
        scale_obs=True)
    env = FrameStack(env, buffer_frames=buffer_frames)
    env = StartGameWrapper(env)
    return env

if __name__ == "__main__":
    game = "PongNoFrameskip-v4"
    buffer_frames = 4
    env = make_env(game, buffer_frames)
