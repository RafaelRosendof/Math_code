from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import (AtariWrapper, ClipRewardEnv, 
                                                     EpisodicLifeEnv, FireResetEnv, 
                                                     MaxAndSkipEnv, NoopResetEnv)
import gymnasium as gym
import ale_py
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env


model = DQN.load("dqn_alien")

# Create environment matching training setup
env = make_atari_env("ALE/Alien-v5", n_envs=1, render_mode="rgb_array")
env = VecFrameStack(env, n_stack=4)  # If you used frame stacking during training

# Run the agent
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones[0]:
        obs = env.reset()