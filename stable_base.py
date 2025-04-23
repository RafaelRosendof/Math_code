from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import (AtariWrapper, ClipRewardEnv, 
                                                     EpisodicLifeEnv, FireResetEnv, 
                                                     MaxAndSkipEnv, NoopResetEnv)
import gymnasium as gym
import ale_py

# Create the environment with Atari-specific wrappers
def make_env():
    env = gym.make("ALE/Alien-v5", render_mode="rgb_array")
    # Apply standard Atari wrappers
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = AtariWrapper(env)
    return env

env = make_env()

# Hyperparameters optimized for Atari games
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./alien_dqn/",
    buffer_size=100_000,  # Replay buffer size
    learning_starts=50_000,  # How many steps to collect before training starts
    batch_size=32,  # Batch size for training
    learning_rate=1e-4,  # Learning rate
    gamma=0.99,  # Discount factor
    target_update_interval=10_000,  # Update target network every X steps
    train_freq=4,  # Update the model every X steps
    gradient_steps=1,  # How many gradient steps after each rollout
    exploration_fraction=0.1,  # Fraction of total timesteps for exploration
    exploration_final_eps=0.01,  # Final exploration epsilon value
    device="auto",  # Use GPU if available
)

# Train for a more reasonable number of timesteps (1M is common for Atari)
model.learn(total_timesteps=1_000_000, progress_bar=True)

# Save the model
model.save("dqn_alien")

# Close the environment when done
env.close()