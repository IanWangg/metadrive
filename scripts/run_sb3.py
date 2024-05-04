import gymnasium as gym

import gymnasium as gym
from metadrive.envs import MetaDriveEnv

import numpy as np

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv

from stable_baselines3.ppo.ppo import PPO

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


# model = PPO(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}")
# model.learn(
#     total_timesteps=config["total_timesteps"],
#     callback=WandbCallback(
#         model_save_path=f"models/{run.id}",
#         verbose=2,
#     ),
# )
# run.finish()

env_cfg = dict(
    use_render=False,
    start_seed=0, # must align with the seed in the argument of make_vec_env
    num_scenarios=20,
    horizon=1000,
)

def make_metadrive_env_fn():
    return MetaDriveEnv


def train():
    env_fn = make_metadrive_env_fn()
    env = make_vec_env(
        env_id=env_fn, 
        vec_env_cls=SubprocVecEnv, # must use SubprocVecEnv, otherwise the error will occur
        env_kwargs=dict(config=env_cfg),
        seed=0, # must pass seed to make_vec_env, otherwise the seed will be randomly generated, which will cause the error in metadrive
        n_envs=2
    )

    eval_env = make_vec_env(
        env_id=env_fn, 
        vec_env_cls=SubprocVecEnv, # must use SubprocVecEnv, otherwise the error will occur
        env_kwargs=dict(config=env_cfg),
        seed=0, # must pass seed to make_vec_env, otherwise the seed will be randomly generated, which will cause the error in metadrive
        n_envs=2
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=1e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        vf_coef=1.0,
        max_grad_norm=1.0,
        verbose=1,
        seed=0,
        ent_coef=0.001,
        tensorboard_log="./tb_logs/",
    )

    checkpoint_callback = checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./logs/",
        name_prefix="regular_metadrive",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)
    
    run = wandb.init(
        project="sb3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    model.learn(
        total_timesteps=1000,
        callback=callbacks,
    )
    env.close()

if __name__ == "__main__":
    train()