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


# stop logging
import logging
logger = logging.getLogger('MetaDrive')
logger.propagate = False


config = dict(
    env = dict(
        use_render=False,
        start_seed=0, # must align with the seed in the argument of make_vec_env
        num_scenarios=1000,
        horizon=1000,
        traffic_density=0.1,
        random_traffic=False,
    ),
    algo = dict(
        learning_rate=3e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=10,
        vf_coef=1.0,
        max_grad_norm=1.0,
        verbose=1,
        seed=0,
        ent_coef=0.0,
        tensorboard_log="./metadrive_ppo_tb_logs/",
    ),
    n_envs=256,
)

exptid = "original-metadrive-ppo-may04"

# run = wandb.init(
#         project="metaurban-rl",
#         config=config,
#         name="original-metadrive-ppo",
#         group=exptid,
#         save_code=True,  # optional
#     )

# wandb.init(project="metaurban-rl", name="original-metadrive-ppo")

env_cfg = config["env"]

def make_metadrive_env_fn():
    return MetaDriveEnv


def train():
    env_fn = make_metadrive_env_fn()
    env = make_vec_env(
        env_id=env_fn, 
        vec_env_cls=SubprocVecEnv, # must use SubprocVecEnv, otherwise the error will occur
        env_kwargs=dict(config=env_cfg),
        seed=0, # must pass seed to make_vec_env, otherwise the seed will be randomly generated, which will cause the error in metadrive
        n_envs=config["n_envs"] # seed 0 - 255
    )

    eval_env = make_vec_env(
        env_id=env_fn, 
        vec_env_cls=SubprocVecEnv, # must use SubprocVecEnv, otherwise the error will occur
        env_kwargs=dict(config=env_cfg),
        seed=250, # must pass seed to make_vec_env, otherwise the seed will be randomly generated, which will cause the error in metadrive
        n_envs=16 # seed 250 - 255
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        **config["algo"],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e5),
        save_path="./ckpt_logs/",
        name_prefix=exptid,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(eval_env, best_model_save_path="./eval_logs/",
                             log_path="./eval_logs/", eval_freq=int(5e4),
                             deterministic=True, render=False)
    
    # wandb_callback = WandbCallback()

    # callbacks = CallbackList([checkpoint_callback, eval_callback, wandb_callback])
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    model.learn(
        total_timesteps=int(1e7),
        callback=callbacks,
    )
    env.close()

if __name__ == "__main__":
    train()