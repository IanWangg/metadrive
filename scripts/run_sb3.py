import gymnasium as gym
from metadrive.envs import MetaDriveEnv

import numpy as np

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.ppo.ppo import PPO

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from functools import partial


# stop logging
import logging
logger = logging.getLogger('MetaDrive')
logger.propagate = False


config = dict(
    env = dict(
        use_render=False,
        num_scenarios=1000,
        horizon=1000,
        traffic_density=0.1,
        random_traffic=False,
        accident_prob=0.0,
    ),
    algo = dict(
        learning_rate=5e-5,
        n_steps=200,
        batch_size=256,
        n_epochs=10,
        vf_coef=1.0,
        max_grad_norm=10.0,
        verbose=1,
        seed=0,
        ent_coef=0.0,
        tensorboard_log="./metadrive_ppo-single_scenario_per_process_1e8-tb_logs/",
    ),
    n_envs=256,
)

exptid = "original-metadrive-ppo-single_scenario_per_process_1e8-may04"

# run = wandb.init(
#         project="metaurban-rl",
#         config=config,
#         name="original-metadrive-ppo",
#         group=exptid,
#         save_code=True,  # optional
#     )

# wandb.init(project="metaurban-rl", name="original-metadrive-ppo")

env_cfg = config["env"]

def make_metadrive_env_fn(env_cfg, seed):
    env = MetaDriveEnv(dict(                     
                      start_seed=seed,
                      log_level=50,
                      **env_cfg,
                      ))
    env = Monitor(env)
    return env


def train():
    env_fn = make_metadrive_env_fn
    # env = make_vec_env(
    #     env_id=env_fn, 
    #     vec_env_cls=SubprocVecEnv, # must use SubprocVecEnv, otherwise the error will occur
    #     env_kwargs=dict(config=env_cfg),
    #     seed=0, # must pass seed to make_vec_env, otherwise the seed will be randomly generated, which will cause the error in metadrive
    #     n_envs=config["n_envs"] # seed 0 - 255
    # )

    # eval_env = make_vec_env(
    #     env_id=env_fn, 
    #     vec_env_cls=SubprocVecEnv, # must use SubprocVecEnv, otherwise the error will occur
    #     env_kwargs=dict(config=env_cfg),
    #     seed=250, # must pass seed to make_vec_env, otherwise the seed will be randomly generated, which will cause the error in metadrive
    #     n_envs=16 # seed 250 - 255
    # )

    env_cfg = config["env"]

    env = SubprocVecEnv([partial(env_fn, env_cfg, seed) for seed in range(config["n_envs"])])

    eval_env = SubprocVecEnv([partial(env_fn, env_cfg, seed) for seed in range(config["n_envs"], config["n_envs"]+16)])

    model = PPO(
        "MlpPolicy", 
        env, 
        **config["algo"],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e3),
        save_path="./ckpt_logs/",
        name_prefix=exptid,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(eval_env, best_model_save_path="./eval_logs/",
                             log_path="./eval_logs/", eval_freq=int(1e3),
                             deterministic=True, render=False)
    
    # wandb_callback = WandbCallback()

    # callbacks = CallbackList([checkpoint_callback, eval_callback, wandb_callback])
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    model.learn(
        total_timesteps=int(1e8),
        callback=callbacks,
        log_interval=4,
    )
    env.close()

if __name__ == "__main__":
    train()