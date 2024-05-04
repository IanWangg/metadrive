import gymnasium as gym
from run_sb3_utils import register_metadrive_envs

import gymnasium as gym
from metadrive.envs import MetaDriveEnv

import numpy as np

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv


print(sb3.__path__)

if __name__ == "__main__":

    env_kwargs=dict(
        config=dict(
            use_render=False,
            start_seed=0,
            num_scenarios=20,
            horizon=1000,
        )
    )

    env_fn = MetaDriveEnv

    # test step function
    try:
        env = make_vec_env(
            env_id=env_fn, 
            vec_env_cls=SubprocVecEnv, # must use SubprocVecEnv, otherwise the error will occur
            env_kwargs=env_kwargs,
            seed=0, # must pass seed to make_vec_env, otherwise the seed will be randomly generated, which will cause the error in metadrive
            n_envs=4
        )
        obs = env.reset()
        action = env.action_space.sample()
        print("Observation shape: ", obs.shape, " Should be 4 x")
        print("Action shape: ", action.shape, " Should be 4 x")

        action = action.reshape(1, -1).repeat(4, axis=0)

        next_obs, reward, done, info = env.step(action)
        print("Step function in MetaDrive environment is successful!")
        print("Next observation shape: ", next_obs.shape)
        print("Reward: ", reward.shape)
        print("Done: ", done.shape)
        print("Info: ", len(info)) # info is a list of dictionaries
        for info_ele in info:
            print("Info: ", type(info_ele))
        env.close()
    except Exception as e:
        print("Error in step function in MetaDrive environment: ", e)
        exit()

    print("All tests passed!")

    # print the action space and observation space
    print("Action space: ", env.action_space)
    print("Observation space: ", env.observation_space)