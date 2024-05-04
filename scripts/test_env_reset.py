import gymnasium as gym
from run_sb3_utils import register_metadrive_envs

import gymnasium as gym
from metadrive.envs import MetaDriveEnv

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv

print(sb3.__path__)

if __name__ == "__main__":
    
    # Test the environment registration
    register_metadrive_envs()

    print("Registration of MetaDrive environments is successful!")

    try:
        env = gym.make("MetaDrive-Tut-Hard-v0")
        print("Creation of single MetaDrive environment is successful!")

        env.close()
    except Exception as e:
        print("Error in creating single MetaDrive environment: ", e)
        exit()


    env_kwargs=dict(
        config=dict(
            use_render=False,
            start_seed=0,
            num_scenarios=20,
            horizon=1000,
        )
    )

    env_fn = MetaDriveEnv

    try:
        env = make_vec_env(
            env_id=env_fn, 
            vec_env_cls=SubprocVecEnv, # must use SubprocVecEnv, otherwise the error will occur
            env_kwargs=env_kwargs,
            seed=0, # must pass seed to make_vec_env, otherwise the seed will be randomly generated, which will cause the error in metadrive
            n_envs=4
        )
        obs = env.reset()
        env.reset()
        env.reset()
        env.close()
        print("Reset of multiple MetaDrive environments is successful!")
    except:
        print("Error in creating multiple MetaDrive environments!")
        exit()