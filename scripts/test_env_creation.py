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

    # Test the environment creation
    # try:
    #     env = make_vec_env(
    #         env_id=env_fn, 
    #         vec_env_cls=SubprocVecEnv, 
    #         env_kwargs=env_kwargs,
    #         n_envs=1
    #     )
    #     print("Creation of single MetaDrive environment is successful!")
    #     env.close()
    # except Exception as e:
    #     print("Error in creating single MetaDrive environment: ", e)
    #     exit()


    # # Test multiple environments creation
    # try:
    #     env = make_vec_env(
    #         env_id=env_fn, 
    #         vec_env_cls=SubprocVecEnv, 
    #         env_kwargs=env_kwargs,
    #         n_envs=4
    #     )
    #     print("Creation of multiple MetaDrive environments is successful!")
    #     env.close()
    # except Exception as e:
    #     print("Error in creating multiple MetaDrive environments: ", e)
    #     exit()

    # Test reset function
    try:
        env = make_vec_env(
            env_id=env_fn, 
            vec_env_cls=DummyVecEnv, 
            env_kwargs=env_kwargs,
            n_envs=1
        )
        obs, info = env.reset()
        env.reset()
        env.reset()
        env.seed
        print("Reset function in MetaDrive environment is successful!")
        print("Observation shape: ", obs.shape)
        print("Info: ", info.keys())
        env.close()
    except Exception as e:
        print("Error in reset function in MetaDrive environment: ", e)
        exit()

    # test step function
    try:
        env = gym.make("MetaDrive-Tut-Hard-v0")
        obs = env.reset()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print("Step function in MetaDrive environment is successful!")
        print("Next observation shape: ", next_obs.shape)
        print("Reward: ", reward)
        print("Done: ", done)
        print("Info: ", info)
        env.close()
    except Exception as e:
        print("Error in step function in MetaDrive environment: ", e)
        exit()

    print("All tests passed!")

    # print the action space and observation space
    print("Action space: ", env.action_space)
    print("Observation space: ", env.observation_space)