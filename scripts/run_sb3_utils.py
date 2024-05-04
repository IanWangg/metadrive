import gymnasium as gym
import numpy as np


def register_metadrive_envs():
    from metadrive.envs import MetaDriveEnv
    from metadrive.utils.config import merge_config_with_unknown_keys


    env_names = []
    try:
        class MetaDriveEnvTut(gym.Wrapper):
            def __init__(self, config, *args, render_mode=None, **kwargs):
                # Ignore render_mode
                self._render_mode = render_mode
                super().__init__(MetaDriveEnv(config))

                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    self.action_space = gym.spaces.Discrete(int(np.prod(self.env.action_space.n)))
                else:
                    self.action_space = self.env.action_space

            def reset(self, *args, seed=None, render_mode=None, options=None, **kwargs):
                # Ignore seed and render_mode
                return self.env.reset(*args, **kwargs)

            def render(self):
                return self.env.render(mode=self._render_mode)

        def _make_env(*args, **kwargs):
            return MetaDriveEnvTut(*args, **kwargs)

        env_name = "MetaDrive-Tut-Hard-v0"
        gym.register(id=env_name, entry_point=_make_env, kwargs={"config": dict(
            use_render=False,
            start_seed=1000,
            num_scenarios=20,
            horizon=1000,
        )})
        env_names.append(env_name)

        for env_num in [1, 5, 10, 20, 50, 100]:
            env_name = "MetaDrive-Tut-{}Env-v0".format(env_num)
            gym.register(id=env_name, entry_point=_make_env, kwargs={"config": dict(
                use_render=False,
                start_seed=0,
                num_scenarios=env_num,
                horizon=1000,
            )})
            env_names.append(env_name)

        env_name = "MetaDrive-Tut-Test-v0".format(env_num)
        gym.register(id=env_name, entry_point=_make_env, kwargs={"config": dict(
            use_render=False,
            start_seed=1000,
            num_scenarios=50,
            horizon=1000,
        )})
        env_names.append(env_name)

    except gym.error.Error as e:
        print("Information when registering MetaDrive: ", e)
    else:
        print("Successfully registered MetaDrive environments: ", env_names)