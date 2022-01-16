import numpy as np
import gym3
from procgen import ProcgenGym3Env

def get_procgen_venv(*, env_id, num_envs, distribution_mode, start_level, num_levels, rendering=False, **env_kwargs):
    if rendering:
        env_kwargs["render_human"] = True
    env = ProcgenGym3Env(num=num_envs, env_name=env_id, \
        distribution_mode=distribution_mode, start_level=start_level, \
        num_levels=num_levels, **env_kwargs)

    env = gym3.ExtractDictObWrapper(env, "rgb")

    if rendering:
        env = gym3.ViewerWrapper(env, info_key="rgb")
    return env

def get_venv(num_envs, env_name, distribution_mode, start_level, num_levels, **env_kwargs):
    venv = get_procgen_venv(num_envs=num_envs, env_id=env_name, \
        distribution_mode=distribution_mode, start_level=start_level, \
        num_levels=num_levels, **env_kwargs)

    return venv