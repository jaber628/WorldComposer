import gymnasium as gym
from . import Microwave, Microwave_cfg

gym.register(
    id="WorldComposer-Microwave-v0",
    entry_point="WorldComposer.tasks.Task02_Microwave.Microwave:MicrowaveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Microwave_cfg.MicrowaveEnvCfg,
    },
)
