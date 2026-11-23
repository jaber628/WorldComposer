import gymnasium as gym

gym.register(
    id="WorldComposer-PourWater-v0",
    entry_point=f"{__name__}.PourWater:PourWaterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.PourWater_cfg:PourWaterEnvCfg",
    },
)
