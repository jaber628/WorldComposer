import gymnasium as gym

gym.register(
    id="WorldComposer-Tableware-v0",
    entry_point=f"{__name__}.Tableware:TablewareEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Tableware_cfg:TablewareEnvCfg",
    },
)
