import gymnasium as gym

gym.register(
    id="WorldComposer-AssembleBurger-v0",
    entry_point=f"{__name__}.AssembleBurger:AssembleBurgerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.AssembleBurger_cfg:AssembleBurgerEnvCfg",
    },
)
