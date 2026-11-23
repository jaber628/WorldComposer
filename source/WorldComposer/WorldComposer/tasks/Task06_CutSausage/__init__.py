import gymnasium as gym

gym.register(
    id="WorldComposer-CutSausage-v0",
    entry_point=f"{__name__}.CutSausage:CutSausageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.CutSausage_cfg:CutSausageEnvCfg",
    },
)
