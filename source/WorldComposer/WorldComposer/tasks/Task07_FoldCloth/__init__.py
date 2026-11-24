import gymnasium as gym

gym.register(
    id="WorldComposer-FoldCloth-v0",
    entry_point=f"{__name__}.FoldCloth:FoldClothEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FoldCloth_cfg:FoldClothEnvCfg",
    },
)
