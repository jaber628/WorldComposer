import gymnasium as gym
from . import Drawer, Drawer_cfg

gym.register(
    id="WorldComposer-Drawer-v0",
    entry_point="WorldComposer.tasks.Task03_Drawer.Drawer:DrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Drawer_cfg.DrawerEnvCfg,
    },
)
