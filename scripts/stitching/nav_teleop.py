from __future__ import annotations

import argparse
from queue import Empty, Queue
from typing import Any
from pathlib import Path

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher


from WorldComposer.stitching.utils.viz_utils import depth_to_u8

CV_WINDOW_NAME = "rgb|depth"

parser = argparse.ArgumentParser(
    description="IsaacSim nav example with keyboard teleop."
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--scene-dir", default="Assets/scenes/MultiroomScenes/MultiRoom01")
parser.add_argument("--gs-ply-path", default=None)
parser.add_argument("--steps", type=int, default=0)
parser.add_argument("--camera-height-m", type=float, default=1.4)
parser.add_argument(
    "--visualize-obs", action=argparse.BooleanOptionalAction, default=False
)
args_cli = parser.parse_args()
if hasattr(args_cli, "enable_cameras") and not args_cli.enable_cameras:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from WorldComposer.stitching.env.nav_env import (
    ACTION_MOVE_FORWARD,
    ACTION_STOP,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    NavEnv,
)
from WorldComposer.stitching.env.nav_env_cfg import NavEnvCfg


def _compose_rgb_depth(rgb: np.ndarray, depth01: np.ndarray) -> np.ndarray:
    rgb_np = np.asarray(rgb)
    if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGBA2BGR)
    elif rgb_np.ndim == 3 and rgb_np.shape[-1] == 3:
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    depth_u8 = depth_to_u8(np.asarray(depth01))
    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    if rgb_np.shape[:2] != depth_vis.shape[:2]:
        depth_vis = cv2.resize(
            depth_vis,
            (rgb_np.shape[1], rgb_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.hstack([rgb_np, depth_vis])


def _set_viewport_to_robot_camera(camera_path: str, verbose: bool = False) -> bool:
    if not camera_path:
        return False
    try:
        import omni.kit.viewport.utility as vp_util

        viewport = vp_util.get_active_viewport()
        if viewport is not None:
            current = getattr(viewport, "camera_path", "")
            if current != camera_path:
                viewport.camera_path = camera_path
                if verbose:
                    print(f"[INFO] Viewport camera set to: {camera_path}")
            return True
    except Exception:
        pass
    try:
        import omni.kit.viewport.utility as vp_util

        win = vp_util.get_active_viewport_window()
        if win is not None and hasattr(win, "viewport_api"):
            current = getattr(win.viewport_api, "camera_path", "")
            if current != camera_path:
                win.viewport_api.camera_path = camera_path
                if verbose:
                    print(f"[INFO] Viewport camera set to: {camera_path}")
            return True
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to set viewport camera: {e}")
    return False


def _init_obs_window() -> None:
    cv2.namedWindow(CV_WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass


def _show_obs_window(frame: np.ndarray) -> None:
    cv2.imshow(CV_WINDOW_NAME, frame)
    try:
        cv2.setWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.waitKey(1)


def main() -> None:
    cfg = NavEnvCfg()
    scene_dir = Path(args_cli.scene_dir).resolve()
    cfg.sim.device = args_cli.device
    cfg.gs_ply_path = str(Path(args_cli.gs_ply_path).resolve()) if args_cli.gs_ply_path else str(scene_dir / "multi_room.ply")
    cfg.visual_scene.spawn.usd_path = str(scene_dir / "default.usda")
    cfg.grid_map_path = str(scene_dir / "grid_map.png")
    cfg.grid_meta_path = str(scene_dir / "grid_meta.json")
    cfg.camera_relative_height_m = float(args_cli.camera_height_m)

    env = NavEnv(cfg)
    env.reset()
    print("=" * 60)
    print("Keyboard Teleop Controls")
    print("=" * 60)
    print("W : Move forward")
    print("A : Turn left")
    print("D : Turn right")
    print("S : Stop")
    print("Q / Esc : Quit")
    print("=" * 60)
    print(f"[INFO] Robot camera prim path: {env.camera_prim_path}")
    print(f"[INFO] Follow viewport camera prim path: {env.viewport_camera_prim_path}")
    _set_viewport_to_robot_camera(env.viewport_camera_prim_path, verbose=True)
    if args_cli.visualize_obs:
        _init_obs_window()

    action_queue: Queue[int | str] = Queue()
    current_action: int = ACTION_STOP
    keyboard_sub_id: Any = None
    quit_token = "QUIT"

    try:
        import carb.input
        from carb.input import (
            KeyboardEventType,
            KeyboardInput,
        )
        from omni.appwindow import (
            get_default_app_window,
        )

        app_window = get_default_app_window()
        keyboard = app_window.get_keyboard()
        input_iface = carb.input.acquire_input_interface()

        def _on_key(e, *args: Any) -> None:
            if e.input in [KeyboardInput.Q, KeyboardInput.ESCAPE] and e.type == KeyboardEventType.KEY_PRESS:
                action_queue.put(quit_token)
                return
            if e.input == KeyboardInput.W:
                if e.type in [KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT]:
                    action_queue.put(ACTION_MOVE_FORWARD)
                elif e.type == KeyboardEventType.KEY_RELEASE:
                    action_queue.put(ACTION_STOP)
            elif e.input == KeyboardInput.A:
                if e.type in [KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT]:
                    action_queue.put(ACTION_TURN_LEFT)
                elif e.type == KeyboardEventType.KEY_RELEASE:
                    action_queue.put(ACTION_STOP)
            elif e.input == KeyboardInput.D:
                if e.type in [KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT]:
                    action_queue.put(ACTION_TURN_RIGHT)
                elif e.type == KeyboardEventType.KEY_RELEASE:
                    action_queue.put(ACTION_STOP)
            elif e.input == KeyboardInput.S and e.type == KeyboardEventType.KEY_PRESS:
                action_queue.put(ACTION_STOP)

        keyboard_sub_id = input_iface.subscribe_to_keyboard_events(keyboard, _on_key)
    except Exception as e:
        print(f"[WARN] Keyboard subscription failed: {e}")

    try:
        frame_id = 0
        viewport_bound = False
        while simulation_app.is_running() and (args_cli.steps <= 0 or frame_id < args_cli.steps):
            if not viewport_bound and frame_id < 60 and env.viewport_camera_prim_path:
                viewport_bound = _set_viewport_to_robot_camera(
                    env.viewport_camera_prim_path, verbose=(frame_id == 0)
                )
            frame_id += 1
            try:
                while True:
                    action = action_queue.get_nowait()
                    if action == quit_token:
                        break
                    if isinstance(action, int):
                        current_action = action
            except Empty:
                action = None

            if action == quit_token:
                break
            env.step(torch.tensor([current_action], device=env.device, dtype=torch.long))

            if args_cli.visualize_obs:
                obs = env._get_observations()
                rgb = obs["rgb"]
                depth = obs["depth"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.squeeze(0).cpu().numpy()
                if isinstance(depth, torch.Tensor):
                    depth = depth.squeeze().cpu().numpy()
                _show_obs_window(
                    _compose_rgb_depth(np.asarray(rgb), np.asarray(depth))
                )

            simulation_app.update()
    finally:
        env.close()
        if keyboard_sub_id is not None:
            try:
                import carb.input
                from omni.appwindow import (
                    get_default_app_window,
                )

                input_iface = carb.input.acquire_input_interface()
                keyboard = get_default_app_window().get_keyboard()
                input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub_id)
            except Exception:
                pass
        if args_cli.visualize_obs:
            cv2.destroyAllWindows()
        simulation_app.close()


if __name__ == "__main__":
    main()
