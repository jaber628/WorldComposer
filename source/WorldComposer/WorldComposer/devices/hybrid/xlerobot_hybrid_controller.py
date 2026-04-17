import numpy as np
import torch
from collections.abc import Callable

from ..device_base import Device
from ..keyboard.xlerobot_keyboard import XlerobotKeyboard
from ..lerobot import BiXlerobotLeader


class XlerobotHybridController(Device):
    """Hybrid controller using keyboard for base/head and leader devices for both arms."""

    def __init__(
        self,
        env,
        sensitivity: float = 1.0,
        left_arm_port: str = "/dev/ttyACM0",
        right_arm_port: str = "/dev/ttyACM1",
        recalibrate: bool = False,
    ):
        super().__init__(env)
        self.keyboard_controller = XlerobotKeyboard(env, sensitivity)
        self.bi_arm_controller = BiXlerobotLeader(
            env,
            left_port=left_arm_port,
            right_port=right_arm_port,
            recalibrate=recalibrate,
        )
        self.control_mode = "hybrid"
        self.started = False
        self._reset_state = False

    def set_control_mode(self, mode: str):
        """Set the current control mode."""
        assert mode in ["keyboard", "hybrid", "arms_only"], f"Invalid control mode: {mode}"
        self.control_mode = mode
        print(f"Control mode switched to: {mode}")

    def get_device_state(self):
        """Return the combined controller state."""
        if self.control_mode == "keyboard":
            return self.keyboard_controller.get_device_state()

        arms_state = self.bi_arm_controller.get_device_state()
        if not hasattr(self, "_cached_arms_action"):
            self._cached_arms_action = self.bi_arm_controller.input2action()

        full_state = np.zeros(17)
        if self.control_mode == "hybrid":
            keyboard_state = self.keyboard_controller.get_device_state()
            full_state[0:3] = keyboard_state[0:3]
            full_state[15:17] = keyboard_state[15:17]

        if "left_arm" in arms_state and arms_state["left_arm"] is not None and isinstance(arms_state["left_arm"], dict):
            left_motor_limits = self._cached_arms_action.get("motor_limits", {}).get("left_arm", {})
            full_state[3:9] = self._convert_arm_action(arms_state["left_arm"], left_motor_limits)

        if "right_arm" in arms_state and arms_state["right_arm"] is not None and isinstance(arms_state["right_arm"], dict):
            right_motor_limits = self._cached_arms_action.get("motor_limits", {}).get("right_arm", {})
            full_state[9:15] = self._convert_arm_action(arms_state["right_arm"], right_motor_limits)

        return full_state

    def input2action(self):
        """Convert the active controller state into an action dictionary."""
        if self.control_mode == "keyboard":
            action = self.keyboard_controller.input2action()
            self.started = action.get("started", False)
            return action
        if self.control_mode == "arms_only":
            action = self.bi_arm_controller.input2action()
            self.started = action.get("started", False)
            self._cached_arms_action = action
            return action

        keyboard_action = self.keyboard_controller.input2action()
        arms_action = self.bi_arm_controller.input2action()
        self.started = arms_action.get("started", False)
        self._cached_arms_action = arms_action
        return {
            "reset": keyboard_action.get("reset", False) or arms_action.get("reset", False),
            "started": arms_action.get("started", False),
            "hybrid_controller": True,
            "keyboard": True,
            "bi_so101_leader": True,
            "joint_state": self.get_device_state(),
            "motor_limits": arms_action.get("motor_limits", {}),
        }

    def advance(self):
        """Return the current action tensor."""
        if not self.started:
            return None
        action = self.get_device_state()
        return torch.tensor(action, dtype=torch.float32, device=self.env.device)

    def reset(self):
        """Reset both sub-controllers."""
        self.keyboard_controller.reset()
        self.bi_arm_controller.reset()
        self.started = False
        self._reset_state = False

    def add_callback(self, key: str, func: Callable):
        """Register callbacks for hybrid control."""
        if key == "F6":
            def toggle_mode():
                if self.control_mode == "hybrid":
                    self.set_control_mode("keyboard")
                elif self.control_mode == "keyboard":
                    self.set_control_mode("arms_only")
                else:
                    self.set_control_mode("hybrid")
            self.keyboard_controller.add_callback(key, toggle_mode)
            return

        self.keyboard_controller.add_callback(key, func)
        if key == "B":
            self.bi_arm_controller.add_callback(key, func)

    def __str__(self) -> str:
        """Return a human-readable controller summary."""
        msg = "Xlerobot hybrid controller\n"
        msg += f"\tCurrent mode: {self.control_mode}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tKeyboard: base motion (W/A/S/D) and head motion (Home/End/PageUp/PageDown)\n"
        msg += "\tLeader devices: both arms and grippers\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tF6: switch control mode\n"
        msg += "\tB: start control\n"
        msg += "\tF5: reset environment\n"
        msg += "\tQuit: Ctrl+C\n"
        return msg

    def _convert_arm_action(self, joint_state: dict, motor_limits: dict) -> np.ndarray:
        """Convert a single-arm leader state into simulation joint commands."""
        processed_action = np.zeros(6)
        if not motor_limits:
            print("Warning: motor_limits not found, returning zero action.")
            return processed_action

        from WorldComposer.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS

        joint_mapping = {
            "shoulder_pan": 0,
            "shoulder_lift": 1,
            "elbow_flex": 2,
            "wrist_flex": 3,
            "wrist_roll": 4,
            "gripper": 5,
        }
        joint_direction_correction = {
            "shoulder_pan": 1,
            "shoulder_lift": -1,
            "elbow_flex": 1,
            "wrist_flex": 1,
            "wrist_roll": -1,
            "gripper": 1,
        }
        joint_zero_offset = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 1.57,
            "elbow_flex": 1.57,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

        for joint_name, index in joint_mapping.items():
            if joint_name in joint_state and joint_name in motor_limits:
                motor_limit_range = motor_limits[joint_name]
                joint_limit_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[joint_name]
                processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                    * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
                processed_radius = processed_degree / 180.0 * np.pi
                processed_radius *= joint_direction_correction.get(joint_name, 1)
                processed_action[index] = processed_radius + joint_zero_offset.get(joint_name, 0.0)

        return processed_action
