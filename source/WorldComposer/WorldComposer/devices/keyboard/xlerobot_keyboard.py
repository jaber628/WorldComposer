import weakref
import numpy as np
import torch
from collections.abc import Callable

import carb
import omni
from ..device_base import Device


class XlerobotKeyboard(Device):
    """Keyboard controller for Xlerobot base, arms, grippers, and head."""

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env)
        self.sensitivity = sensitivity

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        self._create_key_bindings()

        self._base_velocity = np.zeros(3)
        self._left_arm_delta = np.zeros(5)
        self._right_arm_delta = np.zeros(5)
        self._left_gripper_delta = 0.0
        self._right_gripper_delta = 0.0
        self._head_delta = np.zeros(2)

        self.started = False
        self._reset_state = False
        self._additional_callbacks = {}
        self._pressed_keys = set()

    def _create_key_bindings(self):
        """Create keyboard mappings for all controlled joints."""
        self._key_bindings = {
            "W": ("base_joint", 0, 1.0),
            "S": ("base_joint", 0, -1.0),
            "A": ("base_joint", 1, 1.0),
            "D": ("base_joint", 1, -1.0),
            "Q": ("base_joint", 2, 1.0),
            "E": ("base_joint", 2, -1.0),
            "R": ("left_arm", 0, 1.0),
            "F": ("left_arm", 0, -1.0),
            "T": ("left_arm", 1, 1.0),
            "G": ("left_arm", 1, -1.0),
            "Y": ("left_arm", 2, 1.0),
            "H": ("left_arm", 2, -1.0),
            "U": ("left_arm", 3, 1.0),
            "J": ("left_arm", 3, -1.0),
            "I": ("left_arm", 4, 1.0),
            "K": ("left_arm", 4, -1.0),
            "NUMPAD_8": ("right_arm", 0, 1.0),
            "NUMPAD_2": ("right_arm", 0, -1.0),
            "NUMPAD_4": ("right_arm", 1, 1.0),
            "NUMPAD_6": ("right_arm", 1, -1.0),
            "NUMPAD_7": ("right_arm", 2, 1.0),
            "NUMPAD_9": ("right_arm", 2, -1.0),
            "NUMPAD_1": ("right_arm", 3, 1.0),
            "NUMPAD_3": ("right_arm", 3, -1.0),
            "NUMPAD_0": ("right_arm", 4, 1.0),
            "NUMPAD_PERIOD": ("right_arm", 4, -1.0),
            "F1": ("left_gripper", 0, 1.0),
            "F2": ("left_gripper", 0, -1.0),
            "F3": ("right_gripper", 0, 1.0),
            "F4": ("right_gripper", 0, -1.0),
            "HOME": ("head", 0, 1.0),
            "END": ("head", 0, -1.0),
            "PAGE_UP": ("head", 1, 1.0),
            "PAGE_DOWN": ("head", 1, -1.0),
            "B": ("control", "start", 1.0),
            "N": ("control", "success", 1.0),
            "F5": ("control", "reset", 1.0),
        }

    def get_device_state(self):
        """Return the current control state as a 17D action vector."""
        action = np.zeros(17)
        action[0] = self._base_velocity[0] * self.sensitivity
        action[1] = self._base_velocity[1] * self.sensitivity
        action[2] = self._base_velocity[2] * self.sensitivity
        action[3] = self._left_arm_delta[0] * self.sensitivity
        action[4] = self._left_arm_delta[1] * self.sensitivity
        action[5] = self._left_arm_delta[2] * self.sensitivity
        action[6] = self._left_arm_delta[3] * self.sensitivity
        action[7] = self._left_arm_delta[4] * self.sensitivity
        action[8] = self._left_gripper_delta * self.sensitivity
        action[9] = self._right_arm_delta[0] * self.sensitivity
        action[10] = self._right_arm_delta[1] * self.sensitivity
        action[11] = self._right_arm_delta[2] * self.sensitivity
        action[12] = self._right_arm_delta[3] * self.sensitivity
        action[13] = self._right_arm_delta[4] * self.sensitivity
        action[14] = self._right_gripper_delta * self.sensitivity
        action[15] = self._head_delta[0] * self.sensitivity
        action[16] = self._head_delta[1] * self.sensitivity
        return action

    def input2action(self):
        """Convert the current keyboard state into an action dictionary."""
        state = {}
        reset = state["reset"] = self._reset_state
        state["started"] = self.started
        if reset:
            self._reset_state = False
            return state
        state["joint_state"] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict["started"] = self.started
        ac_dict["keyboard"] = True
        if reset:
            return ac_dict
        ac_dict["joint_state"] = state["joint_state"]
        return ac_dict

    def advance(self):
        """Return the current action tensor."""
        if not self.started:
            return None
        action = self.get_device_state()
        return torch.tensor(action, dtype=torch.float32, device=self.env.device)

    def reset(self):
        """Reset controller state."""
        self._base_velocity.fill(0)
        self._left_arm_delta.fill(0)
        self._right_arm_delta.fill(0)
        self._left_gripper_delta = 0.0
        self._right_gripper_delta = 0.0
        self._head_delta.fill(0)
        self._pressed_keys.clear()

    def add_callback(self, key: str, func: Callable):
        """Register an additional callback."""
        self._additional_callbacks[key] = func

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args):
        """Process keyboard events."""
        try:
            if hasattr(event, "input") and hasattr(event.input, "name"):
                key_name = event.input.name
            elif hasattr(event, "name"):
                key_name = event.name
            else:
                return

            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if key_name in self._key_bindings:
                    control_type, index, value = self._key_bindings[key_name]
                    if control_type == "control":
                        if index == "start":
                            self.started = True
                            self._reset_state = False
                            print("Xlerobot control started.")
                        elif index == "success":
                            self.started = False
                            self._reset_state = True
                            if "N" in self._additional_callbacks:
                                self._additional_callbacks["N"]()
                        elif index == "reset":
                            self._reset_state = True
                            self.started = False
                            self.reset()
                            if "R" in self._additional_callbacks:
                                self._additional_callbacks["R"]()
                        return

                    if control_type == "base_joint":
                        self._base_velocity[index] = value
                    elif control_type == "left_arm":
                        self._left_arm_delta[index] = value
                    elif control_type == "right_arm":
                        self._right_arm_delta[index] = value
                    elif control_type == "left_gripper":
                        self._left_gripper_delta = value
                    elif control_type == "right_gripper":
                        self._right_gripper_delta = value
                    elif control_type == "head":
                        self._head_delta[index] = value

                    self._pressed_keys.add(key_name)

            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                if key_name in self._key_bindings:
                    control_type, index, value = self._key_bindings[key_name]
                    if control_type == "base_joint":
                        self._base_velocity[index] = 0.0
                    elif control_type == "left_arm":
                        self._left_arm_delta[index] = 0.0
                    elif control_type == "right_arm":
                        self._right_arm_delta[index] = 0.0
                    elif control_type == "left_gripper":
                        self._left_gripper_delta = 0.0
                    elif control_type == "right_gripper":
                        self._right_gripper_delta = 0.0
                    elif control_type == "head":
                        self._head_delta[index] = 0.0

                    self._pressed_keys.discard(key_name)

        except Exception as exc:
            print(f"Keyboard event error: {exc}")

    def __str__(self) -> str:
        """Return a human-readable controller summary."""
        msg = "Xlerobot keyboard controller\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += f"\tSensitivity: {self.sensitivity}\n"
        msg += f"\tStarted: {self.started}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tBase: W/S, A/D, Q/E\n"
        msg += "\tLeft arm: R/F, T/G, Y/H, U/J, I/K\n"
        msg += "\tRight arm: NUMPAD 8/2, 4/6, 7/9, 1/3, 0/.\n"
        msg += "\tGrippers: F1/F2 and F3/F4\n"
        msg += "\tHead: Home/End, PageUp/PageDown\n"
        msg += "\tStart: B\n"
        msg += "\tSuccess: N\n"
        msg += "\tReset: F5\n"
        msg += "\tQuit: Ctrl+C\n"
        msg += f"\tPressed keys: {list(self._pressed_keys)}"
        return msg
