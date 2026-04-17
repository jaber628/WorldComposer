import torch
from typing import Any

import isaaclab.envs.mdp as mdp

XLEROBOT_JOINT_LIMITS = {
    "root_x_axis_joint": (-20.0, 20.0),
    "root_z_rotation_joint": (-3.14159, 3.14159),
    "Rotation": (-2.1, 2.1),
    "Pitch": (-0.1, 3.45),
    "Elbow": (-0.2, 3.14159),
    "Wrist_Pitch": (-1.8, 1.8),
    "Wrist_Roll": (-3.14159, 3.14159),
    "Jaw": (-0.5, 0.5),
    "Rotation_2": (-2.1, 2.1),
    "Pitch_2": (-0.1, 3.45),
    "Elbow_2": (-0.2, 3.14159),
    "Wrist_Pitch_2": (-1.8, 1.8),
    "Wrist_Roll_2": (-3.14159, 3.14159),
    "Jaw_2": (-0.5, 0.5),
    "head_pan_joint": (-1.57, 1.57),
    "head_tilt_joint": (-0.76, 1.45),
}


def init_xlerobot_action_cfg(action_cfg, device):
    """Initialize action configuration for Xlerobot teleoperation."""
    if device in ["keyboard"]:
        action_cfg.left_arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=3.0,
        )
        action_cfg.left_gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            scale=2.0,
        )
        action_cfg.right_arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"],
            scale=3.0,
        )
        action_cfg.right_gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw_2"],
            scale=2.0,
        )
        action_cfg.head_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_pan_joint", "head_tilt_joint"],
            scale=2.0,
        )
        action_cfg.base_action = None
    elif device in ["xlerobot_leader"]:
        action_cfg.base_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["root_x_axis_joint", "root_z_rotation_joint"],
            scale=1.0,
        )
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            scale=1.0,
        )
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw_2"],
            scale=1.0,
        )
        action_cfg.head_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_pan_joint", "head_tilt_joint"],
            scale=1.0,
        )
    elif device in ["xbox", "gamepad"]:
        action_cfg.unified_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "root_x_axis_joint", "root_z_rotation_joint",
                "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw",
                "Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2", "Jaw_2",
                "head_pan_joint", "head_tilt_joint",
            ],
            scale=1.0,
        )
        action_cfg.base_action = None
        action_cfg.left_arm_action = None
        action_cfg.left_gripper_action = None
        action_cfg.right_arm_action = None
        action_cfg.right_gripper_action = None
        action_cfg.head_action = None
    else:
        action_cfg.base_action = None
        action_cfg.left_arm_action = None
        action_cfg.left_gripper_action = None
        action_cfg.right_arm_action = None
        action_cfg.right_gripper_action = None
        action_cfg.head_action = None
    return action_cfg


xlerobot_joint_names_to_motor_ids = {
    "root_z_rotation_joint": 0,
    "Rotation": 1,
    "Pitch": 2,
    "Elbow": 3,
    "Wrist_Pitch": 4,
    "Wrist_Roll": 5,
    "Jaw": 6,
    "Rotation_2": 7,
    "Pitch_2": 8,
    "Elbow_2": 9,
    "Wrist_Pitch_2": 10,
    "Wrist_Roll_2": 11,
    "Jaw_2": 12,
    "head_pan_joint": 13,
    "head_tilt_joint": 14,
}


def convert_action_from_xlerobot_leader(joint_state: dict[str, float], motor_limits: dict[str, tuple[float, float]], teleop_device) -> torch.Tensor:
    """Convert Xlerobot leader readings into simulation joint actions."""
    processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)

    for joint_name, motor_id in xlerobot_joint_names_to_motor_ids.items():
        if joint_name in joint_state and joint_name in motor_limits:
            motor_limit_range = motor_limits[joint_name]
            joint_limit_range = XLEROBOT_JOINT_LIMITS[joint_name]
            processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
            if joint_name in ["root_x_axis_joint", "root_y_axis_joint"]:
                processed_action[:, motor_id] = processed_degree
            else:
                processed_action[:, motor_id] = processed_degree / 180.0 * torch.pi

    return processed_action


def preprocess_xlerobot_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    """Preprocess Xlerobot teleoperation actions."""
    if action.get("hybrid_controller") is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action["joint_state"]
    elif action.get("xlerobot_leader") is not None:
        processed_action = convert_action_from_xlerobot_leader(action["joint_state"], action["motor_limits"], teleop_device)
    elif action.get("keyboard") is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action["joint_state"]
    elif action.get("xbox") is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action["joint_state"]
    elif action.get("bi_xlerobot_leader") is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = convert_action_from_xlerobot_leader(action["joint_state"], action["motor_limits"], teleop_device)
    else:
        raise NotImplementedError(
            "Only teleoperation with xlerobot_leader, bi_xlerobot_leader, keyboard, xbox, hybrid_controller is supported for xlerobot."
        )

    return processed_action


def get_xlerobot_action_space_size():
    """Return the Xlerobot action space size."""
    return 15


def get_xlerobot_joint_names():
    """Return all Xlerobot joint names."""
    return list(xlerobot_joint_names_to_motor_ids.keys())


def get_xlerobot_joint_limits():
    """Return a copy of Xlerobot joint limits."""
    return XLEROBOT_JOINT_LIMITS.copy()
