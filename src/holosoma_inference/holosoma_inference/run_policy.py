#!/usr/bin/env python3
"""
Policy Runner Script with Tyro Configuration

This script uses Tyro configuration system to run different policy types.

Usage:
    python run_policy.py inference:g1-29dof-loco --task.model-path path/to/model.onnx
    python run_policy.py inference:g1-29dof-loco --task.model-path wandb://project/run/model.onnx
    python run_policy.py inference:g1-29dof-loco --task.model-path https://wandb-url/files/model.onnx
"""

import sys
import traceback

import tyro
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values.inference import AnnotatedInferenceConfig
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.policies.wbttp import WholeBodyTrackingTargetPosePolicy
from holosoma_inference.policies.wbt_multi import WholeBodyTrackingMultiMotionPolicy
from holosoma_inference.policies.wbt_multi_tp import WholeBodyTrackingMultiMotionTargetPosePolicy
from holosoma_inference.utils.misc import restore_terminal_settings


def _print_control_guide(policy_class, use_joystick: bool):
    """Print control guide for users."""
    is_wbt = policy_class.__name__ == "WholeBodyTrackingPolicy"
    is_wbttp = policy_class.__name__ == "WholeBodyTrackingTargetPosePolicy"
    is_wbt_multi = policy_class.__name__ == "WholeBodyTrackingMultiMotionPolicy"

    logger.info("=" * 80)
    logger.info("🎮 POLICY CONTROLS")
    logger.info("=" * 80)
    logger.info("")

    if use_joystick:
        logger.info("📝 Using JOYSTICK control mode")
        logger.info("")
        logger.info("General Controls:")
        logger.info("  A button       - Start the policy")
        logger.info("  B button       - Stop the policy")
        logger.info("  Y button       - Set robot to default pose")
        logger.info("  L1+R1 (LB+RB)  - Kill controller program")

        if is_wbt:
            logger.info("")
            logger.info("Whole-Body Tracking Controls:")
            logger.info("  Start button   - Start motion clip")
        elif is_wbttp:
            logger.info("")
            logger.info("Whole-Body Tracking Target Pose Controls:")
            logger.info("  Start button   - Start motion clip")
        else:
            logger.info("")
            logger.info("Locomotion Controls:")
            logger.info("  Start button   - Switch walking/standing mode")
            logger.info("  Left stick     - Adjust linear velocity (forward/backward/left/right)")
            logger.info("  Right stick    - Adjust angular velocity (turn left/right)")
    else:
        logger.info("⌨️  Using KEYBOARD control mode")
        logger.info("")
        logger.info("⚠️  IMPORTANT: Make sure THIS TERMINAL is active to receive keyboard input!")
        logger.info("⚠️  All commands below must be entered in THIS terminal window.")
        logger.info("")
        logger.info("General Controls:")
        logger.info("  ]  - Start the policy")
        logger.info("  o  - Stop the policy")
        logger.info("  i  - Set robot to default pose")

        if is_wbt:
            logger.info("")
            logger.info("Whole-Body Tracking Controls:")
            logger.info("  s  - Start motion clip")
        elif is_wbttp:
            logger.info("")
            logger.info("Whole-Body Tracking Target Pose Controls:")
            logger.info("  s  - Start motion clip")
        elif is_wbt_multi:
            logger.info("")
            logger.info("Whole-Body Tracking Multi-Motion Controls:")
            logger.info("  s  - Start motion clip")
        else:
            logger.info("")
            logger.info("Locomotion Controls:")
            logger.info("  =          - Switch walking/standing mode")
            logger.info("  w/s        - Increase/decrease forward velocity")
            logger.info("  a/d        - Increase/decrease lateral velocity")
            logger.info("  q/e        - Increase/decrease angular velocity (turn left/right)")
            logger.info("  z          - Set all velocities to zero")

    logger.info("")
    logger.info("🎬 MuJoCo Simulator Controls (⚠️  ONLY in MuJoCo window, NOT this terminal!):")
    logger.info("  7/8        - Decrease/increase elastic band length")
    logger.info("  9          - Toggle elastic band enable/disable")
    logger.info("  BACKSPACE  - Reset simulation")

    logger.info("")
    logger.info("=" * 80)
    logger.info("👆 Press the appropriate button/key to begin!")
    logger.info("=" * 80)
    logger.info("")


def run_policy(config: InferenceConfig):
    """Run policy with Tyro configuration."""
    logger.info("🚀 Starting Policy with Tyro configuration...")
    logger.info(f"🤖 Robot: {config.robot.robot_type}")
    logger.info(f"📋 Observation groups: {list(config.observation.obs_dict.keys())}")
    logger.info(f"⚙️ RL Rate: {config.task.rl_rate} Hz")
    logger.info(f"📁 Model path: {config.task.model_path}")

    try:
        # Determine policy class based on observation type
        actor_obs = config.observation.obs_dict.get("actor_obs", [])
        
        if config.task.target_pose_path is None:
            # policy_class = WholeBodyTrackingPolicy if "motion_command" in actor_obs else LocomotionPolicy
            # logger.info(f"Using {policy_class.__name__}")
            # policy: LocomotionPolicy | WholeBodyTrackingPolicy = policy_class(config=config)
            policy_class = WholeBodyTrackingMultiMotionPolicy if "motion_command" in actor_obs else LocomotionPolicy
            logger.info(f"Using {policy_class.__name__}")
            policy: LocomotionPolicy | WholeBodyTrackingMultiMotionPolicy = policy_class(config=config)
        else:
            if "motion_command" not in actor_obs:
                logger.error("Target pose provided but 'motion_command' not in observation groups.")
                sys.exit(1)
            policy_class = WholeBodyTrackingMultiMotionTargetPosePolicy
            logger.info(f"Using {policy_class.__name__}")
            policy: WholeBodyTrackingMultiMotionTargetPosePolicy = policy_class(config=config)

        logger.info("✅ Policy initialized successfully!")
        _print_control_guide(policy_class, config.task.use_joystick)
        policy.run()
        logger.info("✅ Policy execution completed!")

    except Exception as e:
        logger.error(f"❌ Error running policy: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_terminal_settings()


def main():
    config = tyro.cli(
        AnnotatedInferenceConfig,
        config=TYRO_CONFIG,
    )
    run_policy(config)


if __name__ == "__main__":
    main()
