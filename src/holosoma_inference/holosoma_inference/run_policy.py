#!/usr/bin/env python3
"""
Policy Runner Script with Tyro Configuration

This script uses Tyro configuration system to run different policy types.

Usage:
    python run_policy.py inference:g1-29dof-loco --task.model-path path/to/model.onnx
    python run_policy.py inference:g1-29dof-loco secondary:none
    python run_policy.py inference:g1-29dof-loco secondary:g1-29dof-safety-loco
"""

from __future__ import annotations

import sys
import traceback

import tyro
from loguru import logger

from holosoma_inference.config.config_types.inference import DualModePolicyConfig
from holosoma_inference.config.config_values.inference import AnnotatedDualModePolicyConfig
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.policies.dual_mode import DualModePolicy, _select_policy_class
from holosoma_inference.utils.misc import restore_terminal_settings


def _print_control_guide(policy_class, use_joystick: bool, dual_mode: bool = False):
    """Print control guide for users."""
    is_wbt = policy_class.__name__ == "WholeBodyTrackingPolicy"

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

    if dual_mode:
        logger.info("")
        logger.info("🔀 Dual-Mode Controls:")
        if use_joystick:
            logger.info("  X button       - Switch between primary and secondary policy")
        else:
            logger.info("  x              - Switch between primary and secondary policy")

    logger.info("")
    logger.info("=" * 80)
    logger.info("👆 Press the appropriate button/key to begin!")
    logger.info("=" * 80)
    logger.info("")


def run_policy(config: DualModePolicyConfig):
    """Run policy with Tyro configuration."""
    logger.info("🚀 Starting Policy with Tyro configuration...")
    logger.info(f"🤖 Robot: {config.primary.robot.robot_type}")
    logger.info(f"📋 Observation groups: {list(config.primary.observation.obs_dict.keys())}")
    logger.info(f"⚙️ RL Rate: {config.primary.task.rl_rate} Hz")
    logger.info(f"📁 Model path: {config.primary.task.model_path}")

    try:
        # Determine policy class based on observation type
        policy_class = _select_policy_class(config.primary)
        dual_mode = config.secondary is not None

        if dual_mode:
            logger.info(f"Using {policy_class.__name__} (dual-mode enabled)")
            policy = DualModePolicy(config=config)
        else:
            logger.info(f"Using {policy_class.__name__}")
            policy = policy_class(config=config.primary)

        logger.info("✅ Policy initialized successfully!")
        _print_control_guide(policy_class, config.primary.task.use_joystick, dual_mode=dual_mode)
        policy.run()
        logger.info("✅ Policy execution completed!")

    except Exception as e:
        logger.error(f"❌ Error running policy: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_terminal_settings()


def main():
    """Main entry point."""
    config = tyro.cli(AnnotatedDualModePolicyConfig, config=TYRO_CONFIG)
    run_policy(config)


if __name__ == "__main__":
    main()
