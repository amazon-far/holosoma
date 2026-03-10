#!/usr/bin/env python3
"""
Policy Runner Script with Tyro Configuration

This script uses Tyro configuration system to run different policy types.

Usage:
    python run_policy.py inference:g1-29dof-loco --task.model-path path/to/model.onnx
    python run_policy.py inference:g1-29dof-loco --task.model-path wandb://project/run/model.onnx
    python run_policy.py inference:g1-29dof-loco --task.model-path https://wandb-url/files/model.onnx
"""

from __future__ import annotations

import sys
import traceback

import tyro
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values.inference import get_annotated_inference_config
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


def run_policy(config: InferenceConfig):
    """Run policy with Tyro configuration."""
    logger.info("🚀 Starting Policy with Tyro configuration...")
    logger.info(f"🤖 Robot: {config.robot.robot_type}")
    logger.info(f"📋 Observation groups: {list(config.observation.obs_dict.keys())}")
    logger.info(f"⚙️ RL Rate: {config.task.rl_rate} Hz")
    logger.info(f"📁 Model path: {config.task.model_path}")

    try:
        # Determine policy class based on observation type
        policy_class = _select_policy_class(config)
        dual_mode = config.secondary is not None

        if dual_mode:
            logger.info(f"Using {policy_class.__name__} (dual-mode enabled)")
            policy = DualModePolicy(primary_config=config, secondary_config=config.secondary)
        else:
            logger.info(f"Using {policy_class.__name__}")
            policy = policy_class(config=config)

        logger.info("✅ Policy initialized successfully!")
        _print_control_guide(policy_class, config.task.use_joystick, dual_mode=dual_mode)
        policy.run()
        logger.info("✅ Policy execution completed!")

    except Exception as e:
        logger.error(f"❌ Error running policy: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_terminal_settings()


def _split_secondary_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split --secondary.* args out of argv, renaming them for standalone parsing.

    Returns (primary_argv, secondary_argv) where secondary args have the
    ``--secondary.`` prefix stripped, e.g. ``--secondary.task.model-path X``
    becomes ``--task.model-path X``.
    """
    primary = []
    secondary = []
    expect_secondary_value = False
    for arg in argv:
        if arg.startswith("--secondary."):
            renamed = "--" + arg[len("--secondary.") :]
            secondary.append(renamed)
            # If not --key=value form, the next token might be the value
            expect_secondary_value = "=" not in renamed
        elif expect_secondary_value and not arg.startswith("--"):
            secondary.append(arg)
            expect_secondary_value = False
        else:
            primary.append(arg)
            expect_secondary_value = False
    return primary, secondary


def main(annotated_config=None):
    """Main entry point. Extensions can pass their own AnnotatedInferenceConfig."""
    import argparse

    from holosoma_inference.config.config_values.inference import DEFAULTS

    # Pre-parse --secondary-preset and --secondary none before tyro.
    # Tyro can't build a CLI parser for InferenceConfig | None when it
    # contains dict[str, Any] fields, so we handle secondary selection ourselves.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument(
        "--secondary-preset",
        default=None,
        metavar="NAME",
        help=f"Select a preset for the secondary policy. Choices: {list(DEFAULTS.keys())}",
    )
    pre.add_argument("--secondary", default=None, help="Set to 'none' to disable dual-mode.")
    known, remaining = pre.parse_known_args()

    disable_secondary = known.secondary is not None and known.secondary.lower() == "none"
    secondary_preset = known.secondary_preset

    # Strip --secondary.* args from remaining so tyro doesn't see them
    primary_argv, secondary_argv = _split_secondary_args(remaining)
    sys.argv = [sys.argv[0]] + primary_argv

    if annotated_config is None:
        # Use factory function to lazily load extension configs
        annotated_config = get_annotated_inference_config()
    config = tyro.cli(annotated_config, config=TYRO_CONFIG)

    from dataclasses import replace as _replace

    if disable_secondary:
        config = _replace(config, secondary=None)
    elif secondary_preset:
        preset = DEFAULTS.get(secondary_preset)
        if preset is None:
            logger.error(f"Unknown secondary preset: {secondary_preset}")
            logger.info(f"Available presets: {list(DEFAULTS.keys())}")
            sys.exit(1)
        preset = _replace(preset, secondary=None)

        # Parse secondary overrides against the preset defaults
        if secondary_argv:
            sys.argv = [sys.argv[0]] + secondary_argv
            secondary = tyro.cli(InferenceConfig, default=preset, config=TYRO_CONFIG)
        else:
            secondary = preset
        config = _replace(config, secondary=secondary)
    elif secondary_argv:
        # --secondary.* overrides on the config's default secondary
        if config.secondary is not None:
            sys.argv = [sys.argv[0]] + secondary_argv
            secondary = tyro.cli(InferenceConfig, default=config.secondary, config=TYRO_CONFIG)
            config = _replace(config, secondary=secondary)
        else:
            logger.warning("--secondary.* args ignored: no default secondary in this config")

    run_policy(config)


if __name__ == "__main__":
    main()
