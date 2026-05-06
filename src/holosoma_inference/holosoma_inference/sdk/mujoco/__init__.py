"""MuJoCo virtual-robot backend for holosoma_inference.

Drop-in replacement for the Unitree binding that runs the robot in a
MuJoCo simulation instead of talking to real hardware. Lets the full WBT
pipeline (retargeter + policy) execute without needing a robot or
CycloneDDS.
"""

from holosoma_inference.sdk.mujoco.mujoco_interface import MujocoInterface

__all__ = ["MujocoInterface"]
