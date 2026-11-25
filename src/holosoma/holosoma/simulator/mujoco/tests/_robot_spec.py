"""Compile inline robot specs through the prefixed scene attachment path."""

import mujoco

from holosoma.simulator.mujoco.robot_addressing import validate_robot_joint_types
from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta, MujocoSceneManager


def compile_robot_spec(robot_spec: mujoco.MjSpec, prefix: str = "robot_") -> tuple[mujoco.MjModel, ActorSpecMeta]:
    root_body = robot_spec.worldbody.bodies[0]
    validate_robot_joint_types(robot_spec, str(root_body.name))
    meta = MujocoSceneManager._capture_spec_meta(None, robot_spec, prefix, str(root_body.name))  # type: ignore[arg-type]
    world = mujoco.MjSpec()
    world.copy_during_attach = True
    world.attach(robot_spec, frame=world.worldbody.add_frame(), prefix=prefix)
    return world.compile(), meta
