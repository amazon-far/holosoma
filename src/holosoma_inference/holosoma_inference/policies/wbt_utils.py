from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pinocchio as pin
from defusedxml import ElementTree

from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.misc import get_index_of_a_in_b

if TYPE_CHECKING:
    from loguru import Logger


class MotionClockUtil:
    """Tracks elapsed milliseconds from an external clock source, handling backward clock jumps."""

    __slots__ = ("_anchor", "_clock_sub", "_elapsed_ms_at_anchor", "_last")

    def __init__(self, clock_sub: ClockSub):
        self._clock_sub = clock_sub
        self._anchor: int | None = None
        self._last: int | None = None
        self._elapsed_ms_at_anchor: int = 0

    def reset(self):
        """Reset clock state and underlying clock source."""
        self._anchor = self._last = None
        self._elapsed_ms_at_anchor = 0
        self._clock_sub.reset_origin()

    def elapsed_ms(self, log: Logger | None = None) -> int:
        """Return elapsed milliseconds since reset, handling backward clock jumps."""
        now = self._clock_sub.get_clock()

        if self._anchor is None:
            self._anchor = now

        if self._last is not None and now < self._last:
            # Clock jumped backwards (e.g., sim reset) - re-anchor preserving progress
            self._elapsed_ms_at_anchor += self._last - self._anchor
            self._anchor = now
            if log:
                log.warning("Clock jumped back; re-anchoring.")

        self._last = now
        return self._elapsed_ms_at_anchor + (now - self._anchor)


class TimestepUtil:
    """Converts elapsed time to timesteps, handling forward jump at start."""

    __slots__ = ("_clock", "_interval_ms", "_start_timestep", "_timestep")

    def __init__(self, clock: MotionClockUtil, interval_ms: float, start_timestep: int = 0):
        self._clock = clock
        self._interval_ms = interval_ms
        self._start_timestep = start_timestep
        self._timestep = start_timestep

    def reset(self, start_timestep: int | None = None):
        """Reset timestep state and underlying clock."""
        if start_timestep is not None:
            self._start_timestep = start_timestep
        self._timestep = self._start_timestep
        self._clock.reset()

    def get_timestep(self, log: Logger | None = None) -> int:
        """Return current timestep based on elapsed time, handling forward jump at start."""
        elapsed = self._clock.elapsed_ms(log)
        elapsed_steps = int(elapsed // self._interval_ms)

        # Handle clock jump ahead at start - re-anchor if we're at start but clock jumped
        if self._timestep == self._start_timestep and elapsed_steps > 1:
            if log:
                log.warning("Clock jumped ahead at start; re-anchoring.")
            self._clock.reset()
            return self._timestep

        self._timestep = elapsed_steps + self._start_timestep
        return self._timestep

    @property
    def timestep(self) -> int:
        """Current timestep (read-only access without clock update)."""
        return self._timestep


class PinocchioRobot:
    def __init__(self, robot_cfg: RobotConfig, urdf_text: str):
        # create pinocchio robot
        xml_text = self._create_xml_from_urdf(urdf_text)
        self.robot_model = pin.buildModelFromXML(xml_text, pin.JointModelFreeFlyer())
        self.robot_data = self.robot_model.createData()

        # get joint names in pinocchio robot and real robot
        joint_names_in_real_robot = robot_cfg.dof_names
        joint_names_in_pinocchio_robot = [
            name for name in self.robot_model.names if name not in ["universe", "root_joint"]
        ]
        assert len(joint_names_in_pinocchio_robot) == len(joint_names_in_real_robot), (
            "The number of joints in the pinocchio robot and the real robot are not the same"
        )
        self.real2pinocchio_index = get_index_of_a_in_b(joint_names_in_pinocchio_robot, joint_names_in_real_robot)

        # get ref body frame id in pinocchio robot
        self.ref_body_frame_id = self.robot_model.getFrameId(robot_cfg.motion["body_name_ref"][0])

    def fk_and_get_ref_body_orientation_in_world(self, configuration: np.ndarray) -> np.ndarray:
        # forward kinematics
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        # get ref body pose in world
        ref_body_pose_in_world = self.robot_data.oMf[self.ref_body_frame_id]
        quaternion = pin.Quaternion(ref_body_pose_in_world.rotation)  # (4, )

        return np.expand_dims(quaternion.coeffs(), axis=0)  # xyzw, (1, 4)

    @staticmethod
    def _create_xml_from_urdf(urdf_text: str) -> str:
        """Strip visuals/collisions from URDF text and return XML text."""
        root = ElementTree.fromstring(urdf_text)

        def _is_visual_or_collision(tag: str) -> bool:
            # Handle optional XML namespaces by only checking the suffix after '}'.
            return tag.rsplit("}", maxsplit=1)[-1] in {"visual", "collision"}

        for parent in root.iter():
            for child in list(parent):
                if _is_visual_or_collision(child.tag):
                    parent.remove(child)

        xml_text = ElementTree.tostring(root, encoding="unicode")
        if not xml_text.lstrip().startswith("<?xml"):
            xml_text = '<?xml version="1.0"?>\n' + xml_text
        return xml_text
