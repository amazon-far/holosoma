from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pygame
from loguru import logger

from holosoma.config_types.robot import RobotConfig
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.simulator import BridgeConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class BasicSdk2Bridge(ABC):
    """Abstract base class for SDK2Py bridge implementations."""

    def __init__(
        self,
        simulator: BaseSimulator,
        robot_config: RobotConfig,
        bridge_config: BridgeConfig,
        lcm: Any = None,
    ) -> None:
        self.lcm = lcm
        self.robot = robot_config
        self.bridge_config = bridge_config
        self.sdk_type = robot_config.bridge.sdk_type
        self.motor_type = robot_config.bridge.motor_type

        # Store simulator reference for generic access
        self.simulator = simulator

        # The DOFs this bridge controls (subset of the simulator's DOFs, in bridge order). Default
        # (empty controlled/excluded config) is every DOF, matching the historical 1:1 SDK-motor <->
        # sim-DOF behavior. A robot with more sim DOFs than the SDK models (so a co-controller owns
        # the rest) narrows it via RobotBridgeConfig.controlled_dof_names / excluded_dof_names.
        self.dof_indices = self._resolve_dof_indices(simulator, robot_config)
        # None when the subset is every DOF in natural order -> simulator applies the fast full-width
        # ctrl write; a list -> the simulator scatters into ONLY these DOFs' ctrl slots.
        self._apply_indices: list[int] | None = (
            None if self.dof_indices == list(range(simulator.num_dof)) else self.dof_indices
        )
        self._dof_index: npt.NDArray[np.intp] | None = (
            None if self._apply_indices is None else np.asarray(self.dof_indices, dtype=np.intp)
        )

        # SDK motor count is the size of the controlled subset (was: simulator.num_dof).
        self.num_motor = len(self.dof_indices)
        self.torques = np.zeros(self.num_motor)  # Avoids config/model mismatches
        self.torque_limit = np.array([self.robot.dof_effort_limit_list[i] for i in self.dof_indices])
        self.low_cmd: Any = None

        # robot_type presented to the SDK (its type gate + motor-vector sizing). Falls back to the
        # asset's own robot_type, so this changes nothing unless bridge.sdk_robot_type is set.
        self.sdk_robot_type = robot_config.bridge.sdk_robot_type or robot_config.asset.robot_type

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        self.joystick: Any = None

        # The root state this exchange is reading; see transport_exchange.
        self._root_state: _RootState | None = None

        # Initialize SDK-specific components
        self._init_sdk_components()

    def transport_exchange(self) -> None:
        """Exchange one round of data with the SDK: robot state out, commands in.

        The call order is the bridge's own: state first, then base odometry when configured, then
        any joystick, and the incoming command last so it is the freshest thing the control step
        will act on. A bridge whose transport can do better than one call per publish overrides
        this to wrap the sequence.

        ``robot_root_states`` is a backend-owned proxy, not a tensor: every index rebuilds the whole
        ``[pos, quat, lin_vel, ang_vel]`` 13-vector and rotates the angular velocity into the world
        frame. The publishes in one exchange read it at the same simulator state -- nothing steps
        the simulator in here -- so it is read once for all of them.
        """
        self._root_state = self._fresh_root_state()
        try:
            self.publish_low_state()
            if self.bridge_config.publish_odom:
                self.publish_odom()
            if self.joystick:
                self.publish_wireless_controller()
            self.low_cmd_handler()
        finally:
            self._root_state = None

    def _read_root_state(self) -> _RootState:
        """This exchange's root state, or a fresh read for a publish made outside one."""
        if self._root_state is not None:
            return self._root_state
        return self._fresh_root_state()

    def _fresh_root_state(self) -> _RootState:
        """Read env 0's root state.

        Env 0: one bridge drives one robot over one SDK connection, and the torque it writes back
        goes through ``apply_torques_at_dof``, which takes no environment index at all.
        """
        return _RootState(self.simulator.robot_root_states[0].detach().cpu().numpy())

    def _controlled(self, values: torch.Tensor) -> npt.NDArray[np.floating[Any]]:
        """Env 0's row of a ``[num_envs, num_dof]`` tensor, narrowed to the controlled DOFs, as numpy."""
        row = cast("npt.NDArray[np.floating[Any]]", values[0].detach().cpu().numpy())
        return row.copy() if self._dof_index is None else row[self._dof_index]

    def _resolve_dof_indices(self, simulator: BaseSimulator, robot_config: RobotConfig) -> list[int]:
        """Resolve which simulator DOFs this bridge controls, as indices into ``simulator.dof_names``.

        ``controlled_dof_names`` (allow-list, in order) wins over ``excluded_dof_names``
        (control-everything-else). Both empty -> every DOF in natural order (the default, so an
        SDK that owns the whole robot is unchanged). Names are validated against the loaded robot.
        """
        bridge_cfg = robot_config.bridge

        # Default (no subset configured): control every DOF. Uses only num_dof, so it needs no
        # dof_names — keeps the historical whole-robot behavior for any simulator.
        if not bridge_cfg.controlled_dof_names and not bridge_cfg.excluded_dof_names:
            return list(range(simulator.num_dof))

        if bridge_cfg.controlled_dof_names:
            try:
                indices = simulator.dof_indices(bridge_cfg.controlled_dof_names)
            except ValueError as exc:
                raise ValueError(f"bridge.controlled_dof_names: {exc}") from exc
        else:  # excluded_dof_names (controlled empty, but not both — the both-empty case returned above)
            try:
                excluded = set(simulator.dof_indices(bridge_cfg.excluded_dof_names))
            except ValueError as exc:
                raise ValueError(f"bridge.excluded_dof_names: {exc}") from exc
            indices = [i for i in range(simulator.num_dof) if i not in excluded]

        if not indices:
            raise ValueError("bridge DOF subset resolved to empty; nothing to control.")
        return indices

    @abstractmethod
    def _init_sdk_components(self) -> None:
        """Initialize SDK-specific components. Must be implemented by subclasses."""

    @abstractmethod
    def low_cmd_handler(self, msg: Any = None) -> None:
        """Activate the latest available low-level command."""

    @abstractmethod
    def publish_low_state(self) -> None:
        """Publish low-level state. Must be implemented by subclasses."""

    @abstractmethod
    def compute_torques(self) -> npt.NDArray[np.floating[Any]]:
        """Compute torque from current simulator state and the active command."""

    def close(self) -> None:  # noqa: B027 - optional ownership hook; most in-process bridges own nothing.
        """Release resources owned by this bridge.

        In-process bridges need no cleanup by default. Bridges that own threads,
        nodes, contexts, or child processes should override this hook.
        """

    def _compute_pd_torques(
        self,
        tau_ff: Any,
        kp: Any,
        kd: Any,
        q_target: Any,
        dq_target: Any,
    ) -> npt.NDArray[np.floating[Any]]:
        """Helper method for PD control computation (shared logic).

        Parameters
        ----------
        tau_ff : array-like
            Feedforward torques (numpy array or torch tensor)
        kp : array-like
            Proportional gains (numpy array or torch tensor)
        kd : array-like
            Derivative gains (numpy array or torch tensor)
        q_target : array-like
            Target positions (numpy array or torch tensor)
        dq_target : array-like
            Target velocities (numpy array or torch tensor)

        Returns
        -------
        numpy.ndarray
            Computed torques with limits applied
        """
        # Get actual state from simulator, narrowed to the DOFs this bridge controls (so the SDK's
        # num_motor-length kp/q_target broadcast against a matching-length state vector).
        q_actual = self._controlled(self.simulator.dof_pos)
        dq_actual = self._controlled(self.simulator.dof_vel)

        dtype = q_actual.dtype
        tau = _as_array(tau_ff, dtype)
        kp_a = _as_array(kp, dtype)
        kd_a = _as_array(kd, dtype)
        q_des = _as_array(q_target, dtype)
        dq_des = _as_array(dq_target, dtype)

        # PD control computation, then limits
        torques = tau + kp_a * (q_des - q_actual) + kd_a * (dq_des - dq_actual)
        self.torques = np.clip(torques, -self.torque_limit, self.torque_limit)
        return self.torques

    def publish_wireless_controller(self) -> None:
        """Publish wireless controller data."""
        if self.joystick is not None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
            key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
            key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
            key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
            key_state[self.key_map["R2"]] = self.joystick.get_axis(self.axis_id["RT"]) > 0
            key_state[self.key_map["L2"]] = self.joystick.get_axis(self.axis_id["LT"]) > 0
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            if hasattr(self, "wireless_controller"):
                self.wireless_controller.keys = key_value
                self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
                self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
                self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
                self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

                # Debug logging for joystick values
                logger.debug(
                    f"Joystick axes - LX: {self.wireless_controller.lx:.3f}, "
                    f"LY: {self.wireless_controller.ly:.3f}, "
                    f"RX: {self.wireless_controller.rx:.3f}, "
                    f"RY: {self.wireless_controller.ry:.3f}, "
                    f"keys: 0x{key_value:04x}"
                )

                # Only publish if the subclass has a publisher (C++ bindings handle this differently)
                if hasattr(self, "wireless_controller_puber"):
                    self.wireless_controller_puber.Write(self.wireless_controller)

    def setup_joystick(self, device_id: int = 0, js_type: str = "xbox") -> None:
        """Setup joystick/gamepad."""

        # Platform check - pygame only works on Linux/macOS
        if sys.platform not in ["linux", "darwin"]:
            raise RuntimeError(f"Joystick not supported on {sys.platform}. Pygame joystick requires Linux or macOS.")

        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            raise RuntimeError("No joystick detected")

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 3,  # Right stick axis x
                    "RY": 4,  # Right stick axis y
                    "LT": 2,  # Left trigger
                    "RT": 5,  # Right trigger
                    "DX": 6,  # Directional pad x
                    "DY": 7,  # Directional pad y
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 4,
                    "RB": 5,
                    "SELECT": 6,
                    "START": 7,
                    "XBOX": 8,
                    "LSB": 9,
                    "RSB": 10,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 2,  # Right stick axis x
                    "RY": 3,  # Right stick axis y
                    "LT": 4,  # Left trigger
                    "RT": 5,  # Right trigger
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 9,
                    "RB": 10,
                    "SELECT": 4,
                    "START": 6,
                    "XBOX": 5,
                    "LSB": 7,
                    "RSB": 8,
                    "DYU": 11,
                    "DYD": 12,
                    "DXL": 13,
                    "DXR": 14,
                }
            else:
                print("Unsupported OS. ")

        elif js_type == "switch":
            # may differ for different OS, need to be checked
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def _get_dof_states(
        self,
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Get DOF positions, velocities, accelerations (simulator-agnostic).

        Returns:
            tuple: (positions, velocities, accelerations) as numpy arrays
        """
        # Use generic simulator interface - works for all simulators. Narrowed to the controlled
        # DOFs so publish_low_state reports exactly the SDK's num_motor joints.
        positions = self._controlled(self.simulator.dof_pos)
        velocities = self._controlled(self.simulator.dof_vel)

        if not hasattr(self.simulator, "dof_acc"):
            raise RuntimeError("DOF acceleration not available (is the bridge enabled?)")

        accelerations = self._controlled(self.simulator.dof_acc)

        return positions, velocities, accelerations

    @property
    def sim_time(self) -> float:
        """Get the simulation time."""
        return self.simulator.time()

    def _get_actuator_forces(self) -> npt.NDArray[np.floating[Any]]:
        """Get actuator forces (simulator-agnostic).

        Returns:
            numpy.ndarray: Actuator forces
        """
        # Env 0, like every other simulator read this bridge makes.
        forces = cast("npt.NDArray[np.floating[Any]]", self.simulator.get_dof_forces(0).detach().cpu().numpy())
        # Force sensors may be disabled (enable_dof_force_sensors=False) -> empty tensor; a
        # fancy-index would raise, so pass the empty tensor through as the full-width path does.
        if forces.size == 0 or self._dof_index is None:
            return forces.copy()
        return forces[self._dof_index]

    def _get_base_imu_data(
        self,
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Get base IMU data: quaternion, angular velocity, linear acceleration (simulator-agnostic).

        Returns:
            tuple: (quaternion, gyro, acceleration) as numpy arrays
                - quaternion: [w, x, y, z] format (4 elements) - bridge SDK format
                - gyro: body-frame angular velocity [wx, wy, wz] (3 elements)
                - acceleration: linear acceleration [ax, ay, az] (3 elements)
        """
        root = self._read_root_state()

        if not hasattr(self.simulator, "base_linear_acc"):
            logger.warning(
                "Base linear acceleration not available (bridge may be disabled in config). "
                "Returning zero acceleration."
            )
            acceleration = np.zeros(3, dtype=root.quat_wxyz.dtype)
        else:
            acceleration = self.simulator.base_linear_acc[0].detach().cpu().numpy()

        return root.quat_wxyz.copy(), root.ang_vel_body.copy(), acceleration

    def _get_base_odometry(self) -> tuple[list[float], list[float], list[float], float]:
        """Get base odometry: position, orientation, body-frame linear velocity, yaw rate.

        Simulator-agnostic — reads the unified ``robot_root_states`` 13-vector
        ``[pos(3), quat_xyzw(4), lin_vel_world(3), ang_vel_world(3)]``, the sim analog of the
        robot's onboard sport/odom estimate. World-frame velocities are rotated into the base
        (body) frame, as for the IMU gyro, matching the real robot's ``SportModeState``
        (``rt/odommodestate``) whose ``velocity`` twist is body-frame.

        Returns:
            tuple: (position, quat_wxyz, lin_vel_body, yaw_speed)
                - position: [x, y, z] world/odom frame (3 floats)
                - quat_wxyz: [w, x, y, z] SDK order (4 floats)
                - lin_vel_body: [vx, vy, vz] body frame (3 floats)
                - yaw_speed: body-frame yaw rate [rad/s] (float)
        """
        root = self._read_root_state()
        return root.position.tolist(), root.quat_wxyz.tolist(), root.lin_vel_body.tolist(), float(root.ang_vel_body[2])

    def publish_odom(self) -> None:  # noqa: B027 -- deliberate concrete no-op default; SDKs without an odometry channel (e.g. booster) inherit it
        """Publish base odometry over the SDK. Default no-op.

        SDKs with a base-state channel (Unitree's ``rt/odommodestate`` / ``SportModeState``)
        override this. Only invoked by ``SimulatorBridge.transport_step`` when
        ``BridgeConfig.publish_odom`` is set, so bridges without such a channel (e.g. booster)
        simply do nothing.
        """

    def _get_sensor_data(self) -> npt.NDArray[np.floating[Any]]:
        """Get sensor data (Mujoco-only).

        Returns:
            numpy.ndarray: Raw sensor data array
        """
        if not hasattr(self.simulator, "root_data"):
            raise NotImplementedError(f"Sensor data access not implemented for {type(self.simulator).__name__}")

        return cast("npt.NDArray[np.floating[Any]]", self.simulator.root_data.sensordata)


def _as_array(values: Any, dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


class _RootState:
    """Env 0's root state ``[pos, quat_xyzw, lin_vel_world, ang_vel_world]``, with base-frame velocities.

    ``robot_root_states`` velocities are world-frame on every backend; an IMU gyro and the odometry
    twist report them in the base (body) frame.
    """

    __slots__ = ("ang_vel_body", "lin_vel_body", "position", "quat_wxyz")

    def __init__(self, root: npt.NDArray[np.floating[Any]]) -> None:
        self.position = root[0:3].copy()
        x, y, z, w = root[3:7].tolist()
        self.quat_wxyz = np.array([w, x, y, z], dtype=root.dtype)
        # Body-to-world rotation of the unit quaternion; v_world @ R is R^T v_world.
        rotation = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )
        self.lin_vel_body, self.ang_vel_body = root[7:13].reshape(2, 3) @ rotation
