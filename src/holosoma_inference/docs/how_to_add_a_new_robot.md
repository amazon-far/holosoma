# Adding a New Robot to Holosoma Inference

This document is meant to serve as a high-level overview on how to add a new robot to work with `holosoma_inference`.

---

## 1. Implement Interface.

Create `MyRobotInterface(BaseInterface)` and implement all abstract methods. Most importantly, `get_low_state()` and  `send_low_command()`:

```python
from holosoma_inference.sdk.base.base_interface import BaseInterface

class MyRobotInterface(BaseInterface):

    def __init__(self, robot_config, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        # Initialize your SDK / communication layer here

    def get_low_state(self) -> np.ndarray:
        """Return shape (1, 3+4+N+3+3+N) array:
        [base_pos(3), quat(4), joint_pos(N), lin_vel(3), ang_vel(3), joint_vel(N)]
        """
        ...

    def send_low_command(self, cmd_q, cmd_dq, cmd_tau,
                         dof_pos_latest=None, kp_override=None, kd_override=None):
        """Map joint-space commands to your robot's motor API."""
        ...
```

Notes:
- `MyRobotInterface` can use robot-specific IPC (like ROS2) under the hood to communicate with the hardware. However, those libraries must not be required dependencies of `holosoma` or `holosoma_inference` itself.
- `get_low_state()` must return a `(1, 2*(3+N)+4+3)` numpy array in the exact field order above. The policy reads this array by offset, so the layout matters.
- `send_low_command()` receives arrays of length `num_joints` (joint-space). You are responsible for remapping to motor-space if they differ.
- Gain properties (`kp_level`/`kd_level`) are float multipliers (default 1.0) that scale `motor_kp`/`motor_kd` from the config.

---

## 2. Implement `BasicSdk2Bridge` (simulation only)

In order to test `sim2sim` workflow, create `bridge/myrobot/myrobot_sdk2py_bridge.py` and subclass `BasicSdk2Bridge`. Implement the four abstract methods:

```python
from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge

class MyRobotSdk2Bridge(BasicSdk2Bridge):

    def _init_sdk_components(self):
        """Initialize SDK-specific state (message types, publishers, etc.)."""
        ...

    def low_cmd_handler(self, msg):
        """Handle incoming low-level command messages from the policy."""
        ...

    def publish_low_state(self):
        """Read simulator state and publish it in your SDK's format."""
        ...

    def compute_torques(self):
        """Compute motor torques from the latest command.
        Use the helper `_compute_pd_torques()` for standard PD control.
        """
        ...
```

Notes:
- Helper methods `_get_dof_states()`, `_get_base_imu_data()`, and `_get_actuator_forces()` are provided by the base class for reading simulator state.
- `_compute_pd_torques(tau_ff, kp, kd, q_target, dq_target)` handles PD control + torque limiting — use it in `compute_torques()` unless you need custom logic.
- See `bridge/unitree/unitree_sdk2py_bridge.py` for a complete example.

---

## 3. Register entrypoints

It's possible for your package implementation to live in a separate package separate from `holosoma`.
To achieve that, implement `MyRobotSdk2Bridge` and `MyRobotInterface` in your python package, and register the SDK using [entry points](https://packaging.python.org/en/latest/specifications/entry-points/). Thanks to that, `create_interface()` and `create_sdk2py_bridge()` will discover your implementations at runtime.

In your `pyproject.toml`:
```toml
[project.entry-points."holosoma.sdk"]
myrobot = "my_package.sdk.myrobot_interface:MyRobotInterface"

[project.entry-points."holosoma.bridge"]
myrobot = "my_package.bridge.myrobot_sdk2py_bridge:MyRobotSdk2Bridge"
```

The key (e.g. `myrobot`) must match the `sdk_type` field in your robot config. Once your package is pip-installed, `create_interface()` and `create_sdk2py_bridge()` will resolve your implementation by name without any changes to the core codebase.
