from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Tuple, cast

import mujoco  # type: ignore[import-not-found]
import mujoco.viewer as mjv  # type: ignore[import-not-found]
import numpy as np
import torch
import torch.nn.functional as F
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.data_conversion import DataConversionConfig  # noqa: E402
from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402

DynamicState = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
]
StaticState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]

# Parse the arguments using the config structure
args_cli = tyro.cli(DataConversionConfig)


"""Rest everything follows."""


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    *,
    object_name: str | None = None,
) -> SimpleNamespace:
    """Create a mutable namespace with robot and motion data attributes."""
    namespace = SimpleNamespace()

    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(namespace, attr, getattr(robot_config, attr))

    for attr, value in motion_data_config.legacy_constants().items():
        setattr(namespace, attr, value)

    if object_name is not None:
        namespace.OBJECT_NAME = object_name

    if namespace.OBJECT_NAME != "ground":
        namespace.OBJECT_URDF_FILE = f"models/{namespace.OBJECT_NAME}/{namespace.OBJECT_NAME}.urdf"
        namespace.OBJECT_MESH_FILE = f"models/{namespace.OBJECT_NAME}/{namespace.OBJECT_NAME}.obj"
        namespace.OBJECT_URDF_TEMPLATE = f"models/templates/{namespace.OBJECT_NAME}.urdf.jinja"
        namespace.SCENE_XML_FILE = (
            f"models/{robot_config.robot_type}/"
            f"{robot_config.robot_type}_{namespace.ROBOT_DOF}dof_w_{namespace.OBJECT_NAME}.xml"
        )
    else:
        namespace.OBJECT_URDF_FILE = namespace.ROBOT_URDF_FILE
        namespace.OBJECT_MESH_FILE = ""
        namespace.SCENE_XML_FILE = namespace.ROBOT_URDF_FILE.replace(".urdf", ".xml")

    return namespace


def quat_conjugate(q):  # (...,4) [w,x,y,z]
    qc = q.clone()
    qc[..., 1:] = -qc[..., 1:]
    return qc


def quat_mul(a, b):  # Hamilton product, (...,4)
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_to_rotvec(q, eps=1e-8):  # axis-angle vector (rotvec), (...,3)
    q = F.normalize(q, dim=-1)
    # shortest path: flip if needed
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., 0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)  # (...,)
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))  # (...,)
    axis = torch.where(s[..., None] > eps, q[..., 1:] / s[..., None], torch.zeros_like(q[..., 1:]))
    return axis * angle[..., None]


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        line_range: tuple[int, int] | None,
        has_dynamic_object: bool,
        use_omniretarget_data: bool,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.line_range = line_range
        self.has_dynamic_object = has_dynamic_object
        self.use_omniretarget_data = use_omniretarget_data
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.motion_file.endswith(".npz"):
            data = np.load(self.motion_file)
            self.input_fps = round(1 / data.get("fps", 1 / self.input_fps))
            motion = torch.from_numpy(data["qpos"]).to(torch.float32)
        else:
            raise ValueError("Unsupported motion file format. Use .csv or .npz.")

        motion = motion.to(torch.float32).to(self.device)
        if self.use_omniretarget_data:
            self.motion_base_poss_input = motion[:, 4:7]
            self.motion_base_rots_input = motion[:, :4]
        else:
            self.motion_base_poss_input = motion[:, :3]
            self.motion_base_rots_input = motion[:, 3:7]

        self.motion_dof_poss_input = motion[:, 7:36]

        if self.has_dynamic_object:
            if self.use_omniretarget_data:
                self.motion_object_poss_input = motion[:, -3:]
                self.motion_object_rots_input = motion[:, -7:-3]
            else:
                self.motion_object_poss_input = motion[:, -7:-4]
                self.motion_object_rots_input = motion[:, -4:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

        if self.has_dynamic_object:
            self.motion_object_poss = self._lerp(
                self.motion_object_poss_input[index_0],
                self.motion_object_poss_input[index_1],
                blend.unsqueeze(1),
            )
            self.motion_object_rots = self._slerp(
                self.motion_object_rots_input[index_0],
                self.motion_object_rots_input[index_1],
                blend,
            )
        print(
            "Motion interpolated, input frames: "
            f"{self.input_frames}, input fps: {self.input_fps}, "
            f"output frames: {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-8):
        """
        q0, q1: (..., 4) unit quaternions (wxyz or xyzw—just be consistent between inputs).
        t:      (...)  blend in [0,1] that broadcasts against the leading dims of q0/q1.
        """
        q0 = F.normalize(q0, dim=-1)
        q1 = F.normalize(q1, dim=-1)

        # Make sure t has a trailing dim for broadcasting with (...,4)
        if t.ndim == q0.ndim - 1:
            t = t.unsqueeze(-1)

        # Antipodal fix (take shortest path)
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        # If very close, fall back to lerp
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        close = sin_theta.abs() < eps

        s0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
        s1 = torch.sin(t * theta) / (sin_theta + eps)
        out = s0 * q0 + s1 * q1

        # Linear fallback for tiny angles
        out = torch.where(close, (1.0 - t) * q0 + t * q1, out)
        return F.normalize(out, dim=-1)

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

        if self.has_dynamic_object:
            self.motion_object_lin_vels = torch.gradient(self.motion_object_poss, spacing=self.output_dt, dim=0)[0]
            self.motion_object_ang_vels = self._so3_derivative(self.motion_object_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        rotvec = quat_to_rotvec(q_rel)  # (T-2,3), rotation over 2*dt
        omega = rotvec / (2.0 * dt)
        # pad ends (copy-first/last) to keep length T
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self) -> DynamicState | StaticState:
        """Gets the next state of the motion."""
        state: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        )
        if self.has_dynamic_object:
            state = (
                self.motion_base_poss[self.current_idx : self.current_idx + 1],
                self.motion_base_rots[self.current_idx : self.current_idx + 1],
                self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
                self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
                self.motion_dof_poss[self.current_idx : self.current_idx + 1],
                self.motion_dof_vels[self.current_idx : self.current_idx + 1],
                self.motion_object_poss[self.current_idx : self.current_idx + 1],
                self.motion_object_rots[self.current_idx : self.current_idx + 1],
                self.motion_object_lin_vels[self.current_idx : self.current_idx + 1],
                self.motion_object_ang_vels[self.current_idx : self.current_idx + 1],
            )
        else:
            state = (
                self.motion_base_poss[self.current_idx : self.current_idx + 1],
                self.motion_base_rots[self.current_idx : self.current_idx + 1],
                self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
                self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
                self.motion_dof_poss[self.current_idx : self.current_idx + 1],
                self.motion_dof_vels[self.current_idx : self.current_idx + 1],
            )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return (*state, reset_flag)  # type: ignore[return-value]


def world_body_velocities(model, data):
    """
    Per-body COM velocities in the world frame.

    Returns:
        lin_w: (nbody, 3) linear velocities in world coords
        ang_w: (nbody, 3) angular velocities in world coords
    """
    v = np.zeros((model.nbody, 6))
    for b in range(model.nbody):
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,  # object type = body
            b,  # body id
            v[b],
            0,  # flg_local = 0 -> world orientation
        )

    lin_w = v[:, 3:6]  # [vx, vy, vz] in world frame
    ang_w = v[:, 0:3]  # [wx, wy, wz] in world frame
    return lin_w, ang_w


def run_simulator(joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    device = torch.device("cpu")
    has_dynamic_object = args_cli.has_dynamic_object
    use_omniretarget_data = args_cli.use_omniretarget_data
    line_range: tuple[int, int] | None = args_cli.line_range
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=device,
        line_range=line_range,
        has_dynamic_object=has_dynamic_object,
        use_omniretarget_data=use_omniretarget_data,
    )

    object_name = args_cli.object_name
    if object_name is None:
        object_name = "largebox" if has_dynamic_object else None

    robot_config = RobotConfig(robot_type=args_cli.robot)
    motion_config = MotionDataConfig(
        data_format=args_cli.data_format,
        robot_type=args_cli.robot,
    )
    constants = create_task_constants(
        robot_config,
        motion_config,
        object_name=object_name,
    )

    # Load Mujoco model
    object_name = constants.OBJECT_NAME
    robot_model_path = constants.ROBOT_URDF_FILE
    if object_name == "ground":
        robot_xml_path = robot_model_path.replace(".urdf", ".xml")
    elif object_name == "multi_boxes":
        robot_xml_path = constants.SCENE_XML_FILE
    else:
        if object_name is None:
            raise ValueError("object_name cannot be None when it's not 'ground' or 'multi_boxes'")
        robot_xml_path = robot_model_path.replace(".urdf", "_w_" + object_name + ".xml")

    robot = mujoco.MjModel.from_xml_path(robot_xml_path)
    robot_data = mujoco.MjData(robot)
    print("Loading robot model from: ", robot_xml_path)

    # Prepare dof index for mujoco to correctly assign the values from input data
    dof_name_list = []
    for i in range(robot.njnt):  # 'nv' is the number of DoFs
        if robot.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof_name = mujoco.mj_id2name(robot, mujoco.mjtObj.mjOBJ_JOINT, i)
        dof_name_list.append(dof_name)
    print("The number of DoFs in the robot model is: ", len(dof_name_list))
    print(dof_name_list)
    dof_index_list = [joint_names.index(dof_name) for dof_name in dof_name_list]
    print(dof_index_list)

    # Prepare mujoco viewer
    viewer = mjv.launch_passive(robot, robot_data, show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0

    viewer.cam.distance = 2.0
    viewer.cam.elevation = -20.0
    viewer.cam.azimuth = 45.0

    log: dict[str, Any]
    if has_dynamic_object:
        log = {
            "fps": [args_cli.output_fps],
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
            "object_pos_w": [],
            "object_quat_w": [],
            "object_lin_vel_w": [],
            "object_ang_vel_w": [],
        }
    else:
        log = {
            "fps": [args_cli.output_fps],
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
        }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while True:
        start_time = time.perf_counter()
        if has_dynamic_object:
            result = motion.get_next_state()
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
                motion_object_pos,
                motion_object_rot,
                motion_object_lin_vel,
                motion_object_ang_vel,
                reset_flag,
            ) = cast(
                "DynamicState",
                result,
            )
        else:
            result = motion.get_next_state()
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
                reset_flag,
            ) = cast(
                "StaticState",
                result,
            )

        if has_dynamic_object:
            # set root state
            robot_data.qpos[:] = torch.cat(
                [
                    motion_base_pos,
                    motion_base_rot,
                    motion_dof_pos[:, dof_index_list],
                    motion_object_pos,
                    motion_object_rot,
                ],
                dim=1,
            )
            robot_data.qvel[:] = torch.cat(
                [
                    motion_base_lin_vel,
                    motion_base_ang_vel,
                    motion_dof_vel[:, dof_index_list],
                    motion_object_lin_vel,
                    motion_object_ang_vel,
                ],
                dim=1,
            )

        else:
            # set root state
            robot_data.qpos[:] = torch.cat([motion_base_pos, motion_base_rot, motion_dof_pos[:, dof_index_list]], dim=1)
            robot_data.qvel[:] = torch.cat(
                [motion_base_lin_vel, motion_base_ang_vel, motion_dof_vel[:, dof_index_list]], dim=1
            )

        mujoco.mj_forward(robot, robot_data)
        viewer.sync()

        end_time = time.perf_counter()
        time.sleep(max(0, motion.output_dt - (end_time - start_time)))

        if not file_saved:
            lin_vel_w, ang_vel_w = world_body_velocities(robot, robot_data)
            if has_dynamic_object:
                log["object_pos_w"].append(robot_data.qpos[-7:-4].copy())
                log["object_quat_w"].append(robot_data.qpos[-4:].copy())
                log["object_lin_vel_w"].append(robot_data.qvel[-6:-3].copy())
                log["object_ang_vel_w"].append(robot_data.qvel[-3:].copy())

                # Remove object field from qpos and qvel
                log["joint_pos"].append(robot_data.qpos[:-7].copy())
                log["joint_vel"].append(robot_data.qvel[:-6].copy())
            else:
                log["joint_pos"].append(robot_data.qpos[:].copy())
                log["joint_vel"].append(robot_data.qvel[:].copy())

            log["body_pos_w"].append(robot_data.xpos[:].copy())
            log["body_quat_w"].append(robot_data.xquat[:].copy())
            log["body_lin_vel_w"].append(lin_vel_w[:].copy())
            log["body_ang_vel_w"].append(ang_vel_w[:].copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)[:]

            if has_dynamic_object:
                for k in (
                    "object_pos_w",
                    "object_quat_w",
                    "object_lin_vel_w",
                    "object_ang_vel_w",
                ):
                    log[k] = np.stack(log[k], axis=0)[:]

            # Add joint names and body names to the log
            # Names for qpos/qvel follow joint order
            joint_names = [mujoco.mj_id2name(robot, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(robot.njnt)]
            body_names = [mujoco.mj_id2name(robot, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(robot.nbody)]

            if has_dynamic_object:
                log["joint_names"] = joint_names[1:-1]  # remove the root free joint name and the object joint name
            else:
                log["joint_names"] = joint_names[1:]  # remove the root free joint name

            log["body_names"] = body_names

            if args_cli.output_name is None:
                raise ValueError("output_name cannot be None")
            output_res_folder = Path(args_cli.output_name).parent
            os.makedirs(output_res_folder, exist_ok=True)
            np.savez(args_cli.output_name, **log)

        if args_cli.once and file_saved:
            print("[INFO]: Motion replay completed, exiting...")
            viewer.close()
            break


def main():
    """Main function."""
    # Run the simulator
    run_simulator(
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )


if __name__ == "__main__":
    # run the main function
    main()
