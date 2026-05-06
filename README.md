# Holosoma

Holosoma (Greek: "whole-body") is a comprehensive humanoid robotics framework for training and deploying reinforcement learning policies on humanoid robots, as well as motion retargeting. Supports locomotion (velocity tracking) and whole-body tracking tasks across multiple simulators (IsaacGym, IsaacSim, MJWarp, MuJoCo) with algorithms like PPO and FastSAC.

## Features

- **Multi-simulator support**: IsaacGym, IsaacSim, MuJoCo Warp (MJWarp), and MuJoCo (inference only)
- **Multiple RL algorithms**: PPO and FastSAC
- **Robot support**: Unitree G1 and Booster T1 humanoids
- **Task types**: Locomotion (velocity tracking) and whole-body tracking
- **Sim-to-sim and sim-to-real deployment**: Shared inference pipeline across simulation and real robot control
- **Motion retargeting**: Convert human motion capture data to robot motions while preserving interactions with objects and terrain
- **Service-mode deployment**: Run an inference policy as a standalone process with a pluggable live tracking input, a MuJoCo virtual-robot backend, and a per-command dampening shim for safe real-robot rollouts (see [Service mode](#service-mode) below)
- **Wandb integration**: Video logging, automatic ONNX checkpoint uploads, and direct checkpoint loading from Wandb

## Repository Structure

```
src/
├── holosoma/              # Core training framework (locomotion & whole-body tracking)
├── holosoma_inference/    # Inference and deployment pipeline
└── holosoma_retargeting/  # Motion retargeting from human motion data to robots
```

## Documentation

- **[Training Guide](src/holosoma/README.md)** - Train locomotion and whole-body tracking policies in IsaacGym/IsaacSim
- **[Inference & Deployment Guide](src/holosoma_inference/README.md)** - Deploy policies to real robots or evaluate in MuJoCo simulation
- **[Retargeting Guide](src/holosoma_retargeting/holosoma_retargeting/README.md)** - Convert human motion capture data to robot motions

## Quick Start

### Setup

Choose the appropriate setup script based on your use case:

```bash
# For IsaacGym training
bash scripts/setup_isaacgym.sh

# For IsaacSim training
# Requires Ubuntu 22.04 or later due to IsaacSim dependencies
bash scripts/setup_isaacsim.sh

# For MJWarp training and MuJoCo simulation (inference) — conda
bash scripts/setup_mujoco.sh

# For MJWarp training and MuJoCo simulation (inference) — uv (alternative)
bash scripts/setup_mujoco_via_uv.sh

# For inference/deployment
bash scripts/setup_inference.sh

# For motion retargeting
bash scripts/setup_retargeting.sh
```

### Training

Train a G1 robot with FastSAC on IsaacGym:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1
```

> **Note:** For headless servers, see the [training guide](src/holosoma/README.md#video-recording) for video recording configuration.

See the [Training Guide](src/holosoma/README.md) for more examples and configuration options.

### Quick Demo

We provide scripts to run the complete pipeline: (data downloading and processing for LAFAN), retargeting, data conversion, and whole-body tracking policy training.

```bash
# Run retargeting and whole-body tracking policy training using OMOMO data
bash demo_scripts/demo_omomo_wb_tracking.sh

# Run retargeting and whole-body tracking policy training using LAFAN data
bash demo_scripts/demo_lafan_wb_tracking.sh
```

### Deployment & Evaluation

After training, deploy your policies:

- **Real Robot**: See [Real Robot Locomotion](src/holosoma_inference/docs/workflows/real-robot-locomotion.md) or [Real Robot WBT](src/holosoma_inference/docs/workflows/real-robot-wbt.md)
- **MuJoCo Simulation**: See [Sim-to-Sim Locomotion](src/holosoma_inference/docs/workflows/sim-to-sim-locomotion.md) or [Sim-to-Sim WBT](src/holosoma_inference/docs/workflows/sim-to-sim-wbt.md)

Or browse all deployment options in the [Inference & Deployment Guide](src/holosoma_inference/README.md).

## Service mode

Holosoma can run as a standalone inference service and be fed live observations from any upstream process.

Three pieces make service mode work:

1. **Pluggable live-tracking input.** `WholeBodyTrackingPolicy` accepts a `TrackingSource` Protocol on construction. Default behavior is `NullTrackingSource` (byte-identical to the prior clip-driven path). Any implementation can feed a `TrackingPayload` (joint transforms + optional grippers + tracking-quality metadata) on every tick; the policy substitutes the payload's retargeted `motion_command_t` and `ref_quat_xyzw_t` into the observation instead of stepping the default clip. See `holosoma_inference/policies/tracking_source.py`.

2. **Real-time SMPLH → G1 retargeter.** `SMPLRetargeter` in `holosoma_retargeting/src/realtime_smpl_retargeter.py` is a single-frame, mink-based differential IK retargeter (sibling of the offline `InteractionMeshRetargeter`). ~4 ms per frame in isolation on CPU. Per-joint finite-difference velocity, ground-anchor logic, and a cached asset manifest so repeated construction doesn't re-read meshes.

3. **MuJoCo virtual-robot backend + dampening shim.** `MujocoInterface` (`holosoma_inference/sdk/mujoco/mujoco_interface.py`) is a drop-in alternative to `UnitreeInterface` that runs MuJoCo's sim for the `state.motor.*` feedback loop; use it to validate the full observation path without a physical robot on the subnet. The `Dampener` shim (`holosoma_inference/sdk/dampening.py`) sits between the policy and the interface and applies per-tick `q_slew_per_tick`, `q_limit_scale`, and `blend_alpha` guards so a sharp first-tick command cannot immediately saturate motors. Knobs read from environment variables (`HOLOSOMA_Q_SLEW_PER_TICK`, `HOLOSOMA_Q_LIMIT_SCALE`, `HOLOSOMA_BLEND_ALPHA`, `HOLOSOMA_KP_LEVEL`, `HOLOSOMA_KD_LEVEL`) so operators can retune at launch without rebuilding.

See the [Inference & Deployment Guide](src/holosoma_inference/README.md) for how to wire these together.

### Demo Videos

Watch real-world deployments of Holosoma policies *(click thumbnails to play)*

<table>
  <tr>
    <th>G1 Locomotion</th>
    <th>T1 Locomotion</th>
    <th>G1 Dancing</th>
  </tr>
  <tr>
    <td width="33%">
      <a href="https://youtu.be/YYMgj5BDIMI">
        <img src="https://img.youtube.com/vi/YYMgj5BDIMI/hqdefault.jpg" width="100%" alt="▶ G1 Locomotion">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/Q6rNHJZ2a6Y">
        <img src="https://img.youtube.com/vi/Q6rNHJZ2a6Y/hqdefault.jpg" width="100%" alt="▶ T1 Locomotion">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/ouPk69_eFfE">
        <img src="https://img.youtube.com/vi/ouPk69_eFfE/hqdefault.jpg" width="100%" alt="▶ G1 Dancing">
      </a>
    </td>
  </tr>
</table>


## Issue Reporting

We welcome feedback and issue reports to help improve holosoma. Please use issues to:

- Report bugs and technical issues
- Request new features

## Support

If you need help with anything aside from issues feel free to join our [discord server](https://discord.gg/TPupMvpqHc).

Use the discord to discuss larger plans and other more involved problems.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Citation

If you use Holosoma in your research, please cite it according to the "Cite this repository" panel on the right sidebar of the Github repo.

## License

This project is licensed under the Apache-2.0 License.
