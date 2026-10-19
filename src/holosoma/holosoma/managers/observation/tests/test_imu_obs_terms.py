# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Integration tests for IMU observation terms through the real ObservationManager (no sim).

A stub simulator returning IMU tensors exercises the full term -> ObservationManager path without a
backend. The point of interest is the opposite of the camera terms: IMU outputs are flat, so they
must be *accepted* in a ``concatenate=True`` policy group and land in the flat vector in the
manager's sorted-key order.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

_TERMS = "holosoma.managers.observation.terms.imu"


class _ImuSim:
    """Stub simulator: returns IMU tensors for a fixed (name, data_type) map."""

    def __init__(self, outputs: dict[tuple[str, str], torch.Tensor]) -> None:
        self._outputs = outputs

    def get_imu_data(
        self, name: str, data_type: str = "angular_velocity", env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        buf = self._outputs[(name, data_type)]
        return buf if env_ids is None else buf[env_ids]


class _Env:
    def __init__(self, sim: _ImuSim, num_envs: int) -> None:
        self.simulator = sim
        self.device = "cpu"
        self.num_envs = num_envs


_GYRO = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
_ACCEL = torch.tensor([[0.0, 0.0, 9.81], [1.0, 2.0, 3.0]])
_QUAT = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.7071, 0.7071]])


def _make_env() -> _Env:
    sim = _ImuSim(
        {
            ("pelvis_imu", "angular_velocity"): _GYRO,
            ("pelvis_imu", "linear_acceleration"): _ACCEL,
            ("pelvis_imu", "orientation"): _QUAT,
        }
    )
    return _Env(sim, num_envs=2)


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        ("imu_angular_velocity", _GYRO),
        ("imu_linear_acceleration", _ACCEL),
        ("imu_orientation", _QUAT),
    ],
)
def test_each_term_returns_its_output_unaltered(func: str, expected: torch.Tensor) -> None:
    env = _make_env()
    cfg = ObservationManagerCfg(
        groups={
            "policy": ObsGroupCfg(
                concatenate=False,
                terms={"imu": ObsTermCfg(func=f"{_TERMS}:{func}", params={"sensor": "pelvis_imu"})},
            )
        }
    )
    obs = ObservationManager(cfg, env, device="cpu").compute()

    assert torch.equal(obs["policy"]["imu"], expected)


def test_imu_terms_are_accepted_in_a_concatenated_policy_group() -> None:
    """The camera terms are rejected here (4-D); IMU outputs are 2-D, so they must pass.

    Terms concatenate in sorted-key order, so 'accel' (3) precedes 'gyro' (3) in the flat vector.
    """
    env = _make_env()
    cfg = ObservationManagerCfg(
        groups={
            "policy": ObsGroupCfg(
                concatenate=True,
                terms={
                    "gyro": ObsTermCfg(func=f"{_TERMS}:imu_angular_velocity", params={"sensor": "pelvis_imu"}),
                    "accel": ObsTermCfg(func=f"{_TERMS}:imu_linear_acceleration", params={"sensor": "pelvis_imu"}),
                },
            )
        }
    )
    obs = ObservationManager(cfg, env, device="cpu").compute()

    policy = obs["policy"]
    assert isinstance(policy, torch.Tensor)  # concatenate=True yields one flat tensor, not a dict
    assert policy.shape == (2, 6)
    assert torch.equal(policy, torch.cat([_ACCEL, _GYRO], dim=-1))


def test_get_obs_dims_reports_flat_widths() -> None:
    env = _make_env()
    cfg = ObservationManagerCfg(
        groups={
            "policy": ObsGroupCfg(
                concatenate=True,
                terms={
                    "gyro": ObsTermCfg(func=f"{_TERMS}:imu_angular_velocity", params={"sensor": "pelvis_imu"}),
                    "quat": ObsTermCfg(func=f"{_TERMS}:imu_orientation", params={"sensor": "pelvis_imu"}),
                },
            )
        }
    )

    assert ObservationManager(cfg, env, device="cpu").get_obs_dims()["policy"] == 7
