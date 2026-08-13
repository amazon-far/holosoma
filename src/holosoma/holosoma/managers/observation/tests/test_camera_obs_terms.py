"""Integration tests for camera observation terms through the real ObservationManager (no sim).

The camera terms (``camera_rgb``/``camera_depth``) read the ``get_camera_data`` accessor,
so a stub simulator returning camera tensors exercises the full term -> ObservationManager path
without a backend. Pins:
- an image term flows through a ``concatenate=False`` dict group with its dtype/values intact
  (the uint8 RGB is NOT promoted to float by the scale=1.0 path);
- ``get_obs_dims`` reports the per-term image shape for a dict group;
- an image term in a ``concatenate=True`` group is rejected with a clear, named error.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

_TERMS = "holosoma.managers.observation.terms.cameras"


class _CameraSim:
    """Stub simulator: returns camera tensors for a fixed (name -> tensor) map."""

    def __init__(self, frames: dict[tuple[str, str], torch.Tensor]):
        self._frames = frames

    def get_camera_data(self, name: str, data_type: str = "rgb", env_ids=None) -> torch.Tensor:
        buf = self._frames[(name, data_type)]
        return buf if env_ids is None else buf[env_ids]


class _Env:
    def __init__(self, sim, num_envs):
        self.simulator = sim
        self.device = "cpu"
        self.num_envs = num_envs


def _make_env(num_envs=2, h=4, w=6):
    rgb = torch.randint(0, 256, (num_envs, h, w, 3), dtype=torch.uint8)
    depth = torch.full((num_envs, h, w, 1), 2.5, dtype=torch.float32)
    sim = _CameraSim({("head", "rgb"): rgb, ("head", "depth"): depth})
    return _Env(sim, num_envs), rgb, depth


def test_rgb_term_flows_through_dict_group_unaltered():
    env, rgb, _ = _make_env()
    cfg = ObservationManagerCfg(
        groups={
            "image": ObsGroupCfg(
                concatenate=False,
                terms={"head_rgb": ObsTermCfg(func=f"{_TERMS}:camera_rgb", params={"sensor": "head"})},
            )
        }
    )
    mgr = ObservationManager(cfg, env, device="cpu")
    obs = mgr.compute()["image"]
    assert isinstance(obs, dict) and set(obs) == {"head_rgb"}
    # uint8 preserved (scale=1.0 must not promote to float) and values unchanged.
    assert obs["head_rgb"].dtype == torch.uint8
    assert torch.equal(obs["head_rgb"], rgb)


def test_depth_term_dict_group_shape_and_dtype():
    env, _, depth = _make_env()
    cfg = ObservationManagerCfg(
        groups={
            "image": ObsGroupCfg(
                concatenate=False,
                terms={"head_depth": ObsTermCfg(func=f"{_TERMS}:camera_depth", params={"sensor": "head"})},
            )
        }
    )
    mgr = ObservationManager(cfg, env, device="cpu")
    obs = mgr.compute()["image"]
    assert obs["head_depth"].dtype == torch.float32
    assert tuple(obs["head_depth"].shape) == tuple(depth.shape)


def test_get_obs_dims_reports_image_shape_for_dict_group():
    env, _, _ = _make_env(num_envs=2, h=4, w=6)
    cfg = ObservationManagerCfg(
        groups={
            "image": ObsGroupCfg(
                concatenate=False,
                terms={
                    "head_rgb": ObsTermCfg(func=f"{_TERMS}:camera_rgb", params={"sensor": "head"}),
                    "head_depth": ObsTermCfg(func=f"{_TERMS}:camera_depth", params={"sensor": "head"}),
                },
            )
        }
    )
    mgr = ObservationManager(cfg, env, device="cpu")
    dims = mgr.get_obs_dims()["image"]
    assert dims == {"head_rgb": (4, 6, 3), "head_depth": (4, 6, 1)}


def test_image_term_in_concatenate_group_is_rejected():
    env, _, _ = _make_env()
    cfg = ObservationManagerCfg(
        groups={
            "bad": ObsGroupCfg(
                concatenate=True,
                terms={"head_rgb": ObsTermCfg(func=f"{_TERMS}:camera_rgb", params={"sensor": "head"})},
            )
        }
    )
    mgr = ObservationManager(cfg, env, device="cpu")
    with pytest.raises(ValueError, match="concatenate=True group 'bad'"):
        mgr.get_obs_dims()
    with pytest.raises(ValueError, match="concatenate=True group 'bad'"):
        mgr.compute()


def test_rgb_term_transform_to_chw_float_for_policy():
    """A transform param reshapes the term to the CHW float [0,1] a visual policy expects, and
    get_obs_dims reports the transformed shape."""
    env, _, _ = _make_env(num_envs=2, h=8, w=6)
    transform = {"resize": [4, 4], "layout": "CHW", "scale": "float01"}
    cfg = ObservationManagerCfg(
        groups={
            "image": ObsGroupCfg(
                concatenate=False,
                terms={
                    "head_rgb": ObsTermCfg(
                        func=f"{_TERMS}:camera_rgb", params={"sensor": "head", "transform": transform}
                    )
                },
            )
        }
    )
    mgr = ObservationManager(cfg, env, device="cpu")
    out = mgr.compute()["image"]["head_rgb"]
    assert tuple(out.shape) == (2, 3, 4, 4) and out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert mgr.get_obs_dims()["image"]["head_rgb"] == (3, 4, 4)
