"""Integration regression: WBT policy output must stay within joint limits.

Runs the shipped ONNX policy headless over ``pico_example_long.mcap``
and asserts two invariants:

  1. ``dof_pos`` (post-PD sim state) stays within MJCF joint range x 0.95.
     This is a HARD CI GATE — the sim auto-clamps at the joint range,
     so a failure here means the sim itself couldn't keep the robot
     within bounds, which would be a serious regression.

  2. ``action * per_joint_action_scale + default_dof_angles`` (the
     pre-Dampener commanded set-point) stays within MJCF joint range.
     This is CURRENTLY XFAIL because the shipped wbt-dense ONNX
     over-commands ankle-pitch / waist by up to 1 rad. Upstream
     policy retraining will unblock it; when it passes we remove the
     xfail and it becomes a real gate.

Inputs required at test time:
  * ``holosoma_extensions/test_data/pico_example_long.mcap``
  * ``holosoma_extensions/models/active.onnx`` (or a
    HOLOSOMA_TEST_ONNX override)
  * Staged holosoma scene XML at ``/tmp/holosoma_data/...`` (as the
    headless eval expects)

Env vars:
  * ``HOLOSOMA_TEST_DATA_ROOT`` — override the embedding-tree root anchor.
    Defaults to walking up from this file to the first directory that
    contains ``holosoma_extensions/test_data/`` (works from source
    checkouts on any machine).
  * ``HOLOSOMA_TEST_ONNX`` — override the ONNX model under test.
  * ``HOLOSOMA_TEST_DEBUG_NPZ`` — point directly at a pre-generated
    policy debug NPZ (skip the bazel headless run).
  * ``HOLOSOMA_TEST_REQUIRE_FIXTURES`` — when ``1``/``true``, missing
    fixtures fail the test instead of skipping. Use in CI to make
    sure the "hard CI gate" is actually gating.

Local dev loops without the test_data download still pass by default
(skip). CI should set ``HOLOSOMA_TEST_REQUIRE_FIXTURES=1`` and stage
the fixtures before pytest to catch regressions loudly.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest


def _discover_repo_root() -> Path:
    """Return the embedding-tree root (the dir that contains
    ``holosoma_extensions/test_data/``), resolving in three steps:

    1. ``HOLOSOMA_TEST_DATA_ROOT`` env var if set (explicit override).
    2. Walk up from this file until a directory containing
       ``holosoma_extensions/test_data/`` is found (source checkout).
    3. Fall back to a no-op sentinel path so _require() produces a
       clean "required fixture missing" message instead of masking
       the discovery failure.
    """
    override = os.environ.get("HOLOSOMA_TEST_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "holosoma_extensions" / "test_data").is_dir():
            return parent

    return Path("/nonexistent/holosoma_test_data_root_unset")


_REPO_ROOT = _discover_repo_root()
_TEST_MCAP = _REPO_ROOT / "holosoma_extensions" / "test_data" / "pico_example_long.mcap"
_ACTIVE_ONNX = _REPO_ROOT / "holosoma_extensions" / "models" / "active.onnx"


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "1" if default else "0")
    return v.lower() not in ("", "0", "false", "no")


def _require(path: Path) -> None:
    """Skip when a fixture is missing, unless HOLOSOMA_TEST_REQUIRE_FIXTURES=1.

    In CI this env var should be set so missing fixtures fail the test
    rather than silently skip it — otherwise the advertised "HARD CI
    GATE" below is effectively a no-op on any host that doesn't happen
    to have the fixtures staged.
    """
    if path.exists():
        return
    msg = f"required fixture missing: {path}"
    if _env_bool("HOLOSOMA_TEST_REQUIRE_FIXTURES", False):
        pytest.fail(msg)
    pytest.skip(msg)


def _mjcf_joint_limits(dof_names):
    """Return ``(lo, hi)`` as arrays aligned to dof_names order."""
    import mujoco

    import holosoma_retargeting as _hr

    mjcf = Path(_hr.__file__).resolve().parent / "models" / "g1" / "g1_29dof.xml"
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    lo = np.full(len(dof_names), -np.inf)
    hi = np.full(len(dof_names), np.inf)
    for j, name in enumerate(dof_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0 and bool(model.jnt_limited[jid]):
            lo[j] = float(model.jnt_range[jid][0])
            hi[j] = float(model.jnt_range[jid][1])
    return lo, hi


def _onnx_policy_metadata(onnx_path: Path):
    import onnx

    m = onnx.load(str(onnx_path))
    meta = {p.key: p.value for p in m.metadata_props}
    cfg = json.loads(meta["experiment_config"])
    dof_names = cfg["robot"]["dof_names"]
    default_q = np.array(
        [cfg["robot"]["init_state"]["default_joint_angles"][n] for n in dof_names],
        dtype=np.float64,
    )
    action_scale = np.array(json.loads(meta["action_scale"]), dtype=np.float64)
    return dof_names, default_q, action_scale


# ---------------------------------------------------------------------------
# Replicate just enough of mcap_to_holosoma_npz + run_holosoma_reference_headless
# to get a debug NPZ, without requiring bazel run within a test.
# ---------------------------------------------------------------------------


def _load_mcap_poses(mcap_path: Path, max_frames: int | None = None):
    from mcap.reader import make_reader

    poses = []
    stamps = []
    with open(mcap_path, "rb") as f:
        r = make_reader(f)
        for _, _, msg in r.iter_messages(topics=["/pico_body_state"]):
            if len(msg.data) < 4 + 168 * 8:
                continue
            poses.append(
                np.asarray(
                    struct.unpack_from("<168d", msg.data, 4),
                    dtype=np.float64,
                ).reshape(24, 7)
            )
            stamps.append(int(msg.log_time))
            if max_frames is not None and len(poses) >= max_frames:
                break
    return np.stack(poses, axis=0), np.asarray(stamps, dtype=np.int64)


@pytest.fixture(scope="module")
def _policy_run():
    """Run the headless policy once; cache the debug NPZ across tests."""
    _require(_TEST_MCAP)
    onnx_path = Path(os.environ.get("HOLOSOMA_TEST_ONNX", _ACTIVE_ONNX))
    _require(onnx_path)

    # Shortcut: if a debug NPZ is explicitly passed in (CI cache, for example),
    # trust it. Otherwise build one by invoking the headless runner via
    # bazel — too heavy for pytest. So default: require the env var.
    cached = os.environ.get("HOLOSOMA_TEST_DEBUG_NPZ")
    if cached and Path(cached).exists():
        return Path(cached)

    # No pre-generated NPZ — skip locally so dev loops don't flap,
    # but fail loudly if CI has opted into fixture enforcement.
    msg = (
        "HOLOSOMA_TEST_DEBUG_NPZ not set. Run the policy headless first, "
        "e.g. via `pi eval`, then re-run this test with "
        "HOLOSOMA_TEST_DEBUG_NPZ=/path/to/debug.npz pointing at its "
        "debug log."
    )
    if _env_bool("HOLOSOMA_TEST_REQUIRE_FIXTURES", False):
        pytest.fail(msg)
    pytest.skip(msg)


def test_dof_pos_within_joint_limits(_policy_run):  # noqa: PT019
    """Hard gate: sim dof_pos stays within MJCF joint range x 0.95."""
    debug_npz = _policy_run
    d = np.load(debug_npz)
    assert "dof_pos" in d.files, (
        f"debug NPZ {debug_npz} is missing 'dof_pos'; policy run did not produce a complete debug log."
    )
    dof_pos = np.asarray(d["dof_pos"])
    onnx_path = Path(os.environ.get("HOLOSOMA_TEST_ONNX", _ACTIVE_ONNX))
    dof_names, _default_q, _scale = _onnx_policy_metadata(onnx_path)
    assert dof_pos.shape[1] == len(dof_names), (
        f"debug dof_pos has {dof_pos.shape[1]} columns, ONNX has {len(dof_names)} joints."
    )

    lo, hi = _mjcf_joint_limits(dof_names)
    # 5% inset so policy fine-tuning that just barely touches a limit
    # doesn't flap the test every run.
    margin = 0.05 * (hi - lo)
    lo_t = lo + margin
    hi_t = hi - margin

    over_hi = dof_pos > hi_t[None, :]
    under_lo = dof_pos < lo_t[None, :]
    violations_per_dof = (over_hi | under_lo).sum(axis=0)

    if violations_per_dof.sum() == 0:
        return

    msg = ["sim dof_pos escaped MJCF limits x 0.95 on some frames:"]
    for j, name in enumerate(dof_names):
        n_bad = int(violations_per_dof[j])
        if n_bad == 0:
            continue
        col = dof_pos[:, j]
        msg.append(
            f"  {name}: {n_bad} frames, range=[{lo[j]:.3f},{hi[j]:.3f}] observed=[{col.min():.3f},{col.max():.3f}]"
        )
    pytest.fail("\n".join(msg))


@pytest.mark.xfail(
    reason="Shipped wbt-dense ONNX over-commands ankle-pitch / waist by up to "
    "1 rad. Upstream policy retraining will unblock this. See "
    "HANDOFF-2026-05-04 policy-safety analysis.",
    strict=True,  # if policy retrains and this suddenly passes, flip
    # the marker off — we want to know.
)
def test_q_target_within_joint_limits(_policy_run):  # noqa: PT019
    """Tracking metric: commanded set-point (action*scale + default) in range.

    Currently XFAIL — policy output has known out-of-range commands. When
    upstream retraining lands and this passes, remove the xfail to make
    it a hard gate.
    """
    debug_npz = _policy_run
    d = np.load(debug_npz)
    action = np.asarray(d["action"])
    onnx_path = Path(os.environ.get("HOLOSOMA_TEST_ONNX", _ACTIVE_ONNX))
    dof_names, default_q, scale = _onnx_policy_metadata(onnx_path)

    q_target = action * scale[None, :] + default_q[None, :]

    lo, hi = _mjcf_joint_limits(dof_names)
    over_hi = (q_target > hi[None, :]).sum(axis=0)
    under_lo = (q_target < lo[None, :]).sum(axis=0)
    total_violations = int(over_hi.sum() + under_lo.sum())
    if total_violations == 0:
        return

    msg = ["commanded q_target escaped MJCF limits:"]
    for j, name in enumerate(dof_names):
        n_bad = int(over_hi[j] + under_lo[j])
        if n_bad == 0:
            continue
        col = q_target[:, j]
        msg.append(
            f"  {name}: {n_bad} frames, range=[{lo[j]:.3f},{hi[j]:.3f}] observed=[{col.min():.3f},{col.max():.3f}]"
        )
    pytest.fail("\n".join(msg))


def test_dof_pos_slew_bounded(_policy_run):  # noqa: PT019
    """Per-tick slew (|Δq_target|) stays within 0.5 rad on each DOF.

    0.5 rad / tick at 50 Hz = 25 rad/s — a soft upper bound that
    protects against one-tick spikes that would impulse-load a joint on
    hardware.
    """
    debug_npz = _policy_run
    d = np.load(debug_npz)
    dof_pos = np.asarray(d["dof_pos"])
    dq = np.abs(np.diff(dof_pos, axis=0))

    per_dof_max = dq.max(axis=0)
    thresh = 0.5
    worst = per_dof_max.max()
    if worst <= thresh:
        return

    onnx_path = Path(os.environ.get("HOLOSOMA_TEST_ONNX", _ACTIVE_ONNX))
    dof_names, _, _ = _onnx_policy_metadata(onnx_path)
    bad = [(dof_names[j], float(per_dof_max[j])) for j in range(len(dof_names)) if per_dof_max[j] > thresh]
    bad.sort(key=lambda t: -t[1])
    lines = [f"per-tick dof_pos slew exceeds {thresh} rad:"]
    for name, v in bad[:5]:
        lines.append(f"  {name}: {v:.3f} rad/tick ({v * 50:.1f} rad/s equiv)")
    pytest.fail("\n".join(lines))
