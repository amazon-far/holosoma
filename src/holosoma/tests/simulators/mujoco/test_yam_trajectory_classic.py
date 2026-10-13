"""Live CPU (ClassicBackend) run of the YAM joint-trajectory harness.

Routes the MuJoCo ClassicBackend through yam_trajectory_assert.py: the ``yam`` preset (8 DOF, welded base)
follows its shipped joint trajectory under the ``yam-trajectory`` plugin and the gripper body reaches the
expected poses. One sim per subprocess; the verdict is the ``--result-file`` sentinel.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_classic

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "yam_trajectory_assert.py"


def test_yam_trajectory_classic(tmp_path: Path) -> None:
    """Also records the run: exercises the MuJoCo video recorder at 1280x720 (offscreen framebuffer growth)."""
    result_file = tmp_path / "yam_trajectory_mujoco.txt"
    record_dir = tmp_path / "video"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        "--record",
        str(record_dir),
        label="mujoco/yam-trajectory",
        timeout=600,
        result_file=result_file,
    )
    videos = list(record_dir.glob("*.mp4"))
    assert len(videos) == 1 and videos[0].stat().st_size > 100_000, videos

    capture = cv2.VideoCapture(str(videos[0]))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert (width, height) == (1280, 720)
        assert fps == pytest.approx(50.0, abs=0.5)
        assert 500 <= frame_count <= 550

        frames = []
        for index in (0, frame_count // 2, frame_count - 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            assert ok and frame is not None, f"could not decode frame {index}/{frame_count}"
            frames.append(np.asarray(frame, dtype=np.float32))
        assert frames[1].std() > 5.0  # visible scene content, not a blank frame
        assert np.abs(frames[0] - frames[1]).mean() > 1.0  # the arm visibly moves; not a repeated stale frame
        assert np.abs(frames[1] - frames[2]).mean() > 1.0  # recording continues after the midpoint
    finally:
        capture.release()
