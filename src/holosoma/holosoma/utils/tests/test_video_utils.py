"""Unit tests for the video encoders in :mod:`holosoma.utils.video_utils` (real ffmpeg, tmp dir I/O).

The streaming encoder: a crashed producer leaves only ``.partial``; NVENC falls back to libx264 when
unavailable or over the session cap, and releases its session (the real-NVENC case runs only on a
host with an NVIDIA GPU).

Covers the decoded round trip (frame count, playback rate, pixel content), that an ffmpeg failure
raises instead of leaving a partial file behind, and that ``create_video`` writes exactly one file
per format: h264 in a single pass with no OpenCV intermediate, mp4v failing loudly with no partial file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from holosoma.utils import video_utils
from holosoma.utils.video_utils import H264StreamEncoder, create_video, encode_h264

pytestmark = [
    pytest.mark.no_sim,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available"),
]


def _decode(path: Path) -> tuple[list[npt.NDArray[np.uint8]], float]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames: list[npt.NDArray[np.uint8]] = []
        while True:
            valid, bgr = capture.read()
            if not valid:
                return frames, fps
            frames.append(np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8))
    finally:
        capture.release()


def test_round_trip_preserves_count_rate_and_content(tmp_path: Path) -> None:
    # Distinct flat colors per frame, so channel order and frame order are both observable.
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (120, 120, 120)]
    frames = [np.full((32, 48, 3), c, dtype=np.uint8) for c in colors]
    path = encode_h264(frames, 12.5, tmp_path / "clip.mp4")

    decoded, fps = _decode(path)
    assert fps == pytest.approx(12.5, rel=1e-3)
    assert len(decoded) == len(frames)
    for src, out in zip(frames, decoded, strict=False):
        assert out.shape == src.shape
        assert np.abs(out.astype(int) - src.astype(int)).mean(axis=(0, 1)) == pytest.approx([0, 0, 0], abs=4)


def test_accepts_strided_frame_views(tmp_path: Path) -> None:
    # Per-env frames arrive as views into a batched [N, H, W, 3] tensor; a column slice is non-contiguous.
    batch = np.zeros((3, 16, 32, 3), dtype=np.uint8)
    batch[:, :, :, 1] = 180
    frames = [batch[i, :, :16] for i in range(3)]
    assert not frames[0].flags.c_contiguous
    decoded, _ = _decode(encode_h264(frames, 30.0, tmp_path / "views.mp4"))
    assert len(decoded) == 3
    assert decoded[0].shape == (16, 16, 3)
    assert decoded[0][..., 1].mean() == pytest.approx(180, abs=4)


def test_ffmpeg_failure_raises_and_removes_partial_output(tmp_path: Path) -> None:
    # yuv420p needs even dimensions; libx264 rejects a 5-pixel-wide clip.
    frames = [np.zeros((4, 5, 3), dtype=np.uint8)] * 3
    path = tmp_path / "odd.mp4"
    with pytest.raises(RuntimeError, match="ffmpeg \\(libx264\\) failed"):
        encode_h264(frames, 30.0, path)
    assert list(tmp_path.iterdir()) == []


def test_mismatched_frame_shape_raises(tmp_path: Path) -> None:
    frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 10, 3), dtype=np.uint8)]
    path = tmp_path / "bad.mp4"
    with pytest.raises(ValueError, match="expected uint8 frames"):
        encode_h264(frames, 30.0, path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("output_format", ["h264", "mp4"])
def test_create_video_writes_one_file_without_intermediate(tmp_path: Path, output_format: str) -> None:
    frames = np.stack([np.full((32, 48, 3), (40 * i, 100, 200 - 40 * i), dtype=np.uint8) for i in range(4)])
    path = create_video(frames, fps=25, save_dir=tmp_path, output_format=output_format, wandb_logging=False)

    assert path is not None
    assert list(tmp_path.iterdir()) == [path]
    decoded, fps = _decode(path)
    assert fps == pytest.approx(25.0, rel=1e-3)
    assert len(decoded) == len(frames)
    assert decoded[2].mean(axis=(0, 1)) == pytest.approx(frames[2].mean(axis=(0, 1)), abs=6)


def test_create_video_h264_encodes_in_one_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The h264 path must not go through an OpenCV intermediate file and a second transcode.
    def no_opencv_writer(*args: object, **kwargs: object) -> None:
        raise AssertionError("h264 must not write an OpenCV intermediate")

    monkeypatch.setattr("holosoma.utils.video_utils.cv2.VideoWriter", no_opencv_writer)
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    path = create_video(frames, fps=30, save_dir=tmp_path, output_format="h264", wandb_logging=False)
    assert path is not None
    assert len(_decode(path)[0]) == 3


def test_create_video_mp4_raises_when_writer_cannot_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _UnopenedWriter:
        def __init__(self, *args: object) -> None:
            pass

        def isOpened(self) -> bool:  # noqa: N802 (cv2 API name)
            return False

    monkeypatch.setattr("holosoma.utils.video_utils.cv2.VideoWriter", _UnopenedWriter)
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Video creation failed") as excinfo:
        create_video(frames, fps=30, save_dir=tmp_path, output_format="mp4", wandb_logging=False)
    assert "could not open" in str(excinfo.value.__cause__)
    assert list(tmp_path.iterdir()) == []


def test_create_video_mp4_removes_partial_file_on_bad_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fail the RGB->BGR conversion on the second frame, after the writer has started the file.
    real_cvt = cv2.cvtColor
    calls = []

    def failing_cvt(frame: npt.NDArray[np.uint8], code: int) -> npt.NDArray[np.uint8]:
        calls.append(code)
        if len(calls) == 2:
            raise cv2.error("bad frame")
        return np.asarray(real_cvt(frame, code), dtype=np.uint8)

    monkeypatch.setattr("holosoma.utils.video_utils.cv2.cvtColor", failing_cvt)
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Video creation failed"):
        create_video(frames, fps=30, save_dir=tmp_path, output_format="mp4", wandb_logging=False)
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_crash_mid_stream_leaves_only_a_partial_file(tmp_path: Path) -> None:
    # A producer that dies without finishing (os._exit skips every cleanup) must not leave a
    # complete-looking mp4; ffmpeg sees EOF and finalizes what it has under the .partial name.
    child = f"""
import numpy as np
from pathlib import Path
from holosoma.utils.video_utils import H264StreamEncoder
encoder = H264StreamEncoder(Path({str(tmp_path / "clip.mp4")!r}), 32, 32, 25.0)
for i in range(5):
    encoder.write(np.full((32, 32, 3), 40 * i, dtype=np.uint8))
import os; os._exit(1)
"""
    result = subprocess.run([sys.executable, "-c", child], env=os.environ, timeout=60, check=False)
    assert result.returncode == 1
    partial = tmp_path / "clip.mp4.partial"
    for _ in range(100):  # the orphaned ffmpeg finishes on its own after the parent is gone
        if len(_decode(partial)[0]) == 5:
            break
        time.sleep(0.05)
    assert not (tmp_path / "clip.mp4").exists()
    assert len(_decode(partial)[0]) == 5


def test_nvenc_session_cap_zero_uses_libx264_without_probing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_probe(width: int, height: int) -> bool:
        raise AssertionError("the session cap must be checked before probing NVENC")

    monkeypatch.setattr(video_utils, "nvenc_available", no_probe)
    encoder = H264StreamEncoder(tmp_path / "x.mp4", 256, 256, 30.0, codec="h264_nvenc", nvenc_max_sessions=0)
    assert encoder.codec == "libx264"
    encoder.discard()


def test_nvenc_unavailable_falls_back_to_libx264(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_utils, "nvenc_available", lambda *_size: False)
    encoder = H264StreamEncoder(tmp_path / "x.mp4", 256, 256, 30.0, codec="h264_nvenc")
    assert encoder.codec == "libx264"
    for _ in range(3):
        encoder.write(np.zeros((256, 256, 3), dtype=np.uint8))
    assert len(_decode(encoder.finish())[0]) == 3


def test_unknown_codec_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported H.264 encoder"):
        H264StreamEncoder(tmp_path / "x.mp4", 32, 32, 30.0, codec="mpeg4")


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not video_utils.nvenc_available(256, 256), reason="NVENC not available"
)
def test_nvenc_sessions_capped_and_released(tmp_path: Path) -> None:
    frame = np.full((256, 256, 3), (200, 30, 30), dtype=np.uint8)
    first = H264StreamEncoder(tmp_path / "a.mp4", 256, 256, 30.0, codec="h264_nvenc", nvenc_max_sessions=1)
    second = H264StreamEncoder(tmp_path / "b.mp4", 256, 256, 30.0, codec="h264_nvenc", nvenc_max_sessions=1)
    assert (first.codec, second.codec) == ("h264_nvenc", "libx264")
    for _ in range(4):
        first.write(frame)
        second.write(frame)
    paths = [first.finish(), second.finish()]
    third = H264StreamEncoder(tmp_path / "c.mp4", 256, 256, 30.0, codec="h264_nvenc", nvenc_max_sessions=1)
    assert third.codec == "h264_nvenc"  # the first's session was released
    third.discard()
    for path in paths:
        decoded, fps = _decode(path)
        assert fps == pytest.approx(30.0, rel=1e-3)
        assert len(decoded) == 4
        assert decoded[0].mean(axis=(0, 1)) == pytest.approx([200, 30, 30], abs=6)


def test_process_group_sigint_does_not_stop_the_encoder(tmp_path: Path) -> None:
    # A terminal Ctrl-C signals the whole foreground process group. The producer's teardown must
    # still find ffmpeg alive and finalize the partial cleanly (the child exits 0 only if it did).
    child = f"""
import os, signal, time
import numpy as np
from pathlib import Path
from holosoma.utils.video_utils import H264StreamEncoder
encoder = H264StreamEncoder(Path({str(tmp_path / "clip.mp4")!r}), 32, 32, 25.0)
try:
    for i in range(5):
        encoder.write(np.full((32, 32, 3), 40 * i, dtype=np.uint8))
    os.killpg(os.getpgid(0), signal.SIGINT)
    time.sleep(5)
except KeyboardInterrupt:
    encoder.finish(publish=False)
"""
    result = subprocess.run(
        [sys.executable, "-c", child], env=os.environ, timeout=60, check=False, start_new_session=True
    )
    assert result.returncode == 0
    assert not (tmp_path / "clip.mp4").exists()
    assert len(_decode(tmp_path / "clip.mp4.partial")[0]) == 5


@pytest.mark.parametrize("fps", [0.0, -5.0, float("inf"), float("nan")])
def test_bad_fps_raises_before_taking_an_nvenc_session(tmp_path: Path, fps: float) -> None:
    before = H264StreamEncoder._nvenc_open
    with pytest.raises(ValueError, match="fps must be positive"):
        H264StreamEncoder(tmp_path / "x.mp4", 256, 256, fps, codec="h264_nvenc")
    assert H264StreamEncoder._nvenc_open == before


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (FileNotFoundError(2, "No such file"), "ffmpeg not found"),
        (PermissionError(13, "denied"), "could not start ffmpeg"),
    ],
)
def test_failed_ffmpeg_start_releases_the_nvenc_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError, message: str
) -> None:
    monkeypatch.setattr(video_utils, "nvenc_available", lambda _width, _height: True)

    def failing_popen(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(video_utils.subprocess, "Popen", failing_popen)
    before = H264StreamEncoder._nvenc_open
    with pytest.raises(RuntimeError, match=message):
        H264StreamEncoder(tmp_path / "x.mp4", 256, 256, 30.0, codec="h264_nvenc")
    assert H264StreamEncoder._nvenc_open == before
