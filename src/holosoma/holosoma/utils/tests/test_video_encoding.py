"""Delivered movies when the simulator's OpenCV bundle cannot encode them."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest

from holosoma.utils import video_utils

pytestmark = pytest.mark.no_sim


@pytest.fixture
def no_opencv_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fault injection at the native codec boundary; the fallback and file
    # encoding remain real and are checked through independently decoded pixels.
    monkeypatch.setattr(
        video_utils.cv2,
        "VideoWriter",
        lambda *_args: SimpleNamespace(isOpened=lambda: False, release=lambda: None),
    )


@pytest.mark.parametrize("output_format", ["mp4", "h264"])
def test_video_without_opencv_codec_preserves_frames_and_dimensions(
    tmp_path: Path, no_opencv_encoder: None, output_format: str
) -> None:
    frames = np.zeros((72, 98, 130, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        left = 15 if index < 24 or index >= 48 else 75
        frame[30:60, left : left + 20, 0] = 255
    movie = video_utils.create_video(frames, 50, tmp_path, output_format=output_format, wandb_logging=False)
    assert movie is not None and movie.is_file()
    decoder = imageio_ffmpeg.read_frames(str(movie), pix_fmt="rgb24")
    try:
        metadata = next(decoder)
        assert metadata["size"] == (130, 98), "encoder resized non-macroblock frame dimensions"
        assert metadata["fps"] == 50
        assert metadata["codec"] == {"mp4": "mpeg4", "h264": "h264"}[output_format]
        frame_count = 0
        for index, raw in enumerate(decoder):
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(98, 130, 3).astype(np.int16)
            rows, cols = np.nonzero((rgb[..., 0] > rgb[..., 1] + 40) & (rgb[..., 0] > rgb[..., 2] + 40))
            assert cols.size > 500
            assert abs(float(rows.mean()) - 44.5) < 1
            expected_col = 24.5 if index < 24 or index >= 48 else 84.5
            assert abs(float(cols.mean()) - expected_col) < 1, "encoded video lost or reordered motion"
            frame_count += 1
        assert frame_count == 72
    finally:
        decoder.close()
    assert list(tmp_path.glob("*temp_raw*")) == []


@pytest.mark.parametrize("invalid_output", [False, True], ids=["no-output", "invalid-output"])
def test_video_without_opencv_codec_rejects_failed_encoder(
    tmp_path: Path, no_opencv_encoder: None, monkeypatch: pytest.MonkeyPatch, invalid_output: bool
) -> None:
    # A real process consumes every frame and fails, optionally leaving a
    # nonempty corrupt artifact. File existence cannot stand in for encoding.
    executable = tmp_path / "failed-encoder"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" != "-y" ]; then exit 1; fi\n'
        "for output do :; done\n"
        "cat >/dev/null\n" + ('printf "invalid video" > "$output"\n' if invalid_output else "") + "exit 1\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(executable))
    with pytest.raises(RuntimeError, match="Video creation failed"):
        video_utils.create_video(np.zeros((72, 98, 130, 3), dtype=np.uint8), 50, tmp_path, wandb_logging=False)
    assert list(tmp_path.glob("*.mp4")) == []
