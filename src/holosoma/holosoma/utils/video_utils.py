from __future__ import annotations

import fcntl
import functools
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from contextlib import suppress
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import numpy.typing as npt
import wandb
from loguru import logger


def _is_wandb_available() -> bool:
    """Check if wandb is initialized and ready for logging.

    Returns
    -------
    bool
        True if wandb is initialized and ready for logging, False otherwise.
    """
    try:
        return wandb.run is not None
    except Exception:
        return False


def overlay_text_on_image(
    image: npt.NDArray[np.uint8],
    text: str,
    position: tuple[int, int] = (50, 50),
    font_scale: float = 1.0,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """Overlay text on an image using OpenCV.

    Args:
        image: numpy array of the image to overlay text on
        text: text string to overlay
        position: (x, y) position for the text
        font_scale: font scale factor
        color: RGB color tuple for the text
        thickness: text thickness

    Returns:
        numpy array: image with text overlaid
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Split long text into multiple lines if needed
    max_width = image.shape[1] - 100  # Leave some margin
    words = text.split(", ")
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + (", " if current_line else "") + word
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]

        if text_size[0] > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    # Draw each line
    text_x, text_y = position
    line_height = int(font_scale * 30)  # Approximate line height

    for i, line in enumerate(lines):
        line_y = text_y + i * line_height
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]

        # Draw black rectangle background for each line
        cv2.rectangle(
            image, (text_x - 10, line_y - text_size[1] - 5), (text_x + text_size[0] + 10, line_y + 5), (0, 0, 0), -1
        )

        # Draw white text
        cv2.putText(image, line, (text_x, line_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return image


def format_command_labels(commands: Any, env_id: int = 0) -> str:
    """Format command values as text labels for overlay on video frames.

    Args:
        commands: numpy array or tensor of command values
        env_id: environment index to get commands from

    Returns:
        str: formatted command labels
    """
    if commands is None:
        return "Commands: Not available"

    # Get commands for the specified environment
    cmd = commands[env_id]

    # Format common command indices based on the command structure
    # Index 0: vx (forward/backward), Index 1: vy (left/right)
    # Index 2: angular velocity (calculated), Index 3: target heading
    labels = []

    if len(cmd) > 0:
        labels.append(f"vx={cmd[0]:.2f}")
    if len(cmd) > 1:
        labels.append(f"vy={cmd[1]:.2f}")
    if len(cmd) > 2:
        labels.append(f"ang_vel={cmd[2]:.2f}")
    if len(cmd) > 5:
        labels.append(f"waist_yaw={cmd[5]:.2f}")
    if len(cmd) > 8:
        labels.append(f"height={cmd[8]:.2f}")

    return ", ".join(labels) if labels else "Commands: No data"


def create_video(
    video_frames: npt.NDArray[np.uint8],
    fps: int,
    save_dir: Path,
    output_format: str = "mp4",
    wandb_logging: bool = True,
    episode_id: int | None = None,
) -> Path | None:
    """Create video with configurable output format and destination.

    Handles both local saving and wandb upload based on configuration.
    Respects output format preference and provides proper wandb validation.

    Parameters
    ----------
    video_frames : np.ndarray
        Video frames array with shape (T, H, W, 3) and dtype uint8.
    fps : int
        Frames per second for video playback.
    save_dir : Path
        Directory to save video files.
    output_format : str, default="mp4"
        Output format: "mp4" (mp4v codec, written by OpenCV) or "h264" (H.264, one libx264 pass
        via :func:`encode_h264`).
    wandb_logging : bool, default=True
        Whether to upload to wandb (if available) or save locally.
    episode_id : int | None, default=None
        Episode ID for filename generation.

    Returns
    -------
    Path | None
        Path to the saved video file, or None if saving failed.
    """
    shape: tuple[int, ...] = video_frames.shape
    h, w = shape[1], shape[2]

    # Ensure the directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate file names
    timestamp = int(time.time())
    episode_str = f"episode_{episode_id}_" if episode_id is not None else ""

    if wandb_logging and _is_wandb_available():
        # Wandb path: write a temp file, upload, then delete it
        temp_id = str(uuid.uuid4())[:8]
        final_video = save_dir / f"temp_{output_format}_{timestamp}_{temp_id}.mp4"
        cleanup_files = True
    else:
        # Local path: create a permanent file
        final_video = save_dir / f"{episode_str}{timestamp}.mp4"
        cleanup_files = False

    try:
        if output_format == "h264":
            # One libx264 pass straight from the raw frames
            encode_h264(video_frames, fps, final_video)
        else:
            # mp4v straight from OpenCV
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            out = cv2.VideoWriter(str(final_video), fourcc, fps, (w, h))
            if not out.isOpened():
                raise RuntimeError(f"cv2.VideoWriter could not open {final_video}")
            try:
                for frame in video_frames:
                    # Convert RGB to BGR for OpenCV
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except BaseException:
                out.release()
                final_video.unlink(missing_ok=True)
                raise
            out.release()

        logger.info(f"Successfully saved video file: {final_video}")

        if wandb_logging and _is_wandb_available():
            wandb.log({"Training rollout": wandb.Video(str(final_video), format="mp4")})

        if cleanup_files:
            final_video.unlink(missing_ok=True)
            return None
        return final_video

    except Exception as e:
        raise RuntimeError("Video creation failed") from e


PARTIAL_SUFFIX = ".partial"
"""Suffix of a video still being encoded. It is renamed to its final name only once complete, so a run
that crashes or is interrupted leaves ``<name>.mp4.partial`` behind rather than a complete-looking mp4."""


@functools.cache
def nvenc_available(width: int, height: int) -> bool:
    """Whether ffmpeg can open an ``h264_nvenc`` encoder at ``width x height`` here (probed once per size).

    Probed at the real size because NVENC has a minimum frame size (128x128 is rejected) on top of
    needing an NVIDIA GPU, a driver, and an ffmpeg built with it.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:d=0.04",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class H264StreamEncoder:
    """One ffmpeg process encoding RGB uint8 ``[H, W, 3]`` frames to H.264 as they are written.

    Frames stream as raw rgb24 into ffmpeg's stdin, so nothing is buffered here and the encode runs
    alongside whatever produces the frames. The video is written to ``<path>.partial`` and renamed to
    ``path`` only by :meth:`finish`, so a crash or interrupt never leaves a complete-looking file.

    ``codec="h264_nvenc"`` falls back to ``libx264`` when NVENC cannot open at this size, or when
    ``nvenc_max_sessions`` encoders of this process already hold an NVENC session (consumer GPUs cap
    concurrent sessions; a streamed video holds its session until it finishes, so waiting for one
    would stall the producer). :attr:`codec` is the encoder actually used.
    """

    _nvenc_lock: ClassVar[threading.Lock] = threading.Lock()
    _nvenc_open: ClassVar[int] = 0
    """NVENC sessions held by open encoders of this process."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        *,
        codec: str = "libx264",
        crf: int = 23,
        preset: str | None = None,
        gop: int | None = None,
        nvenc_max_sessions: int | None = 8,
    ) -> None:
        self.path = path
        self.partial = path.with_name(path.name + PARTIAL_SUFFIX)
        self.shape = (height, width, 3)
        self.frames = 0
        self.codec = "libx264"
        self._holds_session = False
        if not 0 < fps < float("inf"):
            raise ValueError(f"fps must be positive and finite, got {fps}")
        rate = Fraction(fps).limit_denominator(1_000_000)
        self.fps = float(rate)
        if codec == "h264_nvenc":
            with H264StreamEncoder._nvenc_lock:
                if nvenc_max_sessions is not None and H264StreamEncoder._nvenc_open >= nvenc_max_sessions:
                    logger.info(f"[VIDEO] {nvenc_max_sessions} NVENC session(s) in use; {path.name} uses libx264")
                elif not nvenc_available(width, height):
                    logger.info(f"[VIDEO] NVENC unavailable at {width}x{height}; {path.name} uses libx264")
                else:
                    self.codec = "h264_nvenc"
                    self._holds_session = True
                    H264StreamEncoder._nvenc_open += 1
        elif codec != "libx264":
            raise ValueError(f"unsupported H.264 encoder {codec!r}; expected 'libx264' or 'h264_nvenc'")
        if self.codec == "h264_nvenc":
            quality = ["-preset", preset or "p4", "-tune", "hq", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
        else:
            quality = ["-preset", preset or "medium", "-crf", str(crf)]
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            f"{rate.numerator}/{rate.denominator}",
            "-i",
            "-",
            "-c:v",
            self.codec,
            "-pix_fmt",
            "yuv420p",
            *quality,
            *(["-g", str(gop)] if gop is not None else []),
            "-f",
            "mp4",
            str(self.partial),
        ]
        # stderr goes to a file, not a pipe: an unread stderr pipe that fills up would block ffmpeg.
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 -- lives as long as ffmpeg; closed in _cleanup
        try:
            # Unbuffered: each frame reaches ffmpeg when written, not when a Python buffer fills.
            # Own session: a terminal Ctrl-C (SIGINT to the whole process group) must not stop ffmpeg
            # mid-file; the producer's teardown decides how the file ends. If the producer dies,
            # ffmpeg sees EOF on stdin and finalizes what it has.
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
                bufsize=0,
                start_new_session=True,
            )
        except OSError as e:
            self._stderr.close()
            self._release_session()
            if isinstance(e, FileNotFoundError):
                raise RuntimeError("Video creation failed: ffmpeg not found on PATH") from e
            raise RuntimeError(f"Video creation failed: could not start ffmpeg: {e}") from e
        # A pipe that holds a whole frame lets write() return without waiting for ffmpeg to read it.
        # Linux only, and an unprivileged process is capped at /proc/sys/fs/pipe-max-size (1 MiB by
        # default), so this falls back to that cap and then to the default 64 KiB.
        assert self._proc.stdin is not None
        for size in (width * height * 3, 1 << 20):
            try:
                fcntl.fcntl(self._proc.stdin.fileno(), fcntl.F_SETPIPE_SZ, size)
                break
            except (AttributeError, OSError):
                continue

    def write(self, frame: npt.NDArray[np.uint8]) -> None:
        """Hand one frame to the encoder; raises ``RuntimeError`` if ffmpeg has died."""
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 frames of shape {self.shape}, got {frame.dtype} {frame.shape}")
        assert self._proc.stdin is not None
        data = memoryview(np.ascontiguousarray(frame)).cast("B")
        try:
            while data:  # an unbuffered pipe write may take only part of the frame
                data = data[self._proc.stdin.write(data) :]
        except (BrokenPipeError, ValueError) as e:  # ValueError: stdin already closed
            raise self._failure() from e
        self.frames += 1

    def end_input(self) -> None:
        """Signal end of stream so ffmpeg starts flushing; :meth:`finish` then waits for it.

        Calling this on every encoder before finishing any lets their flushes run in parallel.
        """
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass  # ffmpeg exited early; its return code says why

    def finish(self, publish: bool = True) -> Path:
        """End input, wait for ffmpeg; rename to the final path when ``publish``, else keep ``.partial``.

        Returns the file left on disk. Raises ``RuntimeError`` (keeping the partial file) if ffmpeg failed.
        """
        self.end_input()
        if self._proc.wait() != 0:
            raise self._failure()
        self._cleanup()
        if not publish:
            return self.partial
        self.partial.replace(self.path)
        return self.path

    def discard(self) -> None:
        """Stop ffmpeg and delete its output."""
        self.abort()
        self.partial.unlink(missing_ok=True)

    def abort(self) -> None:
        """Stop ffmpeg at once, leaving whatever it wrote at ``.partial``."""
        self._stop_process()
        self._cleanup()

    def _stop_process(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        with suppress(OSError):
            if self._proc.stdin is not None:
                self._proc.stdin.close()

    def _failure(self) -> RuntimeError:
        self._stop_process()
        self._stderr.seek(0)
        detail = self._stderr.read().decode(errors="replace").strip()[-1000:]
        self._cleanup()
        return RuntimeError(f"ffmpeg ({self.codec}) failed encoding {self.partial}: {detail}")

    def _cleanup(self) -> None:
        if not self._stderr.closed:
            self._stderr.close()
        self._release_session()

    def _release_session(self) -> None:
        if self._holds_session:
            self._holds_session = False
            with H264StreamEncoder._nvenc_lock:
                H264StreamEncoder._nvenc_open -= 1


def retime_mp4(path: Path, scale: float) -> None:
    """Scale every timestamp and frame duration of the mp4 at ``path`` by ``scale`` in place, without re-encoding.

    ``setts`` rather than ``-itsscale``: the latter leaves packet durations unscaled, so the
    container's average frame rate (what OpenCV reports as fps) stays wrong.
    """
    tmp = path.with_name(path.name + ".retime" + PARTIAL_SUFFIX)
    k = f"{scale:.9f}"
    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path), "-c", "copy"]
    cmd += ["-bsf:v", f"setts=pts=PTS*{k}:dts=DTS*{k}:duration=DURATION*{k}", "-f", "mp4", str(tmp)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed retiming {path}: {result.stderr.strip()[-1000:]}")
    tmp.replace(path)


def encode_h264(
    frames: Sequence[npt.NDArray[np.uint8]] | npt.NDArray[np.uint8],
    fps: float,
    path: Path,
    crf: int = 23,
    preset: str = "medium",
) -> Path:
    """Encode RGB uint8 ``[H, W, 3]`` frames to an H.264 mp4 at ``path`` in a single libx264 pass.

    Frames stream into ffmpeg one at a time, so there is no stacked copy of the clip and no
    intermediate lossy mp4v file. Raises ``RuntimeError`` if ffmpeg is missing or fails, and
    ``ValueError`` on a frame of the wrong shape or dtype; no output file is left behind on failure.
    """
    h, w = frames[0].shape[:2]
    encoder = H264StreamEncoder(path, w, h, fps, crf=crf, preset=preset)
    try:
        for frame in frames:
            encoder.write(frame)
        return encoder.finish()
    except BaseException:
        encoder.discard()
        raise
