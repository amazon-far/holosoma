from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

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
        Output format: "mp4" (mp4v codec) or "h264" (H.264 codec).
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
        # Wandb path: use temp files, upload, then cleanup
        temp_id = str(uuid.uuid4())[:8]
        temp_raw = save_dir / f"temp_raw_{timestamp}_{temp_id}.mp4"
        final_video = save_dir / f"temp_{output_format}_{timestamp}_{temp_id}.mp4"
        cleanup_files = True
    else:
        # Local path: create permanent files
        final_video = save_dir / f"{episode_str}{timestamp}.mp4"
        temp_raw = save_dir / f"{episode_str}temp_raw_{timestamp}.mp4"
        cleanup_files = False

    temp_files_to_cleanup = []

    try:
        # Step 1: Create intermediate video with OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        out = cv2.VideoWriter(str(temp_raw), fourcc, fps, (w, h))
        temp_files_to_cleanup.append(temp_raw)

        for frame in video_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

        # Step 2: Apply output format
        if output_format == "h264":
            # Convert to H.264 using ffmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_raw),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-maxrate",
                "300k",
                "-preset",
                "medium",
                str(final_video),
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            if result.returncode != 0:
                logger.warning(f"[VIDEO] FFmpeg conversion failed: {result.stderr}")
                logger.info("[VIDEO] Falling back to mp4v format (may have browser compatibility issues)")
                # Fallback to original video
                final_video = temp_raw
        else:
            # Use mp4v format directly - rename and remove from cleanup list
            temp_raw.rename(final_video)
            temp_files_to_cleanup.remove(temp_raw)

        # Log successful video file creation
        logger.info(f"Successfully saved video file: {final_video}")

        # Step 3: Handle wandb upload if requested
        if wandb_logging and _is_wandb_available():
            wandb.log({"Training rollout": wandb.Video(str(final_video), format="mp4")})

        # Step 4: Cleanup temp files if needed
        if cleanup_files:
            if final_video.exists():
                final_video.unlink()
            return None
        return final_video

    except Exception as e:
        raise RuntimeError("Video creation failed") from e
    finally:
        # Always cleanup any remaining temp files
        for temp_file in temp_files_to_cleanup:
            if temp_file.exists():
                temp_file.unlink()
