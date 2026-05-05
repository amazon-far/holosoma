from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import viser  # type: ignore[import-not-found]


def build_record_frame_indices(
    *,
    n_frames: int,
    start_frame: int,
    end_frame: int | None,
    stride: int,
) -> list[int]:
    """Build a validated inclusive frame range for deterministic recording."""
    if n_frames <= 0:
        raise ValueError("Cannot record an empty sequence.")
    if stride <= 0:
        raise ValueError("record_stride must be positive.")

    start = max(0, int(start_frame))
    end = n_frames - 1 if end_frame is None else min(n_frames - 1, int(end_frame))
    if start > end:
        raise ValueError(f"Invalid recording frame range: start={start}, end={end}.")
    return list(range(start, end + 1, int(stride)))


def record_viser_sequence(
    *,
    server: viser.ViserServer,
    apply_frame: Callable[[int], None],
    frame_indices: list[int],
    output_path: str,
    width: int,
    height: int,
    fps: float,
    connect_timeout: float = 120.0,
    start_delay: float = 3.0,
    settle_time: float = 0.02,
    warmup_renders: int = 1,
    transport_format: str = "jpeg",
) -> None:
    """Capture a Viser scene sequence from a connected browser client.

    Viser renders from a browser client, so recording requires opening the viewer
    URL first. The current client camera is used for every captured frame.
    """
    if not frame_indices:
        raise ValueError("No frames were selected for recording.")
    if width <= 0 or height <= 0:
        raise ValueError("record_width and record_height must be positive.")
    if fps <= 0:
        raise ValueError("record_fps must be positive.")
    if warmup_renders < 0:
        raise ValueError("record_warmup_renders cannot be negative.")
    if transport_format not in ("jpeg", "png"):
        raise ValueError("record_transport_format must be 'jpeg' or 'png'.")

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "Video recording requires imageio with ffmpeg support. "
            "Install the retargeting package dependencies or run: pip install 'imageio[ffmpeg]'"
        ) from exc

    client = _wait_for_client(server, timeout_s=connect_timeout)
    if start_delay > 0.0:
        print(f"[recording] Client connected. Recording starts in {start_delay:.1f}s.")
        time.sleep(start_delay)
        client = _ensure_connected_client(server, client, timeout_s=connect_timeout)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[recording] Capturing {len(frame_indices)} frames at {width}x{height}, "
        f"{fps:g} FPS -> {output}"
    )
    print(
        "[recording] The browser view is now driven frame-by-frame for capture; "
        "the saved video uses the requested FPS."
    )

    with imageio.get_writer(str(output), fps=fps) as writer:
        for i, frame_idx in enumerate(frame_indices, start=1):
            client = _ensure_connected_client(server, client, timeout_s=connect_timeout)
            _apply_frame_atomically(server, frame_idx, apply_frame)
            _flush_server(server)
            _flush_client(client)
            if settle_time > 0.0:
                time.sleep(settle_time)
            for _ in range(warmup_renders):
                client.get_render(
                    height=height,
                    width=width,
                    transport_format=transport_format,  # type: ignore[arg-type]
                )
            frame = client.get_render(
                height=height,
                width=width,
                transport_format=transport_format,  # type: ignore[arg-type]
            )
            writer.append_data(_as_rgb_uint8(frame))
            if i == 1 or i == len(frame_indices) or i % 25 == 0:
                print(f"[recording] Captured {i}/{len(frame_indices)} frames")

    print(f"[recording] Saved video: {output}")


def _apply_frame_atomically(
    server: viser.ViserServer,
    frame_idx: int,
    apply_frame: Callable[[int], None],
) -> None:
    atomic = getattr(server, "atomic", None)
    if callable(atomic):
        with atomic():
            apply_frame(frame_idx)
    else:
        apply_frame(frame_idx)


def _wait_for_client(server: viser.ViserServer, *, timeout_s: float) -> Any:
    deadline = time.time() + timeout_s
    print("[recording] Waiting for a browser client. Open the Viser URL printed above.")
    while True:
        clients = server.get_clients()
        if clients:
            return next(iter(clients.values()))
        if time.time() >= deadline:
            raise TimeoutError(f"No Viser browser client connected within {timeout_s:.1f}s.")
        time.sleep(0.1)


def _ensure_connected_client(server: viser.ViserServer, client: Any, *, timeout_s: float) -> Any:
    clients = server.get_clients()
    client_id = getattr(client, "client_id", None)
    if client_id in clients:
        return clients[client_id]
    if clients:
        replacement = next(iter(clients.values()))
        print(f"[recording] Browser client changed; using client {getattr(replacement, 'client_id', 'unknown')}.")
        return replacement
    print("[recording] Browser client disconnected. Waiting for a new client.")
    return _wait_for_client(server, timeout_s=timeout_s)


def _flush_client(client: Any) -> None:
    flush = getattr(client, "flush", None)
    if callable(flush):
        flush()


def _flush_server(server: viser.ViserServer) -> None:
    flush = getattr(server, "flush", None)
    if callable(flush):
        flush()


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA render frame, got shape {arr.shape}.")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr
