"""Structured low-command send logger.

Replaces the ad-hoc ``/tmp/blah.txt`` + ``/tmp/lowstate.txt`` writes that
were sprinkled into ``UnitreeInterface.send_low_command`` during the
2026-05-02 G1 bring-up. Those were informative once but grepping them is
unpleasant and they're not unit-testable.

Usage
-----
    from holosoma_inference.sdk.send_log import SendLogger
    self._send_logger = SendLogger()
    ...
    self._send_logger.maybe_log(q_target=..., kp=..., kd=..., unitree=self.unitree_interface)

Env vars
--------
HOLOSOMA_SEND_LOG         "1" to enable (default off).
HOLOSOMA_SEND_LOG_EVERY   int, default 100. Log every Nth frame.
HOLOSOMA_SEND_LOG_DIR     dir for jsonl files; default /tmp.
HOLOSOMA_SEND_LOG_INCLUDE_STATE  "1" to call read_low_state each log frame
                                 (slower; only useful on hardware).
HOLOSOMA_SEND_LOG_FULL           "1" to log all 29 joint values instead of
                                 just the first 6 (default off; files grow
                                 ~4.8x when enabled). Enable when the
                                 analyzer needs upper-body coverage.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "1" if default else "0")
    return v not in ("", "0", "false", "False", "no", "NO")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


class SendLogger:
    """Per-interface JSONL ring logger for outgoing commands."""

    def __init__(self) -> None:
        self._counter = 0
        self._path: Path | None = None
        self._enabled_cached: bool | None = None

    @property
    def enabled(self) -> bool:
        if self._enabled_cached is None:
            self._enabled_cached = _env_bool("HOLOSOMA_SEND_LOG", False)
        return self._enabled_cached

    def _resolve_path(self) -> Path:
        if self._path is None:
            d = Path(os.environ.get("HOLOSOMA_SEND_LOG_DIR", "/tmp"))
            d.mkdir(parents=True, exist_ok=True)
            self._path = d / f"holosoma_send_log-{os.getpid()}.jsonl"
        return self._path

    def maybe_log(
        self,
        *,
        q_target,
        kp,
        kd,
        unitree: Any = None,
    ) -> None:
        if not self.enabled:
            return
        self._counter += 1
        every = _env_int("HOLOSOMA_SEND_LOG_EVERY", 100)
        if every <= 0 or self._counter % every != 0:
            return
        full = _env_bool("HOLOSOMA_SEND_LOG_FULL", False)
        q_slice = list(q_target) if full else list(q_target[:6])
        record: dict[str, Any] = {
            "ts": time.time(),
            "pid": os.getpid(),
            "frame": self._counter,
            "q_target": [float(v) for v in q_slice],
            "kp_mean": float(sum(kp) / max(1, len(kp))),
            "kd_mean": float(sum(kd) / max(1, len(kd))),
        }
        if _env_bool("HOLOSOMA_SEND_LOG_INCLUDE_STATE", False) and unitree is not None:
            try:
                state = unitree.read_low_state()
                # Match q_target's truncation: FULL=1 captures all 29 motors,
                # FULL=0 stays at the first 6 for leg-focused debug. imu_quat
                # is always 4 floats.
                q_list = list(state.motor.q) if full else list(state.motor.q)[:6]
                dq_list = list(state.motor.dq) if full else list(state.motor.dq)[:6]
                record["state"] = {
                    "q": [float(v) for v in q_list],
                    "dq": [float(v) for v in dq_list],
                    "imu_quat": [float(v) for v in list(state.imu.quat)],
                }
            except Exception as exc:
                record["state_error"] = repr(exc)
        try:
            with open(self._resolve_path(), "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:  # noqa: S110
            # Deliberate: never crash the control loop on a log-write
            # failure. The logger is diagnostic and must not influence
            # the inference tick cadence.
            pass
