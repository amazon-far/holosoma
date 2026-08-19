"""Gait cycle analysis for foot-height-based phase detection.

Collects per-env foot height data over a configurable analysis window,
counts full gait cycles via zero-crossings, then provides vectorized
per-env gait phase detection.

Phases detected (based on left foot state):
    swing_to_stance = left foot touchdown (air -> ground)
    stance_to_swing = left foot liftoff  (ground -> air)
    mid_stance      = left foot on ground, right foot at peak
    mid_swing       = left foot at peak (in air)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

_PEAK_THRESHOLD = 0.65


class GaitAnalyser:
    """Analyses foot height oscillations to detect gait phases.

    Lifecycle:
        1. Call ``record_foot_heights()`` each step during the analysis window.
        2. Call ``try_finalize()`` once enough steps have elapsed.
           When it returns True, phase detection is ready.
        3. Call ``detect_phases()`` each step to get per-env phase labels.
        4. Call ``get_metadata()`` to retrieve analysis results for recording.
    """

    def __init__(
        self,
        sim: Any,
        left_foot_isaac_id: int,
        right_foot_isaac_id: int,
        num_conditions: int,
        *,
        foot_contact_threshold: float = 0.5,
        min_gait_cycles: int = 3,
    ):
        self._sim = sim
        self._left_foot_isaac_id = left_foot_isaac_id
        self._right_foot_isaac_id = right_foot_isaac_id
        self._num_conditions = num_conditions
        self._foot_contact_threshold = foot_contact_threshold
        self._min_gait_cycles = min_gait_cycles

        self._done = False
        self._left_foot_z_history: list[np.ndarray] = []
        self._right_foot_z_history: list[np.ndarray] = []
        self._left_foot_z_range: tuple[np.ndarray, np.ndarray] | None = None
        self._right_foot_z_range: tuple[np.ndarray, np.ndarray] | None = None
        self._prev_left_on_ground: np.ndarray = np.zeros(num_conditions, dtype=bool)

        self._metadata: dict[str, Any] = {}

    @property
    def done(self) -> bool:
        return self._done

    def record_foot_heights(self) -> None:
        n = self._num_conditions
        left_z = self._sim._robot.data.body_pos_w[:n, self._left_foot_isaac_id, 2]
        right_z = self._sim._robot.data.body_pos_w[:n, self._right_foot_isaac_id, 2]
        self._left_foot_z_history.append(left_z.detach().cpu().numpy().copy())
        self._right_foot_z_history.append(right_z.detach().cpu().numpy().copy())

    def try_finalize(self, *, force: bool = False) -> bool:
        if len(self._left_foot_z_history) < 2:
            return False

        left_z = np.stack(self._left_foot_z_history, axis=0)
        right_z = np.stack(self._right_foot_z_history, axis=0)
        cycles = _count_gait_cycles(left_z)

        if not force and cycles < self._min_gait_cycles:
            return False

        if cycles < self._min_gait_cycles:
            logger.warning(
                f"GaitAnalyser: forced finalization with only "
                f"{cycles} cycles (need {self._min_gait_cycles}). "
                f"Phase detection may be unreliable."
            )

        self._left_foot_z_range = (np.min(left_z, axis=0), np.max(left_z, axis=0))
        self._right_foot_z_range = (np.min(right_z, axis=0), np.max(right_z, axis=0))

        # Initialize ground-contact state from the last recorded frame
        left_range = np.maximum(self._left_foot_z_range[1] - self._left_foot_z_range[0], 1e-4)
        left_norm = (left_z[-1] - self._left_foot_z_range[0]) / left_range
        self._prev_left_on_ground = left_norm < self._foot_contact_threshold

        self._metadata = {
            "gait_left_foot_z_min": self._left_foot_z_range[0].tolist(),
            "gait_left_foot_z_max": self._left_foot_z_range[1].tolist(),
            "gait_right_foot_z_min": self._right_foot_z_range[0].tolist(),
            "gait_right_foot_z_max": self._right_foot_z_range[1].tolist(),
            "gait_analysis_cycles": cycles,
            "gait_analysis_steps": len(self._left_foot_z_history),
        }

        self._left_foot_z_history.clear()
        self._right_foot_z_history.clear()

        logger.info(
            f"GaitAnalyser: analysis complete ({cycles} cycles). "
            f"Left Z: [{self._left_foot_z_range[0].mean():.3f}, {self._left_foot_z_range[1].mean():.3f}], "
            f"Right Z: [{self._right_foot_z_range[0].mean():.3f}, {self._right_foot_z_range[1].mean():.3f}]"
        )

        self._done = True
        return True

    def detect_phases(self) -> np.ndarray:
        """Vectorized gait phase detection for all envs. Single GPU sync."""
        assert self._left_foot_z_range is not None
        assert self._right_foot_z_range is not None

        n = self._num_conditions
        left_z = self._sim._robot.data.body_pos_w[:n, self._left_foot_isaac_id, 2].cpu().numpy()
        right_z = self._sim._robot.data.body_pos_w[:n, self._right_foot_isaac_id, 2].cpu().numpy()

        threshold = self._foot_contact_threshold
        left_min, left_max = self._left_foot_z_range
        right_min, right_max = self._right_foot_z_range

        # --- Normalize foot heights to [0, 1] using observed range ---
        left_range = np.maximum(left_max - left_min, 1e-4)
        right_range = np.maximum(right_max - right_min, 1e-4)

        left_norm = (left_z - left_min) / left_range
        right_norm = (right_z - right_min) / right_range

        # --- Detect ground contact and transitions ---
        left_on_ground = left_norm < threshold
        prev_left_on_ground = self._prev_left_on_ground
        self._prev_left_on_ground = left_on_ground.copy()

        phases: np.ndarray = np.full(n, "", dtype=object)

        # Transitions: left foot crossing the ground threshold
        swing_to_stance = left_on_ground & ~prev_left_on_ground
        stance_to_swing = ~left_on_ground & prev_left_on_ground
        phases[swing_to_stance] = "swing_to_stance"
        phases[stance_to_swing] = "stance_to_swing"

        # Mid-phase: no transition, detect by peak height of the airborne foot
        no_transition = ~swing_to_stance & ~stance_to_swing
        phases[no_transition & ~left_on_ground & (left_norm >= _PEAK_THRESHOLD)] = "mid_swing"
        phases[no_transition & left_on_ground & (right_norm >= _PEAK_THRESHOLD)] = "mid_stance"

        return phases

    def get_metadata(self) -> dict[str, Any]:
        return self._metadata.copy()


def _count_gait_cycles(foot_z: np.ndarray) -> int:
    """Count full gait cycles from zero-crossings of centered foot-Z signal."""
    mean_z = np.mean(foot_z, axis=0, keepdims=True)
    centered = foot_z - mean_z
    sign = np.sign(centered)
    sign[sign == 0] = 1
    crossings = np.sum(np.abs(np.diff(sign, axis=0)) > 0, axis=0)
    cycles = crossings // 2
    return int(np.min(cycles)) if cycles.size > 0 else 0
