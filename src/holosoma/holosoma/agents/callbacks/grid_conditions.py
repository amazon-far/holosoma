"""Grid condition manager for multi-condition evaluation sweeps.

Owns the cross-product logic that combines sweep axes (velocity, payload,
push, etc.) into a flat list of per-env conditions.  Any callback can
register a sweep axis; after all registrations, ``finalize()`` expands
the Cartesian product and validates against ``num_envs``.

The manager is created by EvalRecordingCallback (the single recorder) and
shared with other callbacks via ``_require_recording_cb()``.

Terminology:
    axis   — one factor of the grid, added via ``register_axis()``.
             Each axis contributes one "slot" to the Cartesian product.
             An axis can control a single variable (``name="push_force_n"``)
             or several coupled variables
             (``name=["lin_vel_x", "lin_vel_y", "ang_vel_yaw"]``).
    group  — a label used only for metadata output grouping in the NPZ.
             Does not affect the grid logic.
"""

from __future__ import annotations

import itertools
from typing import Any

from loguru import logger


class GridConditionManager:
    """Manages sweep axes and the resulting per-env condition list.

    Lifecycle:
        1. Created during ``on_pre_evaluate_policy`` by the recording callback.
        2. Other callbacks call ``register_axis()`` during their own
           ``on_pre_evaluate_policy`` (callbacks execute in field-declaration
           order from ``EvalCallbacksConfig``).
        3. Recording callback calls ``finalize()`` on the first env step
           (deferred so all axes are registered).
    """

    def __init__(self) -> None:
        self._axes: list[dict[str, Any]] = []
        self.conditions: list[dict[str, Any]] = []
        self.num_conditions: int = 0
        self.warmup_steps: int = 0
        self._finalized: bool = False

    def register_axis(
        self,
        name: str | list[str],
        values: list,
        *,
        group: str = "",
    ) -> None:
        """Register one axis of the evaluation grid.

        The grid is the Cartesian product of all registered axes.

        Single-variable axis (one key per condition entry)::

            cm.register_axis("push_force_n", [100.0, 150.0], group="push")

        Multi-variable axis (coupled keys that vary together)::

            cm.register_axis(
                ["lin_vel_x", "lin_vel_y", "ang_vel_yaw"],
                [(0.5, 0, 0), (1.0, 0, 0), (0, 0.3, 0)],
                group="velocity",
            )

        Args:
            name: A single key string, or a list of keys for coupled variables.
            values: The sweep values for this axis.  For a single key, a flat
                list (e.g. ``[0.5, 1.0]``).  For coupled keys, a list of
                tuples with one element per key.
            group: Logical group for metadata output (e.g. "velocity", "push").

        Must be called before ``finalize()``.
        """
        if self._finalized:
            raise RuntimeError(f"Cannot register axis '{name}' after conditions are finalized.")

        if isinstance(name, str):
            keys = [name]
            normalized_values = [(v,) for v in values]
        else:
            keys = list(name)
            normalized_values = [tuple(v) for v in values]

        existing_keys = {k for ax in self._axes for k in ax["keys"]}
        duplicates = existing_keys & set(keys)
        if duplicates:
            raise RuntimeError(f"Duplicate axis key(s) {duplicates}. Each key can only belong to one axis.")

        self._axes.append(
            {
                "keys": keys,
                "values": normalized_values,
                "group": group,
            }
        )

    def finalize(self, num_envs: int) -> None:
        """Expand the Cartesian product of all axes and validate env count.

        After this call, ``conditions`` and ``num_conditions`` are set.
        """
        if self._finalized:
            return

        # --- Expand Cartesian product of all axes ---
        if not self._axes:
            self.conditions = [{}]
        else:
            value_lists = [ax["values"] for ax in self._axes]
            key_lists = [ax["keys"] for ax in self._axes]
            self.conditions = []
            for combo in itertools.product(*value_lists):
                cond: dict[str, Any] = {}
                for keys, vals in zip(key_lists, combo):
                    cond.update(dict(zip(keys, vals)))
                self.conditions.append(cond)

        self.num_conditions = len(self.conditions)

        if num_envs < self.num_conditions:
            raise RuntimeError(
                f"GridConditionManager: need num_envs >= {self.num_conditions} "
                f"for {self.num_conditions} conditions, but got num_envs={num_envs}. "
                f"Set --training.num_envs={self.num_conditions}"
            )
        if num_envs > self.num_conditions:
            logger.warning(
                f"GridConditionManager: num_envs={num_envs} > {self.num_conditions} conditions. "
                f"Only using first {self.num_conditions} envs."
            )

        self._finalized = True

        logger.info(f"GridConditionManager: finalized {self.num_conditions} conditions from {len(self._axes)} axes")
        for i, cond in enumerate(self.conditions):
            parts = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in cond.items()]
            logger.info(f"  env {i}: {' '.join(parts)}")

    def get_metadata(self) -> dict[str, Any]:
        """Return condition metadata for NPZ recording.

        Given axes registered as::

            cm.register_axis(["lin_vel_x", "lin_vel_y", "ang_vel_yaw"],
                        [(0.5, 0, 0), (1.0, 0, 0)], group="velocity")
            cm.register_axis("push_force_n", [100.0, 150.0], group="push")

        After finalize (4 conditions), returns::

            {
                "num_conditions": 4,
                "grid_conditions": [
                    {"velocity": {"lin_vel_x": 0.5, "lin_vel_y": 0.0,
                                  "ang_vel_yaw": 0.0},
                     "push": {"push_force_n": 100.0}},
                    {"velocity": {"lin_vel_x": 0.5, "lin_vel_y": 0.0,
                                  "ang_vel_yaw": 0.0},
                     "push": {"push_force_n": 150.0}},
                    ...
                ],
            }

        ``grid_conditions`` nests keys under their axis ``group``.
        Keys with no group stay at the top level of each condition dict.
        """
        # Build key→group mapping from axis registrations.
        key_to_group: dict[str, str] = {}
        for ax in self._axes:
            grp = ax.get("group", "")
            for k in ax["keys"]:
                key_to_group[k] = grp

        # Restructure flat conditions into grouped dicts.
        # e.g. {"lin_vel_x": 0.5, "push_force_n": 100}
        #   → {"velocity": {"lin_vel_x": 0.5}, "push": {"push_force_n": 100}}
        hierarchical: list[dict[str, Any]] = []
        for cond in self.conditions:
            grouped: dict[str, Any] = {}
            for key, val in cond.items():
                grp = key_to_group.get(key, "")
                if grp:
                    grouped.setdefault(grp, {})[key] = val
                else:
                    grouped[key] = val
            hierarchical.append(grouped)

        return {
            "num_conditions": self.num_conditions,
            "grid_conditions": hierarchical,
        }
