"""Configuration for the native depth-conditioned policy, without Python inference."""

from __future__ import annotations

from dataclasses import field

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ContextualConfig:
    command_dim: int = 15
    depth_height: int = 58
    depth_width: int = 87
    require_climb_head: bool = True
    force_manual_climbing: bool = False
    climb_enter: str = "dz_or_pred"
    climb_exit: str = "both_off"
    discretizer: str = "simple"
    yaw_request_gain: float = 2.0
    switch_margin: float = 0.0
    velocity_primitives: dict[str, list[float]] = field(default_factory=dict)
    forward_distance_scale: float = 0.5
    yaw_distance_scale: float = 1.2
    in_place_yaw_threshold: float | None = None
