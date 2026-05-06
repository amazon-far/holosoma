"""Unit tests for ``holosoma_inference.policies.tracking_source``.

Covers the contract that ``TrackingSource`` implementations must
satisfy, and confirms the default ``NullTrackingSource`` behavior
produces the no-op fallback that keeps pre-change behavior
byte-identical.
"""

from __future__ import annotations

import numpy as np

from holosoma_inference.policies.tracking_source import (
    NullTrackingSource,
    TrackingPayload,
    TrackingSource,
)


class TestNullTrackingSource:
    def test_always_returns_none(self) -> None:
        src = NullTrackingSource()
        for _ in range(5):
            assert src.get_latest() is None

    def test_satisfies_protocol(self) -> None:
        src = NullTrackingSource()
        assert isinstance(src, TrackingSource)


class TestTrackingPayload:
    def test_default_payload_is_idle(self) -> None:
        payload = TrackingPayload()
        assert payload.mode == 3  # IDLE
        assert payload.forward_gripper_commands is False
        assert payload.joint_names == []
        assert payload.hand_joint_names == []
        assert payload.left_gripper_value == 0.0
        assert payload.right_gripper_value == 0.0

    def test_smplh_24_joint_body_fits(self) -> None:
        n = 24
        payload = TrackingPayload(
            joint_names=[f"j{i}" for i in range(n)],
            joint_transforms=np.zeros(n * 7, dtype=np.float32),
            joint_confidences=np.ones(n, dtype=np.float32),
            device_type="pico",
            tracking_quality=2,
            mode=1,
        )
        assert len(payload.joint_names) == n
        assert payload.joint_transforms.shape == (n * 7,)
        assert payload.joint_confidences.shape == (n,)
        assert payload.device_type == "pico"
        assert payload.mode == 1

    def test_hand_joints_are_optional(self) -> None:
        payload = TrackingPayload(joint_names=["head"])
        assert payload.hand_joint_names == []
        assert payload.hand_joint_transforms.size == 0
        assert payload.hand_joint_confidences.size == 0


class _StaticSource:
    """Fixture: returns the same payload once per poll, then ``None``."""

    def __init__(self, payload: TrackingPayload | None) -> None:
        self._payload = payload
        self._served = False

    def get_latest(self) -> TrackingPayload | None:
        if self._served or self._payload is None:
            return None
        self._served = True
        return self._payload


class TestProtocolCompatibility:
    def test_custom_source_satisfies_protocol(self) -> None:
        src = _StaticSource(TrackingPayload())
        assert isinstance(src, TrackingSource)

    def test_source_serves_payload_then_none(self) -> None:
        payload = TrackingPayload(mode=1, device_type="avp")
        src = _StaticSource(payload)
        first = src.get_latest()
        second = src.get_latest()
        assert first is payload
        assert second is None
