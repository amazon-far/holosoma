"""Unit tests for the low-command SendLogger."""

from __future__ import annotations

import json

import pytest

from holosoma_inference.sdk.send_log import SendLogger


@pytest.fixture
def tmp_log_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HOLOSOMA_SEND_LOG_DIR", str(tmp_path))
    return tmp_path


def test_disabled_by_default_writes_nothing(tmp_log_dir, monkeypatch):
    monkeypatch.delenv("HOLOSOMA_SEND_LOG", raising=False)
    log = SendLogger()
    for _ in range(1000):
        log.maybe_log(q_target=[0.0] * 29, kp=[100.0] * 29, kd=[5.0] * 29)
    assert list(tmp_log_dir.glob("*.jsonl")) == []


def test_respects_every_n_frames(tmp_log_dir, monkeypatch):
    monkeypatch.setenv("HOLOSOMA_SEND_LOG", "1")
    monkeypatch.setenv("HOLOSOMA_SEND_LOG_EVERY", "50")
    log = SendLogger()
    for i in range(200):
        log.maybe_log(q_target=[float(i)] * 29, kp=[100.0] * 29, kd=[5.0] * 29)
    files = list(tmp_log_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 4  # 50, 100, 150, 200
    rec = json.loads(lines[0])
    assert rec["frame"] == 50
    assert rec["kp_mean"] == 100.0


def test_skips_state_when_unitree_none(tmp_log_dir, monkeypatch):
    monkeypatch.setenv("HOLOSOMA_SEND_LOG", "1")
    monkeypatch.setenv("HOLOSOMA_SEND_LOG_EVERY", "1")
    monkeypatch.setenv("HOLOSOMA_SEND_LOG_INCLUDE_STATE", "1")
    log = SendLogger()
    log.maybe_log(q_target=[0.0] * 29, kp=[100.0] * 29, kd=[5.0] * 29, unitree=None)
    # No crash; file contains one record without a 'state' key.
    files = list(tmp_log_dir.glob("*.jsonl"))
    rec = json.loads(files[0].read_text().splitlines()[0])
    assert "state" not in rec


def test_truncates_q_target_by_default(tmp_log_dir, monkeypatch):
    monkeypatch.setenv("HOLOSOMA_SEND_LOG", "1")
    monkeypatch.setenv("HOLOSOMA_SEND_LOG_EVERY", "1")
    monkeypatch.delenv("HOLOSOMA_SEND_LOG_FULL", raising=False)
    log = SendLogger()
    log.maybe_log(q_target=list(range(29)), kp=[100.0] * 29, kd=[5.0] * 29)
    rec = json.loads(next(iter(tmp_log_dir.glob("*.jsonl"))).read_text().splitlines()[0])
    assert len(rec["q_target"]) == 6


def test_logs_full_q_target_when_requested(tmp_log_dir, monkeypatch):
    monkeypatch.setenv("HOLOSOMA_SEND_LOG", "1")
    monkeypatch.setenv("HOLOSOMA_SEND_LOG_EVERY", "1")
    monkeypatch.setenv("HOLOSOMA_SEND_LOG_FULL", "1")
    log = SendLogger()
    log.maybe_log(q_target=list(range(29)), kp=[100.0] * 29, kd=[5.0] * 29)
    rec = json.loads(next(iter(tmp_log_dir.glob("*.jsonl"))).read_text().splitlines()[0])
    assert len(rec["q_target"]) == 29
    assert rec["q_target"][-1] == 28.0
