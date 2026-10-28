"""Sanity check tests for holosoma_inference/holosoma_inference/run_policy.py commands.

The goal is to test the run_policy commands as closely as possible to how the user would run them,
ensuring they start up correctly and emit inference output within 15 seconds.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.absolute()
_SETUP_TIMEOUT = 300
_STOP_TIMEOUT = 5


def run_mini_training(workflow_name: str) -> str:
    """Run a mini training and return the checkpoint path."""
    env = os.environ.copy()
    env["EXPORT_ONNX"] = "1"

    result = subprocess.run(
        [
            "python",
            f"{REPO_ROOT}/src/holosoma/holosoma/train_agent.py",
            f"exp:{workflow_name}",
            # Pin the backend instead of inheriting the preset default: this job trains in the
            # combined image's MuJoCo env, so an implicit default would try to build whatever
            # backend the preset happens to name and fail on a missing SDK. mjwarp (not classic
            # mujoco) keeps this on the GPU, which ClassicBackend cannot do -- it is single-env only.
            "simulator:mjwarp",
            "--algo.config.num-learning-iterations=2",
            "--training.num-envs=4",
            "--logger.video.enabled=False",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, f"Training failed: {result.stderr}"

    # Parse ONNX path from stdout/stderr
    output = result.stdout + result.stderr
    print(output)

    pt_matches = re.findall(r"Saving checkpoint to (.+/model_\d+\.pt)", output)
    if pt_matches:
        # Return the latest checkpoint (last one saved)
        return pt_matches[-1].replace(".pt", ".onnx")

    raise FileNotFoundError("No ONNX checkpoint path found in training output")


def _signal_process_group(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _run_inference_command(command: str, timeout: float) -> tuple[int, str, bool]:
    """Collect output while bounding both execution and whole-group cleanup."""
    process = subprocess.Popen(
        ["/bin/bash", "-c", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    timed_out = False
    try:
        try:
            process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Killing bash alone leaves Python (or pip) holding the output pipe.
            _signal_process_group(process, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                output, _ = process.communicate(timeout=_STOP_TIMEOUT)
        return process.returncode, output, timed_out
    finally:
        # Also stop descendants that closed their pipes or outlived the leader.
        _signal_process_group(process, signal.SIGKILL)


def _activate() -> str:
    return f"source {shlex.quote(str(REPO_ROOT / 'scripts/source_inference_setup.sh'))} && "


def setup_hsinference() -> None:
    """Install the working tree into the hsinference environment."""
    setup_code, setup_output, setup_timed_out = _run_inference_command(
        _activate() + f"pip install -e {shlex.quote(str(REPO_ROOT / 'src/holosoma_inference') + '[unitree,booster]')}",
        timeout=_SETUP_TIMEOUT,
    )
    assert not setup_timed_out, f"Setup timed out after {_SETUP_TIMEOUT}s. Output: {setup_output}"
    assert setup_code == 0, f"Setup exited with {setup_code}. Output: {setup_output}"


def run_policy_with_hsinference(
    config_name: str, model_path: str, timeout: float = 15, extra_args: tuple[str, ...] = ()
) -> tuple[int, str, bool]:
    """Run run_policy in the hsinference environment; return (exit code, output, timed out)."""
    return _run_inference_command(
        _activate()
        + shlex.join(
            [
                "python",
                str(REPO_ROOT / "src/holosoma_inference/holosoma_inference/run_policy.py"),
                f"inference:{config_name}",
                f"--task.model-path={model_path}",
                "--secondary",
                "none",
                *extra_args,
            ]
        ),
        timeout=timeout,
    )


def assert_run_policy_with_hsinference(
    config_name: str,
    model_path: str,
    timeout: float = 15,
    *,
    setup: bool = True,
    required_output: tuple[str, ...] = ("RL FPS:",),
) -> None:
    """Run policy in hsinference environment with local checkpoint.

    Parameters
    ----------
    config_name : str
        Tyro subcommand name (e.g., 'g1-29dof-loco', 't1-29dof-loco')
    model_path : str
        Path to the ONNX model checkpoint
    timeout : float
        Policy observation window in seconds, excluding dependency installation
    setup : bool
        Install the working tree first; False when a fixture already did
    required_output : tuple[str, ...]
        Lines the policy must print within the window
    """
    if setup:
        setup_hsinference()
    policy_code, output, timed_out = run_policy_with_hsinference(config_name, model_path, timeout)
    assert timed_out or policy_code == 0, f"Policy exited with {policy_code}. Output: {output}"
    for line in required_output:
        assert line in output, f"Didn't find {line!r} in output! Output: {output}"


@contextlib.contextmanager
def low_state_publisher(config_name: str, invalid_after: float | None = None, timeout: float = 60):
    """Publish an upright standing low state for the policy, as the simulator bridge would."""
    import holosoma_inference

    robot = holosoma_inference.get_config(config_name).robot
    motor_q = [0.0] * len(robot.default_dof_angles)
    for joint, motor in enumerate(robot.joint2motor):
        motor_q[motor] = robot.default_dof_angles[joint]
    args = ["python", str(Path(__file__).with_name("low_state_publisher.py")), "--motor-q", json.dumps(motor_q)]
    if invalid_after is not None:
        args += ["--invalid-after", str(invalid_after)]
    with tempfile.TemporaryFile("w+", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["/bin/bash", "-c", _activate() + shlex.join(args)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )

        def output() -> str:
            log.seek(0)
            return log.read()

        try:
            deadline = time.monotonic() + timeout
            while "low-state publisher running" not in output():
                assert process.poll() is None, f"Publisher exited with {process.returncode}. Output: {output()}"
                assert time.monotonic() < deadline, f"Publisher did not start. Output: {output()}"
                time.sleep(0.1)
            yield
            assert process.poll() is None, f"Publisher died during the run. Output: {output()}"
        finally:
            _signal_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                process.wait(timeout=_STOP_TIMEOUT)


_INFERENCE_EVIDENCE = ("Received first valid robot state", "Sent first native command", "RL FPS:")


@pytest.fixture(scope="module")
def g1_checkpoint() -> str:
    """Train and export a mini G1 policy once, and install inference once, for this module."""
    checkpoint = run_mini_training("g1-29dof")
    setup_hsinference()
    return checkpoint


@pytest.mark.requires_inference
def test_run_policy_with_trained_checkpoint(g1_checkpoint: str):
    """A trained, exported policy runs native inference on a live robot state and sends commands."""
    with low_state_publisher("g1-29dof-loco"):
        assert_run_policy_with_hsinference(
            "g1-29dof-loco", g1_checkpoint, setup=False, required_output=_INFERENCE_EVIDENCE
        )


@pytest.mark.requires_inference
def test_run_policy_fails_without_a_robot_state(g1_checkpoint: str):
    """Control: with nothing publishing, the policy must fail at the startup timeout, not idle."""
    code, output, timed_out = run_policy_with_hsinference(
        "g1-29dof-loco", g1_checkpoint, extra_args=("--task.startup-state-timeout=3",)
    )
    assert not timed_out and code != 0, f"Policy exited with {code} (timed out: {timed_out}). Output: {output}"
    assert "No valid robot state within 3s" in output, output
    assert "Sent first native command" not in output, output
    assert "RL FPS:" not in output, output


@pytest.mark.requires_inference
def test_run_policy_fails_when_the_robot_state_becomes_invalid(g1_checkpoint: str):
    """Control: an invalid state after startup is an inference failure, not a wait."""
    with low_state_publisher("g1-29dof-loco", invalid_after=5):
        code, output, timed_out = run_policy_with_hsinference("g1-29dof-loco", g1_checkpoint)
    assert not timed_out and code != 0, f"Policy exited with {code} (timed out: {timed_out}). Output: {output}"
    assert "Sent first native command" in output, output
    assert "Robot orientation must be nonzero" in output, output
