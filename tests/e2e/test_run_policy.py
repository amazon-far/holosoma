"""Sanity check tests for holosoma_inference/holosoma_inference/run_policy.py commands.

The goal is to test the run_policy commands as closely as possible to how the user would run them,
ensuring they start up correctly and run for 30 seconds each.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.absolute()


def run_mini_training(workflow_name: str) -> str:
    """Run a mini training and return the checkpoint path."""
    env = os.environ.copy()
    env["EXPORT_ONNX"] = "1"

    result = subprocess.run(
        [
            "python",
            f"{REPO_ROOT}/src/holosoma/holosoma/train_agent.py",
            f"exp:{workflow_name}",
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


def assert_run_policy_with_hsinference(config_name: str, model_path: str, timeout: int = 15) -> None:
    """Run policy in hsinference environment with local checkpoint.

    Parameters
    ----------
    config_name : str
        Tyro subcommand name (e.g., 'g1-29dof-loco', 't1-29dof-loco')
    model_path : str
        Path to the ONNX model checkpoint
    timeout : int
        Timeout in seconds for the policy to run
    """
    policy_process = subprocess.Popen(
        [
            "/bin/bash",
            "-c",
            f"source {REPO_ROOT}/scripts/source_inference_setup.sh && "
            f"pip install -e {REPO_ROOT}/src/holosoma_inference[unitree,booster] && "
            f"python {REPO_ROOT}/src/holosoma_inference/holosoma_inference/run_policy.py "
            f"inference:{config_name} "
            f"--task.model-path={model_path} "
            f"--secondary none",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )

    try:
        stdout, stderr = policy_process.communicate(timeout=timeout)
        output = stdout + stderr
        assert "RL FPS:" in output, f"Didn't find RL FPS in output! Output: {output}"
    except subprocess.TimeoutExpired:
        # Kill process and get partial output
        policy_process.terminate()
        stdout, stderr = policy_process.communicate(timeout=5)
        output = stdout + stderr
        assert "RL FPS:" in output, f"Didn't find RL FPS in output! Output: {output}"
    finally:
        try:
            policy_process.terminate()
        except (ProcessLookupError, OSError):
            pass


@pytest.mark.requires_inference
@pytest.mark.parametrize(
    ("workflow_name", "config_name"),
    [
        ("g1-29dof", "g1-29dof-loco"),
        # ("t1-29dof", "t1-29dof-loco"),
    ],
)
def test_run_policy_with_trained_checkpoint(workflow_name: str, config_name: str):
    """Train a mini model and test running it in hsinference environment.

    Parameters
    ----------
    workflow_name : str
        Training workflow name (e.g., 'g1-29dof')
    config_name : str
        Tyro subcommand name for run_policy (e.g., 'g1-29dof-loco')
    """
    checkpoint_path = run_mini_training(workflow_name)
    assert_run_policy_with_hsinference(config_name, checkpoint_path)
