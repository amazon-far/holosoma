"""End-to-end training and evaluation tests.

The goal is to test the training and evaluation commands as closely as possible to how the user would run them.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
from pathlib import Path
from threading import Thread

import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent.parent.absolute()

# TODO: Restore whole_body_tracking test after transition to manager-based env


# TODO: Figure out a better way to extract local checkpoint path for eval later
def extract_checkpoint_path(training_output: str) -> str | None:
    """Extract checkpoint path from training output."""
    # Look for the specific log message with full path
    patterns = [
        r"Saving checkpoint to\s+(.+\.pt)",
        r"Saved model at:\s*(.+\.pt)",
        r"Checkpoint saved:\s*(.+\.pt)",
        r"Model saved to:\s*(.+\.pt)",
        r"checkpoint saved to\s*(.+\.pt)",
        r"(logs/[^/\s]+/[^/\s]+/model_\d+\.pt)",  # Full path pattern
    ]

    for pattern in patterns:
        matches = re.findall(pattern, training_output, re.IGNORECASE)
        if matches:
            return matches[-1]  # Return the last checkpoint

    return None


def tee(infile, *files):
    """Print `infile` to `files` in a separate thread."""

    def fanout(infile, *files):
        with infile:
            for line in iter(infile.readline, b""):
                for f in files:
                    f.write(line)
                    f.flush()

    t = Thread(target=fanout, args=(infile,) + files)
    t.daemon = True
    t.start()
    return t


def teed_call(cmd_args, stdout=None, stderr=None, **kwargs):
    p = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE if stdout is not None else None,
        stderr=subprocess.PIPE if stderr is not None else None,
        **kwargs,
    )
    threads = []
    if stdout is not None:
        threads.append(tee(p.stdout, stdout, sys.stdout.buffer))
    if stderr is not None:
        threads.append(tee(p.stderr, stderr, sys.stderr.buffer))
    for t in threads:
        t.join()  # wait for IO completion
    return p.wait()


def assert_training_and_eval_workflow(
    workflow_name: str, multi_gpu: bool = False, extra_config: list[str] | None = None
) -> None:
    """Run evaluation after training and assert whether return code is 0."""
    executable = ["torchrun", f"--nproc_per_node={torch.cuda.device_count()}"] if multi_gpu else ["python"]

    train_stdout = io.BytesIO()
    train_stderr = io.BytesIO()

    # Run training and capture output to get checkpoint

    train_cmd = [
        *executable,
        f"{REPO_ROOT}/src/holosoma/holosoma/train_agent.py",
        f"exp:{workflow_name}",
        "terrain:terrain-locomotion-plane",
        "--algo.config.num-learning-iterations=2",
        "--training.num-envs=128",
        "--logger.video.enabled=False",
    ]
    train_cmd.extend(extra_config or [])

    print("Train command: ", subprocess.list2cmdline(train_cmd))

    train_result = teed_call(train_cmd, stdout=train_stdout, stderr=train_stderr)

    assert train_result == 0

    train_stdout_content = train_stdout.getvalue().decode("utf-8")
    train_stderr_content = train_stderr.getvalue().decode("utf-8")

    # Extract checkpoint and run eval
    checkpoint_path = extract_checkpoint_path(train_stdout_content + train_stderr_content)
    assert checkpoint_path is not None, f"Could not extract checkpoint path for {workflow_name}"

    print(f"Using checkpoint for eval: {checkpoint_path}")

    # Extract wandb run info from training output if available (tests only use local checkpoints)
    eval_cmd = [
        "python",
        f"{REPO_ROOT}/src/holosoma/holosoma/eval_agent.py",
        f"--checkpoint={checkpoint_path}",
        "--training.headless=True",
        "--training.max-eval-steps=4",
    ]
    print("Eval command: ", subprocess.list2cmdline(eval_cmd))
    subprocess.check_call(eval_cmd)


WORKFLOWS = [
    "t1-29dof",
    "g1-29dof",
    "t1-29dof-fast-sac",
    "g1-29dof-fast-sac",
]

ISAACSIM_ONLY_WORKFLOWS = [
    "g1-29dof-wbt",
    "g1-29dof-wbt-fast-sac",
    "g1-29dof-wbt-w-object",
    "g1-29dof-wbt-fast-sac-w-object",
]


# Default workflows run on simulator.isaacgym (the experiment-preset default).
@pytest.mark.isaacgym
@pytest.mark.parametrize(
    "workflow_name",
    WORKFLOWS,
)
@pytest.mark.parametrize(
    "multi_gpu",
    [
        pytest.param(False, id="single-gpu"),
        pytest.param(True, marks=pytest.mark.multi_gpu, id="multi-gpu"),
    ],
)
def test_training_and_eval_workflow(workflow_name: str, multi_gpu: bool):
    assert_training_and_eval_workflow(workflow_name, multi_gpu=multi_gpu)


@pytest.mark.isaacsim
@pytest.mark.parametrize(
    "workflow_name",
    WORKFLOWS + ISAACSIM_ONLY_WORKFLOWS,
)
@pytest.mark.parametrize("multi_gpu", [False])
def test_training_and_eval_workflow_isaacsim(workflow_name: str, multi_gpu: bool):
    assert_training_and_eval_workflow(workflow_name, multi_gpu=multi_gpu, extra_config=["simulator:isaacsim"])
