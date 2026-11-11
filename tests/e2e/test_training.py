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


def assert_fast_sac_optimizer_updates(checkpoint_path: str) -> None:
    """Read the saved artifact to verify actor and Q optimizers performed finite updates."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for name in ("actor", "q"):
        states = checkpoint[f"{name}_optimizer_state_dict"]["state"]
        assert states, f"FastSAC {name} optimizer has no saved parameter state"
        nonzero_first_moment = False
        for state in states.values():
            step = torch.as_tensor(state["step"])
            assert bool(torch.isfinite(step).all() and (step > 0).all()), (
                f"FastSAC {name} optimizer has no finite positive update count"
            )
            for moment in ("exp_avg", "exp_avg_sq"):
                assert bool(torch.isfinite(state[moment]).all()), f"FastSAC {name} optimizer saved nonfinite {moment}"
            nonzero_first_moment |= bool(torch.count_nonzero(state["exp_avg"]))
        assert nonzero_first_moment, f"FastSAC {name} optimizer saved only zero first moments"


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
    executable = (
        [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={torch.cuda.device_count()}"]
        if multi_gpu
        else [sys.executable]
    )

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
    fast_sac = "fast-sac" in workflow_name
    if fast_sac:
        train_cmd.append("--algo.config.learning-starts=0")
    train_cmd.extend(extra_config or [])

    print("Train command: ", subprocess.list2cmdline(train_cmd))

    train_result = teed_call(train_cmd, stdout=train_stdout, stderr=train_stderr)

    assert train_result == 0

    train_stdout_content = train_stdout.getvalue().decode("utf-8")
    train_stderr_content = train_stderr.getvalue().decode("utf-8")

    # Extract checkpoint and run eval
    checkpoint_path = extract_checkpoint_path(train_stdout_content + train_stderr_content)
    assert checkpoint_path is not None, f"Could not extract checkpoint path for {workflow_name}"
    if fast_sac:
        assert_fast_sac_optimizer_updates(checkpoint_path)

    print(f"Using checkpoint for eval: {checkpoint_path}")

    # Extract wandb run info from training output if available (tests only use local checkpoints)
    eval_cmd = [
        sys.executable,
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


@pytest.mark.isaacsim
@pytest.mark.parametrize(
    "workflow_name",
    WORKFLOWS + ISAACSIM_ONLY_WORKFLOWS,
)
@pytest.mark.parametrize("multi_gpu", [False])
def test_training_and_eval_workflow_isaacsim(workflow_name: str, multi_gpu: bool, isaacsim_simulator: str):
    assert_training_and_eval_workflow(
        workflow_name, multi_gpu=multi_gpu, extra_config=[f"simulator:{isaacsim_simulator}"]
    )


@pytest.mark.isaacsim
def test_newton_training_and_eval_workflow_isaacsim(isaacsim_backend: str, isaacsim_simulator: str):
    if isaacsim_backend != "isaacsim6":
        pytest.skip("Newton training requires the explicit Isaac Lab 3 profile")
    assert_training_and_eval_workflow(
        "g1-29dof-fast-sac",
        extra_config=[
            f"simulator:{isaacsim_simulator}",
            "--simulator.config.sim.isaacsim6.physics-backend=newton",
        ],
    )


@pytest.mark.isaacsim
@pytest.mark.parametrize("physics", ["physx", "newton"])
def test_ovrtx_training_and_eval_workflow_isaacsim(physics, isaacsim_backend: str, isaacsim_simulator: str):
    if isaacsim_backend != "isaacsim6":
        pytest.skip("OVRTX training requires the explicit Isaac Lab 3 profile")
    assert_training_and_eval_workflow(
        "g1-29dof-fast-sac",
        extra_config=[
            f"simulator:{isaacsim_simulator}",
            f"--simulator.config.sim.isaacsim6.physics-backend={physics}",
            "--simulator.config.sim.isaacsim6.renderer=ovrtx",
        ],
    )


# Distributed smoke test for the PR gate: one PPO and one FastSAC workflow under torch.distributed.run
# on every visible GPU. Nightly training covers the full multi-GPU matrix.
@pytest.mark.isaacsim
@pytest.mark.multi_gpu
@pytest.mark.parametrize("workflow_name", ["g1-29dof", "g1-29dof-fast-sac"])
def test_distributed_training_and_eval_workflow_isaacsim(workflow_name: str, isaacsim_simulator: str):
    assert_training_and_eval_workflow(workflow_name, multi_gpu=True, extra_config=[f"simulator:{isaacsim_simulator}"])
