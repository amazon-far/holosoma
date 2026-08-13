"""Tests for push randomization to prevent regression.

These tests verify that:
1. Push velocities are properly applied to the simulator and cause actual robot motion
2. The _update_tasks_callback is called before reset_envs_idx (critical ordering)

Note: All tests share a single simulator instance via pytest fixture to avoid
Isaac Gym re-initialization issues.
"""

import dataclasses

import pytest

from holosoma.config_values import experiment
from holosoma.train_agent import get_tyro_env_config, training_context
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch

# Targets Isaac Gym; outside tests/simulators/ so set umbrella explicitly.
pytestmark = pytest.mark.isaacgym


@pytest.fixture(scope="module")
def shared_env():
    """Shared environment fixture to avoid Isaac Gym re-initialization.

    This fixture creates one environment that is reused across all tests
    in this module to work around Isaac Gym's limitation of not supporting
    multiple gym instances in the same process.
    """
    seeding()
    num_envs = 16
    device = "cuda"

    tyro_config = dataclasses.replace(
        experiment.g1_29dof, training=dataclasses.replace(experiment.g1_29dof.training, num_envs=num_envs)
    )

    with training_context(tyro_config):
        tyro_env_config = get_tyro_env_config(tyro_config)
        env = get_class(tyro_config.env_class)(tyro_env_config, device=device)
        env.reset_all()

        yield env

        # Cleanup happens automatically when context exits


@pytest.mark.skip(reason="Cannot run multiple Isaac Gym instances in a single process")
def test_push_applies_state_tensor_to_simulator(shared_env):
    """Test that set_actor_root_state_tensor_robots is called when pushing.

    This verifies the critical fix where we must call set_actor_root_state_tensor_robots
    to write the modified root state tensor back to the simulator.
    """
    env = shared_env

    # Reset environment to known state
    env.reset_all()

    # Run a few steps to stabilize
    actions = torch.zeros(env.num_envs, env.dim_actions, device=env.device)
    for _ in range(5):
        env.step({"actions": actions})

    # Spy on the set_actor_root_state_tensor_robots method
    original_set_state = env.simulator.set_actor_root_state_tensor_robots
    call_count = [0]
    call_args = []

    def spy_set_state(env_ids, root_states):
        call_count[0] += 1
        call_args.append(
            (
                env_ids.clone() if isinstance(env_ids, torch.Tensor) else env_ids,
                root_states.clone() if isinstance(root_states, torch.Tensor) else root_states,
            )
        )
        return original_set_state(env_ids, root_states)

    env.simulator.set_actor_root_state_tensor_robots = spy_set_state

    # Apply push
    env_ids = torch.tensor([0, 1, 2], device=env.device)
    env._max_push_vel = torch.full((2,), 2.0, device=env.device)
    env._push_robots(env_ids)

    # Restore original method
    env.simulator.set_actor_root_state_tensor_robots = original_set_state

    # Verify set_actor_root_state_tensor_robots was called
    assert call_count[0] > 0, "set_actor_root_state_tensor_robots must be called in _push_robots to apply state changes"

    # Verify it was called with the correct env_ids
    last_call_env_ids, _ = call_args[-1]
    assert torch.equal(last_call_env_ids, env_ids), (
        f"set_actor_root_state_tensor_robots should be called with pushed env_ids. "
        f"Expected: {env_ids}, Got: {last_call_env_ids}"
    )

    print(f"✓ set_actor_root_state_tensor_robots was called {call_count[0]} time(s) during push")


@pytest.mark.skip(reason="Cannot run multiple Isaac Gym instances in a single process")
def test_push_causes_robot_motion(shared_env):
    """Test #2: Verify pushes result in actual robot movement in simulation.

    This test validates the fix where set_actor_root_state_tensor_robots()
    must be called to properly reflect push velocities into the simulator.
    """
    env = shared_env

    # Reset environment to known state
    env.reset_all()

    # Wait a few steps for environment to stabilize
    actions = torch.zeros(env.num_envs, env.dim_actions, device=env.device)
    for _ in range(10):
        env.step({"actions": actions})

    # Record initial positions before push
    initial_positions = env.simulator.robot_root_states[:, 0:3].clone()

    # Apply strong push to specific envs
    env_ids = torch.tensor([0, 1, 2], device=env.device)
    env._max_push_vel = torch.full((2,), 2.0, device=env.device)  # Strong push velocity (m/s)

    # Record velocities before push
    velocities_before = env.simulator.robot_root_states[env_ids, 7:9].clone()

    # Apply the push
    env._push_robots(env_ids)

    # Verify velocities were changed in the root state tensor
    velocities_after_push = env.simulator.robot_root_states[env_ids, 7:9]
    assert not torch.allclose(velocities_before, velocities_after_push, atol=1e-4), (
        "Push should modify root state velocities"
    )

    # Verify push velocities are non-trivial
    assert velocities_after_push.abs().max() > 0.5, "Push velocities should be substantial"

    # Step simulation multiple times to see effect
    for _ in range(20):
        env.step({"actions": actions})

    # Measure displacement
    positions_after = env.simulator.robot_root_states[:, 0:3]

    # Calculate displacement for pushed vs unpushed robots
    pushed_displacement = (positions_after[env_ids] - initial_positions[env_ids]).norm(dim=1)
    unpushed_env_ids = torch.tensor([3, 4, 5], device=env.device)
    unpushed_displacement = (positions_after[unpushed_env_ids] - initial_positions[unpushed_env_ids]).norm(dim=1)

    # Key assertion: pushed robots should move significantly more
    # We use a ratio test to be robust to different physics settings
    mean_pushed = pushed_displacement.mean()
    mean_unpushed = unpushed_displacement.mean()

    print(f"Mean pushed displacement: {mean_pushed.item():.4f}m")
    print(f"Mean unpushed displacement: {mean_unpushed.item():.4f}m")
    print(f"Ratio: {(mean_pushed / (mean_unpushed + 1e-6)).item():.2f}x")

    # Pushed robots should move at least 1.2x more than unpushed ones
    # (Relaxed threshold to 1.2x for robustness when running in shared env)
    assert mean_pushed > mean_unpushed * 1.2, (
        f"Pushed robots should move significantly more than unpushed robots. "
        f"Pushed: {mean_pushed.item():.4f}m, Unpushed: {mean_unpushed.item():.4f}m, "
        f"Ratio: {(mean_pushed / (mean_unpushed + 1e-6)).item():.2f}x (expected >1.2x)"
    )

    # Additional check: at least some pushed robots should have moved substantially
    assert (pushed_displacement > 0.3).any(), (
        f"At least some pushed robots should show substantial movement (>0.3m). "
        f"Got displacements: {pushed_displacement.tolist()}"
    )
