from __future__ import annotations

import pytest
import torch

from holosoma.agents.distillation.rollout_storage import RolloutStorageDagger
from holosoma.agents.distillation_ppo.rollout_storage import RolloutStorageDistillation
from holosoma.agents.modules.data_utils import RolloutStorage

pytestmark = pytest.mark.no_sim


def _filled_storage(storage: RolloutStorage) -> RolloutStorage:
    storage.register("ids", shape=(1,), dtype=torch.long)
    ids = torch.arange(storage.num_transitions_per_env * storage.num_envs).view(
        storage.num_transitions_per_env, storage.num_envs, 1
    )
    storage["ids"] = ids
    return storage


def _epoch_orders(storage: RolloutStorage, num_mini_batches: int, num_epochs: int, **kwargs: bool) -> list[list[int]]:
    """Sample ids in the order the generator yields them, one list per epoch."""
    batches = storage.mini_batch_generator(num_mini_batches, num_epochs, **kwargs)
    ids = [b["ids"].flatten() for b in batches]
    return [torch.cat(ids[e * num_mini_batches : (e + 1) * num_mini_batches]).tolist() for e in range(num_epochs)]


def test_default_generator_reuses_one_permutation_across_epochs() -> None:
    torch.manual_seed(0)
    storage = _filled_storage(RolloutStorage(num_envs=4, num_transitions_per_env=8))

    orders = _epoch_orders(storage, num_mini_batches=4, num_epochs=3)

    assert orders[0] == orders[1] == orders[2]
    assert sorted(orders[0]) == list(range(32))


def test_reshuffle_each_epoch_draws_a_new_permutation_per_epoch() -> None:
    torch.manual_seed(0)
    storage = _filled_storage(RolloutStorage(num_envs=4, num_transitions_per_env=8))

    orders = _epoch_orders(storage, num_mini_batches=4, num_epochs=3, reshuffle_each_epoch=True)

    assert orders[0] != orders[1] and orders[1] != orders[2]
    assert all(sorted(order) == list(range(32)) for order in orders)


@pytest.mark.parametrize(
    "storage",
    [
        RolloutStorageDagger(
            num_envs=4, num_transitions_per_env=8, actor_obs_dim=1, teacher_obs_dim=1, depth_shape=(1,), num_actions=1
        ),
        RolloutStorageDistillation(
            num_envs=4,
            num_transitions_per_env=8,
            actor_obs_dim=1,
            teacher_obs_dim=1,
            critic_obs_dim=1,
            depth_shape=(1,),
            num_actions=1,
        ),
    ],
)
def test_distillation_storages_reshuffle_each_epoch(storage: RolloutStorage) -> None:
    torch.manual_seed(0)
    _filled_storage(storage)

    orders = _epoch_orders(storage, num_mini_batches=4, num_epochs=3)

    assert orders[0] != orders[1] and orders[1] != orders[2]
