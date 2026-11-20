"""Configuration type for the MDP a policy is trained and evaluated on."""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import holosoma.config_values.action
import holosoma.config_values.command
import holosoma.config_values.curriculum
import holosoma.config_values.observation
import holosoma.config_values.reward
from holosoma.config_types.action import ActionManagerCfg
from holosoma.config_types.command import CommandManagerCfg
from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.config_types.reward import RewardManagerCfg
from holosoma.utils.config_registry import UseRegistry


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MDPConfig:
    """The managers an ``PolicyTask`` builds: what the policy observes, does, is rewarded for and is commanded."""

    observation: Annotated[
        ObservationManagerCfg | None, UseRegistry(holosoma.config_values.observation.OBSERVATION_REGISTRY)
    ] = holosoma.config_values.observation.none
    action: Annotated[ActionManagerCfg | None, UseRegistry(holosoma.config_values.action.ACTION_REGISTRY)] = (
        holosoma.config_values.action.none
    )
    reward: Annotated[RewardManagerCfg | None, UseRegistry(holosoma.config_values.reward.REWARD_REGISTRY)] = (
        holosoma.config_values.reward.none
    )
    command: Annotated[CommandManagerCfg | None, UseRegistry(holosoma.config_values.command.COMMAND_REGISTRY)] = (
        holosoma.config_values.command.none
    )
    curriculum: Annotated[
        CurriculumManagerCfg | None, UseRegistry(holosoma.config_values.curriculum.CURRICULUM_REGISTRY)
    ] = holosoma.config_values.curriculum.none
