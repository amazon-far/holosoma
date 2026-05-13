from __future__ import annotations

import dataclasses
import json
from dataclasses import field
from pathlib import Path

import yaml
from pydantic.dataclasses import dataclass

from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig, LoggerConfig


STATE_PREDICTOR_CONFIG_NAME = "state_predictor_config.yaml"


@dataclass(frozen=True)
class TransitionDatasetConfig:
    rollout_steps: int = 4096
    cache_file: str = "state_predictor_transitions.pt"
    validation_ratio: float = 0.1
    state_key: str = "actor_obs"
    target_key: str = "actor_obs"
    force_collect: bool = False


@dataclass(frozen=True)
class TransitionModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 256])
    activation: str = "SiLU"
    use_layer_norm: bool = True
    dropout_prob: float = 0.0
    predict_residual: bool = True
    normalize_inputs: bool = True


@dataclass(frozen=True)
class TransitionTrainConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 0
    save_interval: int = 5


@dataclass(frozen=True)
class StatePredictorConfig:
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            project="StatePrediction",
            name="single_action",
            num_envs=1,
            seed=42,
            headless=True,
        )
    )
    logger: LoggerConfig = field(default_factory=DisabledLoggerConfig)
    device: str | None = None
    data: TransitionDatasetConfig = field(default_factory=TransitionDatasetConfig)
    model: TransitionModelConfig = field(default_factory=TransitionModelConfig)
    optim: TransitionTrainConfig = field(default_factory=TransitionTrainConfig)

    def save_config(self, path: str) -> None:
        with open(path, "w") as file:
            yaml.safe_dump(self.to_serializable_dict(), file)

    def to_serializable_dict(self) -> dict:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def get_cache_path(self, experiment_dir: Path) -> Path:
        cache_path = Path(self.data.cache_file)
        if cache_path.is_absolute():
            return cache_path
        return experiment_dir / cache_path