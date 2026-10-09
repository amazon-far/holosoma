from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest
import tyro

from holosoma.config_types.distribution import DistributionSpec
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import MujocoMaterialDR
from holosoma.utils.tyro_utils import TYRO_CONIFG

pytestmark = pytest.mark.no_sim


def test_experiment_config() -> None:
    assert isinstance(tyro.cli(ExperimentConfig, args=(), config=TYRO_CONIFG), ExperimentConfig)


def test_distribution_like_field_is_a_cli_flag() -> None:
    # A DistributionLike range is ONE token holding a Python literal. Without the CLI constructor
    # on DistributionLike, tyro cannot build a parser for the union's Dict[str, Any] member and
    # errors as soon as a flag targets such a field ("struct-type values requires a default value").
    cfg = tyro.cli(MujocoMaterialDR, args=("--sliding-friction", "[0.3, 0.9]"), config=TYRO_CONIFG)
    assert cfg.sliding_friction == [0.3, 0.9]

    cfg = tyro.cli(
        MujocoMaterialDR,
        args=("--sliding-friction", "{'kind': 'gaussian', 'low': 0.1, 'high': 0.9}"),
        config=TYRO_CONIFG,
    )
    # pydantic validates the dict form into the union's DistributionSpec member.
    assert cfg.sliding_friction == DistributionSpec(kind="gaussian", low=0.1, high=0.9)

    cfg = tyro.cli(MujocoMaterialDR, args=("--solref", "0", "[0.02, 0.05]"), config=TYRO_CONIFG)
    assert cfg.solref == {0: [0.02, 0.05]}


def test_distribution_like_invalid_range_rejected_at_cli() -> None:
    with pytest.raises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        tyro.cli(MujocoMaterialDR, args=("--sliding-friction", "[0.9, 0.3]"), config=TYRO_CONIFG)


def test_randomization_configs_build_eager_parser() -> None:
    # Schema exporters and shell completion build the full parser tree up front, forcing every
    # union branch that lazy parsing skips.
    with pytest.warns(DeprecationWarning, match="get_parser"):
        tyro.extras.get_parser(MujocoMaterialDR)
