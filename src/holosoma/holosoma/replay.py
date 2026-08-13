from __future__ import annotations

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.eval_utils import (
    init_sim_imports,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app


def replay(tyro_config: ExperimentConfig):
    simulation_app = init_sim_imports(tyro_config)

    import torch

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    done = False
    while not done:
        env.simulator.sim.step()
        done = env.step_visualize_motion(None)  # type: ignore[attr-defined]

    close_simulation_app(simulation_app)


def main() -> None:
    from holosoma.config_values.experiment import get_annotated_experiment_config
    from holosoma.utils.config_registry import parse_config

    # Pass the factory uncalled so parse_config builds it after plugins load.
    tyro_cfg = parse_config(get_annotated_experiment_config)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
