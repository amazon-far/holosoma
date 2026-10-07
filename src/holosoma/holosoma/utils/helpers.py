from __future__ import annotations

import importlib
from typing import Any, Type, cast

from holosoma.utils.safe_torch_import import torch


def get_class(path: str) -> Type[Any]:
    """Dynamically import and return a class from a module path.

    This is a replacement for hydra.utils.get_class that doesn't require Hydra.

    Parameters
    ----------
    path : str
        The fully qualified path to the class (e.g., "holosoma.envs.MyEnv")

    Returns
    -------
    Type[Any]
        The imported class

    Examples
    --------
    >>> MyClass = get_class("holosoma.envs.MyEnv")
    >>> instance = MyClass(config)
    """
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return cast("Type[Any]", getattr(module, class_name))


def instantiate(config: Any, **kwargs: Any) -> Any:
    """Instantiate an object from a config with a _target_ field.

    This is a replacement for hydra.utils.instantiate that doesn't require Hydra.
    It expects the config to have a _target_ attribute specifying the class path,
    and uses any other attributes as constructor arguments, merged with kwargs.

    Parameters
    ----------
    config : Any
        Configuration object with _target_ field and optional constructor args
    **kwargs : Any
        Additional keyword arguments to pass to the constructor (override config)

    Returns
    -------
    Any
        Instantiated object

    Examples
    --------
    >>> config = OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.001)
    >>> optimizer = instantiate(config, params=model.parameters(), lr=0.001)
    """
    if not hasattr(config, "_target_"):
        raise ValueError(f"Config must have a '_target_' attribute, got: {config}")

    target_class = get_class(config._target_)

    # Extract all config attributes except _target_
    config_dict = {}
    if hasattr(config, "__dict__"):
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    elif hasattr(config, "__dataclass_fields__"):
        # For dataclasses
        import dataclasses

        config_dict = {
            field.name: getattr(config, field.name)
            for field in dataclasses.fields(config)
            if not field.name.startswith("_")
        }

    # Merge config args with kwargs (kwargs take precedence)
    merged_kwargs = {**config_dict, **kwargs}

    return target_class(**merged_kwargs)


def class_to_dict(obj: Any) -> Any:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element: list[Any] | dict[str, Any]
        val = getattr(obj, key)
        if isinstance(val, list):
            element = []
            element.extend(class_to_dict(item) for item in val)
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def parse_observation(
    cls: Any,
    obs_key: str,
    key_list: list[str],
    buf_dict: dict[str, torch.Tensor],
    obs_scales: dict[str, float],
    noise_scales: dict[str, float],
    noise_levels: dict[str, float],
    current_noise_curriculum_value: Any = 1.0,
) -> None:
    """Parse observations for the legged_robot_base class"""
    noise_level = noise_levels[obs_key]
    # print(f"current_noise_curriculum_value: {current_noise_curriculum_value}")
    # print(f"noise_level: {noise_level}")
    for key in key_list:
        obs_noise = noise_scales[key] * current_noise_curriculum_value * noise_level
        actor_obs = getattr(cls, f"_get_obs_{key}")().clone()
        obs_scale = obs_scales[key]
        # use rand_like (uniform 0-1) instead of randn_like (N~[0,1])
        # buf_dict[key] = actor_obs * obs_scale + (torch.randn_like(actor_obs)* 2. - 1.) * obs_noise
        buf_dict[key] = (actor_obs + (torch.rand_like(actor_obs) * 2.0 - 1.0) * obs_noise) * obs_scale
