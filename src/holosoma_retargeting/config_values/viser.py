"""Configuration values for viser visualization."""

from __future__ import annotations

from holosoma_retargeting.config_types.viser import ViserConfig


def get_default_viser_config() -> ViserConfig:
    """Get default viser visualization configuration.

    Returns:
        ViserConfig: Default configuration instance.
    """
    return ViserConfig()


__all__ = ["get_default_viser_config"]
