"""Configuration values for viser visualization."""

from __future__ import annotations

from holosoma_retargeting.config_types.viser import ViserConfig, XsensViserConfig


def get_default_viser_config() -> ViserConfig:
    """Get default viser visualization configuration.

    Returns:
        ViserConfig: Default configuration instance.
    """
    return ViserConfig()


def get_default_xsens_viser_config() -> XsensViserConfig:
    """Get the default Xsens-capable Viser player configuration."""

    return XsensViserConfig()


__all__ = ["get_default_viser_config", "get_default_xsens_viser_config"]
