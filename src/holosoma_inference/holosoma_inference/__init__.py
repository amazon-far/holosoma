"""holosoma_inference: deployment-side policy inference.

The package entry points stay lazy so importing the top level does not drag in a transport, a
simulator or pinocchio. ``get_config`` and ``get_default_policy`` are the facade; see facade.py.
"""

from __future__ import annotations

__all__ = ["JointCommand", "get_config", "get_default_policy"]


def __getattr__(name: str):  # PEP 562 -- keeps the import cost at the first use, not the import
    if name in ("get_config", "get_default_policy"):
        from holosoma_inference import facade

        return getattr(facade, name)
    if name == "JointCommand":
        from holosoma_inference.command import JointCommand

        return JointCommand
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
