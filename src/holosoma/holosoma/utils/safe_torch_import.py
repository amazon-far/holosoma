from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn, optim

if TYPE_CHECKING:
    from tensordict import TensorDict
    from torch.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

# Loaded on first access: tensordict pulls in torch._dynamo, which only some algorithms need.
_LAZY_ATTRS = {
    "GradScaler": ("torch.amp", "GradScaler"),
    "TensorDict": ("tensordict", "TensorDict"),
    "TensorboardSummaryWriter": ("torch.utils.tensorboard", "SummaryWriter"),
    "autocast": ("torch.amp", "autocast"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _LAZY_ATTRS[name]
    value = getattr(importlib.import_module(module), attr)
    globals()[name] = value
    return value


__all__ = ["F", "GradScaler", "TensorDict", "TensorboardSummaryWriter", "autocast", "nn", "optim", "torch"]
