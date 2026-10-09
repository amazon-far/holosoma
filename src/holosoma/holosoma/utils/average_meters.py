from __future__ import annotations

from typing import Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn


class AverageMeter(nn.Module):
    mean: torch.Tensor

    def __init__(self, in_shape: int | Sequence[int], max_size: int) -> None:
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values: torch.Tensor) -> None:
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self) -> None:
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self) -> int:
        return self.current_size

    def get_mean(self) -> npt.NDArray[np.float32]:
        return cast("npt.NDArray[np.float32]", self.mean.squeeze(0).cpu().numpy())


class TensorAverageMeter:
    def __init__(self) -> None:
        self.tensors: list[torch.Tensor] = []

    def add(self, x: torch.Tensor) -> None:
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
        self.tensors.append(x)

    def mean(self) -> torch.Tensor | int:
        if len(self.tensors) == 0:
            return 0
        cat = torch.cat(self.tensors, dim=0)
        if cat.numel() == 0:
            return 0
        return cat.mean()

    def clear(self) -> None:
        self.tensors = []

    def mean_and_clear(self) -> torch.Tensor | int:
        mean = self.mean()
        self.clear()
        return mean


class TensorAverageMeterDict:
    def __init__(self) -> None:
        self.data: dict[str, TensorAverageMeter] = {}

    def add(self, data_dict: Mapping[str, torch.Tensor]) -> None:
        for k, v in data_dict.items():
            # Originally used a defaultdict, this had lambda
            # pickling issues with DDP.
            if k not in self.data:
                self.data[k] = TensorAverageMeter()
            self.data[k].add(v)

    def mean(self) -> dict[str, torch.Tensor | int]:
        return {k: v.mean() for k, v in self.data.items()}

    def clear(self) -> None:
        self.data = {}

    def mean_and_clear(self) -> dict[str, torch.Tensor | int]:
        mean = self.mean()
        self.clear()
        return mean
