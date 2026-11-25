# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Solved Classic MuJoCo contact pairs shared by robot and filtered force readers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from holosoma.utils.safe_torch_import import torch


@dataclass
class ContactPairForces:
    """One native snapshot, including the existing total per-body force reduction."""

    geoms: npt.NDArray[np.int32]
    normal: npt.NDArray[np.float32]
    total: npt.NDArray[np.float32]
    body_forces: torch.Tensor

    def matrix(
        self,
        source_masks: npt.NDArray[np.int32],
        counterpart_masks: npt.NDArray[np.int32],
        *,
        force_mode: str,
    ) -> torch.Tensor:
        matrix = np.zeros((source_masks.shape[0], counterpart_masks.shape[0], 3), dtype=np.float32)
        first, second = self.geoms.T
        forces = self.normal if force_mode == "normal" else self.total
        matrix += np.einsum("sc,tc,ca->sta", source_masks[:, second], counterpart_masks[:, first], forces)
        matrix -= np.einsum("sc,tc,ca->sta", source_masks[:, first], counterpart_masks[:, second], forces)
        return torch.from_numpy(matrix).unsqueeze(0)
