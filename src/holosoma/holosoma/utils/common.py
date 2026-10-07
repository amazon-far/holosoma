# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime
from typing import NoReturn

import numpy as np

from holosoma.utils.safe_torch_import import torch


# if there's overlap between args_list and commandline input, use commandline input
def solve_argv_conflict(args_list: list[str]) -> None:
    arguments_to_be_removed = []
    arguments_size = []

    for argv in sys.argv[1:]:
        if argv.startswith("-"):
            size_count = 1
            for i, args in enumerate(args_list):
                if args == argv:
                    arguments_to_be_removed.append(args)
                    for more_args in args_list[i + 1 :]:
                        if not more_args.startswith("-"):
                            size_count += 1
                        else:
                            break
                    arguments_size.append(size_count)
                    break

    for args, size in zip(arguments_to_be_removed, arguments_size):
        args_index = args_list.index(args)
        for _ in range(size):
            args_list.pop(args_index)


def print_error(*message: object) -> NoReturn:
    print("\033[91m", "ERROR ", *message, "\033[0m")
    raise RuntimeError


def print_ok(*message: object) -> None:
    print("\033[92m", *message, "\033[0m")


def print_warning(*message: object) -> None:
    print("\033[93m", *message, "\033[0m")


def print_info(*message: object) -> None:
    print("\033[96m", *message, "\033[0m")


def get_time_stamp() -> str:
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    second = now.strftime("%S")
    return f"{month}-{day}-{year}-{hour}-{minute}-{second}"


def seeding(seed: int = 0, torch_deterministic: bool = False) -> int:
    logger = logging.getLogger()
    logger.info("Setting seed: %d", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def distance_l2(root_pos: torch.Tensor, wp_pos: torch.Tensor) -> torch.Tensor:
    return torch.norm(wp_pos - root_pos, dim=0)


def value_to_color(value: float, min_value: float, max_value: float) -> tuple[float, int, float]:
    """
    Converts a numerical value to an RGB color.
    The color will range from blue (low values) to red (high values).
    """
    # Ensure value is within the range [0, max_value]
    value = max(min_value, min(value, max_value))

    # Calculate the proportion of the value
    red = (value - min_value) / (max_value - min_value)

    # Map the proportion to the red channel for a red gradient
    # Blue for minimum value and red for maximum value
    blue = 1 - red
    green = 0  # Keep green constant for simplicity

    # Return the RGB color
    return red, green, blue


def normalize(
    x: float,
    min_value: float,
    max_value: float,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> float:
    """
    Normalize a value from a given range to a target range.
    """
    # Normalize the value to the range [0, 1]
    normalized = (x - min_value) / (max_value - min_value)

    # Scale the value to the target range
    return target_min + normalized * (target_max - target_min)


def unnormalize(
    x: float,
    min_value: float,
    max_value: float,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> float:
    """
    Unnormalize a value from a target range to a given range.
    """
    # Normalize the value to the range [0, 1]
    normalized = (x - target_min) / (target_max - target_min)

    # Scale the value to the target range
    return min_value + normalized * (max_value - min_value)
