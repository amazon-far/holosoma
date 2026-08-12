#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`compare_xsens_g1_poses`.

Use ``compare_xsens_g1_poses.py`` for the generalized T-pose/N-pose viewer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tyro

src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.examples.xsens_tennis.compare_xsens_g1_poses import *  # noqa: E402,F403
from holosoma_retargeting.examples.xsens_tennis.compare_xsens_g1_poses import (  # noqa: E402
    XsensG1PoseComparisonConfig,
    main,
)

if __name__ == "__main__":
    main(tyro.cli(XsensG1PoseComparisonConfig))
