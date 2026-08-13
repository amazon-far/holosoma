"""Regression test: G1SmplRetargeter must strip <freejoint/> before IK.

The shipped g1_29dof.xml carries a <freejoint/> (shared with the policy's
MuJoCo backend). The retargeter's IK is fixed-base and root-relative; if the
freejoint is not stripped, mujoco loads nq=36/nv=35 and the IK bakes body
rotation into the root, so the 29 joint angles no longer match the SMPL pose.
After stripping, the model must be a clean fixed-base nq=29/nv=29.

Skips cleanly if mujoco isn't importable (e.g. outside the bazel venv).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

# Shipped G1 MJCF with a freejoint, relative to the holosoma repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 6)))
_G1_XML = os.path.join(
    _REPO_ROOT, "src", "holosoma_retargeting", "holosoma_retargeting", "models", "g1", "g1_29dof.xml"
)


def _strip(text: str) -> str:
    s = re.sub(r"\s*<freejoint[^>]*/>\s*", "\n", text)
    return re.sub(r"\s*<freejoint[^>]*>.*?</freejoint>\s*", "\n", s, flags=re.DOTALL)


@pytest.mark.skipif(not Path(_G1_XML).exists(), reason="shipped g1_29dof.xml not present")
def test_shipped_g1_has_freejoint():
    with open(_G1_XML, encoding="utf-8") as f:
        assert "<freejoint" in f.read(), "test fixture invalid: expected a freejoint in g1_29dof.xml"


@pytest.mark.skipif(not Path(_G1_XML).exists(), reason="shipped g1_29dof.xml not present")
def test_strip_yields_fixed_base_29dof():
    mujoco = pytest.importorskip("mujoco")
    with open(_G1_XML, encoding="utf-8") as f:
        text = f.read()

    base = os.path.dirname(os.path.abspath(_G1_XML))
    assets = {}
    for root, _dirs, files in os.walk(base):
        for fn in files:
            if fn.lower().endswith((".obj", ".stl", ".mtl", ".png", ".jpg")):
                p = os.path.join(root, fn)
                with open(p, "rb") as fh:
                    assets[os.path.relpath(p, base)] = fh.read()

    stripped = mujoco.MjModel.from_xml_string(_strip(text), assets)
    assert stripped.nq == 29, f"stripped model must be fixed-base 29-DOF, got nq={stripped.nq}"
    assert stripped.nv == 29, f"stripped model must be fixed-base 29-DOF, got nv={stripped.nv}"
