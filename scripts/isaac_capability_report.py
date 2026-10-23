#!/usr/bin/env python3
"""Generate the machine-checkable half of ``docs/isaac-sim-backends.md``.

Run inside the Isaac Sim 6.1 environment, from a directory with **no ``isaaclab`` sibling**:

    # From the holosoma repository root:
    source scripts/source_isaacsim6_setup.sh
    python scripts/isaac_capability_report.py --write

Why generated rather than hand-written
--------------------------------------
Three parts of that document go stale the moment anything is re-pinned: the Isaac Sim / Isaac Lab
versions, which physics and visualizer backends the install actually provides, and which holosoma
capability works under which backend. The last one is the dangerous one -- a hand-maintained table
drifts away from what the code enforces, so it ends up either promising a capability that raises
or omitting one that silently works.

So the table is read from :mod:`holosoma.simulator.isaacsim6.capabilities`, the same data
``validate()`` enforces at startup, and the versions are read from the live environment. The
prose about *how* 5.1 and 6.1 disagree stays hand-written below the generated block, because it
needs judgement rather than introspection.

The generated block is delimited by markers so regenerating never touches the prose.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import importlib.metadata
import pathlib
import sys
from typing import Any

BEGIN = "<!-- BEGIN GENERATED: scripts/isaac_capability_report.py -->"
END = "<!-- END GENERATED -->"

_DISTRIBUTIONS = ("isaacsim", "isaaclab", "isaaclab-physx", "isaaclab-newton", "isaaclab-ov")
_PHYSICS_PACKAGES = {
    "physx": ("isaaclab_physx.physics", "PhysxCfg"),
    "newton": ("isaaclab_newton.physics", "NewtonCfg"),
    "ovphysx": ("isaaclab_ov.physics", "OvPhysxCfg"),
}


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _available(module: str, attribute: str) -> bool:
    """Whether ``module.attribute`` resolves in this environment.

    Import errors are treated as "unavailable" rather than propagated: a physics backend whose
    package is absent is exactly what this is reporting, and some of these packages cannot be
    imported at all until the Kit runtime is up.
    """
    try:
        return hasattr(importlib.import_module(module), attribute)
    except Exception:
        return False


_MARKS = {
    "verified": "verified",
    "unverified": "*unverified*",
    "blocked": "blocked",
    "unavailable": "**no**",
    "unported": "**unported**",
}


def _renderers(capability: Any) -> str:
    """The rendering-backend verification cell.

    Isaac Lab 3.0 makes the physics engine and the renderer *independent* choices, so a per-
    capability "verified" that names only the physics backend answers half the question. This is
    the other half: which `isaac_visualizers` selections the capability was actually measured
    under.

    An empty tuple is rendered as "n/a" rather than "none", because the two mean opposite things.
    "n/a" says the renderer is not a variable for this capability (no render path involved, so
    headless is the whole story); "none" would read as "verified under no renderer", i.e. not
    verified.
    """
    from holosoma.simulator.isaacsim6 import capabilities

    if not capability.ported:
        return "—"
    if capability.renderers == capabilities.RENDERER_AGNOSTIC:
        return "n/a (headless)"
    return ", ".join(f"`{name}`" for name in capability.renderers)


def _rows() -> list[str]:
    """The capability table: availability, then verification per physics *and* rendering backend.

    Three separate questions, so three separate column groups. ``Available`` answers "does the 6.1
    lane implement this at all", which is backend-independent. The physics columns answer "what
    was observed when it ran on that engine" -- a cell reading ``blocked`` means the backend never
    finished constructing, so the capability's code never executed and nothing is known about it.
    Rendering those as ``no`` would claim fourteen independent per-capability findings where there
    is really one backend failure. ``Renderers`` answers the same question for the *other*
    independent axis, which the physics columns say nothing about.
    """
    from holosoma.simulator.isaacsim6 import capabilities

    backends = [status.backend for status in capabilities.BACKEND_STATUS]
    header = " | ".join(f"`{backend.value}`" for backend in backends)
    lines = [
        f"| Capability | Available on 6.1 | {header} | Renderers verified | Evidence |",
        "|---|---|" + "---|" * len(backends) + "---|---|",
    ]
    for capability in capabilities.CAPABILITIES:
        available = "yes" if capability.ported else "**no**"
        cells = " | ".join(_MARKS[capability.on(backend).value] for backend in backends)
        lines.append(f"| {capability.name} | {available} | {cells} | {_renderers(capability)} | {capability.note} |")
    return lines


def render() -> str:
    from holosoma.simulator.isaacsim6 import capabilities

    out: list[str] = [BEGIN, ""]

    out.append("### Pinned versions")
    out.append("")
    out.append("| Package | Version |")
    out.append("|---|---|")
    out.extend(f"| `{dist}` | {_version(dist)} |" for dist in _DISTRIBUTIONS)
    out.append(f"| Python | {sys.version.split()[0]} |")
    out.append("")

    out.append("### Physics backends: measured state")
    out.append("")
    out.append(
        "Three different things, deliberately separated — a backend can be installed, and still "
        "not construct, and construction is still not the same as working:"
    )
    out.append("")
    out.append("- **Package** — the config class imports in this environment.")
    out.append(
        "- **Constructs** — a simulator can actually be built on it. This is the column that "
        "decides whether per-capability results below mean anything: when it is `no`, no "
        "capability on that backend ever got to run."
    )
    out.append("- **Usable** — holosoma supports it. `no` means `validate()` rejects it at startup.")
    out.append("")
    out.append("| `isaac_physics_backend` | Package | Constructs | Usable | Measured state |")
    out.append("|---|---|---|---|---|")
    for status in capabilities.BACKEND_STATUS:
        name = status.backend.value
        module, attribute = _PHYSICS_PACKAGES[name]
        present = "yes" if _available(module, attribute) else "no"
        constructs = "yes" if status.reachable else "**no**"
        usable = "yes" if status.usable else "**no**"
        out.append(f"| `{name}` | {present} | {constructs} | {usable} | {status.summary} |")
    out.append("")
    for status in capabilities.BACKEND_STATUS:
        if not status.usable:
            out.append(f"**`{status.backend.value}`** — {status.detail}")
            out.append("")

    out.append("### Rendering backends")
    out.append("")
    out.append(
        "`isaac_visualizers` accepts these names, matching Isaac Lab 3.0's `--visualizer`: "
        + ", ".join(f"`{v}`" for v in capabilities.KNOWN_VISUALIZERS)
        + ". Empty means headless."
    )
    out.append("")
    out.append(
        "| Visualizer | State | Measured |\n"
        "|---|---|---|\n"
        "| *(empty)* | verified | Headless. Cameras and depth still pass — a `TiledCamera` is an "
        "offscreen RTX sensor, and Isaac Lab's RTX renderer enables what it needs on demand. |\n"
        "| `kit` | verified | Required for a GUI viewport. |\n"
        "| `newton` | verified | `camera_assert` passes with non-blank frames. |\n"
        "| `rerun` | **not installed** | Raises `ModuleNotFoundError: No module named 'rerun'` "
        "at startup. The name is valid; the distribution is absent here. |\n"
        "| `viser` | **not installed** | Same — `No module named 'viser'`. |"
    )
    out.append("")
    out.append(
        "Isaac Lab resolves a visualizer by name "
        "(`isaaclab_visualizers.rerun.rerun_visualizer:RerunVisualizer`) and raises when the "
        "backing distribution is missing, so an uninstalled backend fails loudly rather than "
        "quietly rendering nothing. Installing the extra in `setup_isaacsim6.sh` closes that gap."
    )
    out.append("")
    out.append(
        "A correction worth recording. An earlier version of this table claimed all five "
        "selections verified on both engines. They had been run — but `isaac_visualizers` was "
        "validated here and then **never read by anything**: `launcher.apply_launcher_args` "
        "ignored it and resolved every run to `kit` or `None`, so ten runs exercised one "
        "configuration ten times, and the two uninstalled backends looked fine precisely because "
        "the selection never reached the launcher. Honouring the field is what surfaced the real "
        "state — and `build_run_sim_config(visualizers=...)` is what made the axis reachable from "
        "a harness at all."
    )
    out.append("")

    out.append("### Capability matrix")
    out.append("")
    out.append(
        "Read from `holosoma/simulator/isaacsim6/capabilities.py`, the same table `validate()` "
        "enforces when the simulator is built, so the doc cannot promise what the code rejects."
    )
    out.append("")
    out.append(
        "The columns answer three different questions. `Available on 6.1` is whether the lane "
        "implements the capability at all. The physics columns and `Renderers verified` are the "
        "two **independent** verification axes Isaac Lab 3.0 introduced — an engine choice and a "
        "renderer choice — so a capability verified on `physx` is not thereby verified under "
        "every visualizer, and vice versa."
    )
    out.append("")
    out.append("Physics-column values:")
    out.append("")
    out.append("- `verified` — a harness ran on that backend and passed. The harness is named under Evidence.")
    out.append(
        "- `unverified` — implemented, no harness covers it. Not a claim that it works, and not "
        "a claim that it does not."
    )
    out.append(
        "- `blocked` — the backend never finished constructing, so this capability's code never "
        "executed. **Nothing is known about the capability here**; this records a backend "
        "failure, not a capability failure. Fix the backend (see its row above) and re-measure."
    )
    out.append("- `unported` — holosoma has no implementation on the 6.1 lane, so no backend has it.")
    out.append("")
    out.append(
        "Anything not `verified` or `unverified` raises at startup naming itself, rather than degrading quietly."
    )
    out.append("")
    out.append("`Renderers verified` values:")
    out.append("")
    out.append(
        f"- A list of visualizer names — the `isaac_visualizers` selections actually measured. "
        f"`{capabilities.HEADLESS}` is the empty selection. Anything absent from the list is "
        f"**unmeasured**, not known-good."
    )
    out.append(
        "- `n/a (headless)` — the renderer is not a variable for this capability: it involves no "
        "render path, so the headless run is the whole story. A claim, not a shrug — if a "
        "capability turns out to behave differently under some visualizer, this is the cell that "
        "was wrong."
    )
    out.append("- `—` — unported, so there is nothing to verify under any renderer.")
    out.append("")
    out.extend(_rows())
    out.append("")
    out.append(_tally())
    out.append("")
    out.append(END)
    return "\n".join(out)


def _tally() -> str:
    """The per-state totals, counted rather than written down.

    A hand-maintained "N of M pass" sentence is the first thing to go stale when a capability is
    added, and it goes stale silently because nothing cross-checks it.
    """
    from holosoma.simulator.isaacsim6 import capabilities

    counts = collections.Counter(capability.physx.value for capability in capabilities.CAPABILITIES)
    parts = ", ".join(f"{counts[state]} {state}" for state in _MARKS if counts[state])
    return f"Totals on `physx`, out of {len(capabilities.CAPABILITIES)} capabilities: {parts}."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the generated block in docs/isaac-sim-backends.md instead of printing it",
    )
    args = parser.parse_args()

    block = render()
    if not args.write:
        print(block)
        return 0

    doc = pathlib.Path(__file__).resolve().parent.parent / "docs" / "isaac-sim-backends.md"
    text = doc.read_text()
    if BEGIN not in text or END not in text:
        print(f"{doc} is missing the generated-block markers; not writing.", file=sys.stderr)
        return 1
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    doc.write_text(head + block + tail)
    print(f"Updated the generated block in {doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
