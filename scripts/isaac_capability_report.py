#!/usr/bin/env python3
"""Report installed versions and recorded GA coverage without starting a simulator.

Run in hssim6: python scripts/isaac_capability_report.py --write
"""

from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import sys

BEGIN = "<!-- BEGIN GENERATED: scripts/isaac_capability_report.py -->"
END = "<!-- END GENERATED -->"
_DISTRIBUTIONS = ("isaacsim", "isaaclab", "isaaclab-physx", "isaaclab-newton", "isaaclab-ov")


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def render() -> str:
    from holosoma.simulator.isaacsim6.capabilities import BACKEND_STATUS, CAPABILITIES, KNOWN_VISUALIZERS

    out = [BEGIN, "", "### Environment used for validation", "", "| Package | Version |", "|---|---|"]
    out.extend(f"| `{dist}` | {_version(dist)} |" for dist in _DISTRIBUTIONS)
    out.extend(
        [
            f"| Python | {sys.version.split()[0]} |",
            "",
            "The Lab release is **v3.0.0**; its Python packages use a separate version scheme.",
            "",
            "### Physics backends",
            "",
            "| Backend | Supported | Runtime evidence |",
            "|---|---|---|",
        ]
    )
    out.extend(
        f"| `{status.backend.value}` | {'yes' if status.usable else 'no'} | {status.summary} |"
        for status in BACKEND_STATUS
    )
    out.extend(
        [
            "",
            "### Existing Holosoma capabilities",
            "",
            "`verified` means the named harness completed on this GA environment. "
            "`unverified` means runtime coverage is incomplete; `blocked` means the backend is unsupported. "
            "All recorded runs were headless. Desktop visualizers have no runtime verification here.",
            "",
            "| Capability | PhysX | Newton | OvPhysX | Evidence |",
            "|---|---|---|---|---|",
        ]
    )
    for cap in CAPABILITIES:
        cells = " | ".join(cap.on(status.backend).value for status in BACKEND_STATUS)
        out.append(f"| {cap.name} | {cells} | {cap.note} |")
    out.extend(
        [
            "",
            "Accepted visualizer names: " + ", ".join(f"`{name}`" for name in KNOWN_VISUALIZERS) + ". "
            "These names describe upstream choices, not verified desktop support. Install the corresponding "
            "Isaac Lab extras before selecting optional viewers.",
            "",
            END,
        ]
    )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="update docs/isaac-sim-backends.md")
    args = parser.parse_args()
    block = render()
    if not args.write:
        print(block)
        return 0
    doc = pathlib.Path(__file__).resolve().parent.parent / "docs" / "isaac-sim-backends.md"
    text = doc.read_text()
    if BEGIN not in text or END not in text:
        print(f"{doc} is missing generated-block markers", file=sys.stderr)
        return 1
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    doc.write_text(head + block + tail)
    print(f"Updated {doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
