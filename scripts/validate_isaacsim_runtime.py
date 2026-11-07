"""Fail CI before runtime problems can be reported as skipped simulator tests."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys

from holosoma.simulator.isaacsim.backends import get_backend


def validate_runtime(backend_name: str, *, multigpu: bool = False) -> dict[str, str | int]:
    # get_backend permits an absent optional runtime for CPU utilities. CI needs
    # an actual installation and a matching native API family.
    lab_version = importlib.metadata.version("isaaclab")
    backend = get_backend(backend_name)
    torch_version = importlib.metadata.version("torch")
    warp_version = importlib.metadata.version("warp-lang")
    expected_torch, expected_warp = ("2.11.0", "1.17.0") if backend.is_isaaclab3 else ("2.7.0", "1.16.0")
    if torch_version.split("+", 1)[0] != expected_torch:
        raise RuntimeError(f"{backend_name} requires torch {expected_torch}; found {torch_version}.")
    if warp_version != expected_warp:
        raise RuntimeError(f"{backend_name} requires warp-lang {expected_warp}; found {warp_version}.")

    if backend.is_isaaclab3:
        try:
            importer_version = importlib.metadata.version("isaacsim-asset-isolated")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("Isaac Lab importer extra is missing; rerun scripts/setup_isaacsim3.sh.") from exc
        if importer_version != "6.2.0.0":
            raise RuntimeError(f"Isaac Lab requires standalone importers 6.2.0.0; found {importer_version}.")
        for package in ("moviepy", "imageio_ffmpeg", "ovrtx", "ovphysx", "ovstage"):
            if importlib.util.find_spec(package) is None:
                raise RuntimeError(
                    f"Isaac Lab video/OV extras are missing {package}; install the complete Lab 3 profile."
                )

    import torch

    required_devices = 2 if multigpu else 1
    if not torch.cuda.is_available():
        raise RuntimeError("Isaac Sim CI requires a usable CUDA device; simulator tests must not all skip.")
    device_count = torch.cuda.device_count()
    if device_count < required_devices:
        raise RuntimeError(f"This CI suite requires {required_devices} CUDA devices; found {device_count}.")
    return {
        "backend": backend_name,
        "isaaclab_package": lab_version,
        "torch": torch_version,
        "warp": warp_version,
        "cuda_devices": device_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["isaaclab2", "isaaclab3"], required=True)
    parser.add_argument("--multigpu", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(validate_runtime(args.backend, multigpu=args.multigpu), sort_keys=True))
    except (ImportError, RuntimeError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"Isaac Sim runtime preflight failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
