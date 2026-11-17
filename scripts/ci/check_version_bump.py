"""Fail a pull request that changes a published package without bumping its version.

A package published to PyPI is identified by its version string, so a change that ships under an
unchanged version is invisible to every consumer pinning a floor, and the next release collides
with the version already on PyPI. Run in CI on pull requests:

    python scripts/ci/check_version_bump.py --base origin/main --head HEAD

Tests, caches and documentation do not count as a shipped change. A pull request that changes
shipped files without changing behaviour opts out with the ``skip-version-bump`` label.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class Package:
    name: str
    pyproject: str
    shipped: tuple[str, ...]


PACKAGES = (
    Package(
        name="holosoma-inference",
        pyproject="src/holosoma_inference/pyproject.toml",
        # The wheel ships the Python package plus the native extension built from holosoma_cpp.
        shipped=(
            "src/holosoma_inference/holosoma_inference/",
            "src/holosoma_inference/holosoma_cpp/",
            "src/holosoma_inference/pyproject.toml",
        ),
    ),
)

_NOT_SHIPPED_DIRS = {"tests", "__pycache__"}
_NOT_SHIPPED_SUFFIXES = (".md",)
_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def is_shipped(package: Package, path: str) -> bool:
    if not path.startswith(package.shipped):
        return False
    parts = PurePosixPath(path).parts
    return not (_NOT_SHIPPED_DIRS & set(parts)) and not path.endswith(_NOT_SHIPPED_SUFFIXES)


def read_version(pyproject_text: str | None) -> str | None:
    """Return the ``[project]`` version, or None if the file is absent."""
    if pyproject_text is None:
        return None
    section = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", pyproject_text, re.MULTILINE | re.DOTALL)
    match = _VERSION_RE.search(section.group(1)) if section else None
    if match is None:
        raise ValueError("pyproject.toml has no static [project] version")
    return match.group(1)


def _numeric(version: str) -> tuple[int, ...] | None:
    if not re.fullmatch(r"\d+(\.\d+)*", version):
        return None
    return tuple(int(part) for part in version.split("."))


def check(package: Package, changed: list[str], base_version: str | None, head_version: str | None) -> str | None:
    """Return why ``package`` fails the gate, or None if it passes."""
    shipped = [path for path in changed if is_shipped(package, path)]
    if not shipped or base_version is None:
        return None
    if head_version == base_version:
        listed = "\n".join(f"  {path}" for path in shipped)
        return (
            f"{package.name}: {len(shipped)} shipped file(s) changed but the version is still {base_version} "
            f"in {package.pyproject}:\n{listed}"
        )
    base, head = _numeric(base_version), _numeric(head_version or "")
    if base is not None and head is not None and head < base:
        return f"{package.name}: version went backwards, {base_version} -> {head_version}"
    return None


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _show(commit: str, path: str) -> str | None:
    result = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="the branch the pull request merges into")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args(argv)

    merge_base = _git("merge-base", args.base, args.head).strip()
    changed = _git("diff", "--name-only", merge_base, args.head, "--").splitlines()
    failures = []
    for package in PACKAGES:
        base_version = read_version(_show(merge_base, package.pyproject))
        head_version = read_version(_show(args.head, package.pyproject))
        failure = check(package, changed, base_version, head_version)
        print(f"{package.name}: {base_version} -> {head_version}: {'FAIL' if failure else 'ok'}")
        if failure:
            failures.append(failure)
    for failure in failures:
        print(f"::error title=Version bump required::{failure.splitlines()[0]}")
        print(failure)
    if failures:
        print(
            "Bump the version in the same pull request, or add the skip-version-bump label if nothing shipped changes."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
