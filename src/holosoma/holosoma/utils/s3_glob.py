"""S3 object listing with Python ``glob``-compatible key matching."""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any

import boto3  # type: ignore[import-untyped]

_GLOB_MAGIC = "*?["


def _get_s3_client() -> Any:
    """Construct the boto3 client used for object listing."""
    return boto3.client("s3")


def _literal_prefix(pattern: str) -> str:
    """Return the key portion before the first glob metacharacter."""
    first_magic = min((pattern.find(char) for char in _GLOB_MAGIC if char in pattern), default=len(pattern))
    return pattern[:first_magic]


def _match_count(key: str, pattern: str) -> int:
    """Count the ways ``key`` matches a recursive POSIX glob pattern.

    Counting, rather than returning a boolean, preserves the duplicate results
    produced by Python's ``glob`` when a pattern contains multiple ``**``
    components.
    """
    key_parts = key.split("/")
    counts = {0: 1}

    for pattern_part in pattern.split("/"):
        next_counts: dict[int, int] = {}
        for key_index, count in counts.items():
            if pattern_part == "**":
                for consumed in range(key_index, len(key_parts) + 1):
                    if any(part.startswith(".") for part in key_parts[key_index:consumed]):
                        break
                    next_counts[consumed] = next_counts.get(consumed, 0) + count
            elif key_index < len(key_parts):
                key_part = key_parts[key_index]
                if (not key_part.startswith(".") or pattern_part.startswith(".")) and fnmatchcase(
                    key_part, pattern_part
                ):
                    next_counts[key_index + 1] = next_counts.get(key_index + 1, 0) + count
        counts = next_counts

    return counts.get(len(key_parts), 0)


def list_s3_uris(pattern: str) -> list[str]:
    """List S3 objects matching ``pattern`` in stdlib glob order semantics.

    Results are sorted lexicographically. A URI is repeated when Python's local
    ``glob.glob(..., recursive=True)`` would report the corresponding path more
    than once because of multiple ``**`` components.
    """
    if not pattern.startswith("s3://"):
        raise ValueError(f"Invalid S3 pattern: {pattern!r}.")
    bucket, separator, key_pattern = pattern[len("s3://") :].partition("/")
    if not bucket or not separator:
        raise ValueError(f"Invalid S3 pattern: {pattern!r}.")
    if any(char in bucket for char in _GLOB_MAGIC):
        raise ValueError("S3 bucket names cannot contain glob patterns.")
    if not key_pattern:
        raise ValueError("An S3 glob must include an object-key pattern.")

    pages = (
        _get_s3_client()
        .get_paginator("list_objects_v2")
        .paginate(
            Bucket=bucket,
            Prefix=_literal_prefix(key_pattern),
        )
    )

    matches: list[str] = []
    for page in pages:
        for obj in page.get("Contents", ()):
            key = obj["Key"]
            matches.extend([f"s3://{bucket}/{key}"] * _match_count(key, key_pattern))
    return sorted(matches)
