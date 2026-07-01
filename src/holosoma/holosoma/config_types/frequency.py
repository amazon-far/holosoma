"""Resolve rate fields written as frequency strings ("20Hz") into decimations.

A decimation is "act every Nth tick of a base clock" running at ``base_hz``. A rate field is either a
decimation int (every Nth tick) or a target frequency string resolved against a base rate.

Grammar (case-insensitive, surrounding whitespace ignored)::

    "20Hz"     require an exact decimation (base/20 must be a whole number); error otherwise
    ">20Hz"    round to achieve a rate >= target (floor(base/20))
    "<20Hz"    round to achieve a rate <= target (ceil(base/20))
"""

from __future__ import annotations

import math
import re
from typing import Union

from loguru import logger

DecimationLike = Union[int, str]
"""A rate field: a decimation int (every Nth base tick), or a frequency string resolved against a base
rate via :func:`resolve_decimation`. Form-checked by :func:`validate_decimation_like`."""

# an optional comparison prefix (< or >), a positive number (int or float), then a Hz unit.
_FREQ_RE = re.compile(r"^\s*([<>]?)\s*([0-9]*\.?[0-9]+)\s*[Hh][Zz]\s*$")


def is_frequency_string(value: DecimationLike) -> bool:
    """Whether ``value`` is a frequency string (vs an already-numeric decimation)."""
    return isinstance(value, str)


def validate_decimation_like(value: DecimationLike, *, field: str) -> None:
    """Validate form only (int >= 1, or a well-formed freq string), without a base rate.

    Resolution to an int happens later against the base rate. Raises ValueError on a bad form.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise ValueError(f"{field}: expected a decimation int or frequency string, got bool {value!r}.")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{field}: decimation must be >= 1, got {value}.")
        return
    if not isinstance(value, str) or _FREQ_RE.match(value) is None:
        raise ValueError(f"{field}: invalid {value!r}; expected an int >= 1 or e.g. '20Hz', '>20Hz', '<20Hz'.")


def resolve_decimation(value: DecimationLike, base_hz: float, *, field: str, log: bool = False) -> int:
    """Resolve ``value`` to an integer decimation >= 1 against a ``base_hz`` clock.

    An int passes through unchanged (>= 1 enforced). A bare frequency string ("20Hz") requires an
    exact decimation (``base/target`` must be a whole number) and raises otherwise. A ">"-prefixed
    string (">20Hz") floors ``base/target`` so the achieved rate is >= target; a "<"-prefixed string
    ("<20Hz") ceils it so the achieved rate is <= target. A target faster than ``base_hz`` clamps to
    1 (warned when ``log``). ``log`` is passed once at config validation; resolved-property reads stay
    silent (they re-resolve on every access, which would spam per-step logs).
    """
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise ValueError(f"{field}: expected a decimation int or frequency string, got bool {value!r}.")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{field}: decimation must be >= 1, got {value}.")
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected a decimation int or frequency string, got {value!r}.")

    m = _FREQ_RE.match(value)
    if m is None:
        raise ValueError(f"{field}: invalid frequency {value!r}; expected e.g. '20Hz', '>20Hz', '<20Hz'.")
    prefix = m.group(1)
    hz = float(m.group(2))
    if hz <= 0.0:
        raise ValueError(f"{field}: frequency must be positive, got {value!r}.")

    ratio = base_hz / hz
    if prefix == ">":
        dec = math.floor(ratio)  # floor so achieved rate >= target
    elif prefix == "<":
        dec = math.ceil(ratio)  # ceil so achieved rate <= target
    else:
        # Bare "20Hz": require an exact decimation, else the achieved rate would silently differ.
        dec = round(ratio)
        if not math.isclose(ratio, dec, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"{field}: {value} is not exactly achievable from base {base_hz:g}Hz "
                f"(base/target = {ratio:g}). Use '>{hz:g}Hz' (achieve >= target) or "
                f"'<{hz:g}Hz' (achieve <= target) to allow rounding."
            )

    if dec < 1:
        if log:
            logger.warning(f"{field}: requested {value} but base clock is {base_hz:g}Hz; capping at decimation=1.")
        dec = 1
    if log:
        logger.info(f"{field}: {value} @ base {base_hz:g}Hz -> decimation={dec} (achieved {base_hz / dec:g}Hz).")
    return dec
