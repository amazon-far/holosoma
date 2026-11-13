"""Unit tests for the shared simulation utilities (pure, no simulator backend)."""

from __future__ import annotations

from holosoma.utils.sim_utils import format_simulation_performance


def test_format_simulation_performance_names_rates_and_time_ratio() -> None:
    message = format_simulation_performance(
        physics_steps=1000,
        control_steps=250,
        sim_dt=0.005,
        wall_time=2.5,
    )

    assert message == (
        "Simulation performance: 2.00x real time (5.00 sim s / 2.50 wall s) | "
        "400.0 physics steps/s | 100.0 control steps/s"
    )


def test_format_simulation_performance_handles_empty_measurement() -> None:
    assert (
        format_simulation_performance(
            physics_steps=0,
            control_steps=0,
            sim_dt=0.005,
            wall_time=0.0,
        )
        == "Simulation performance: unavailable (no wall time elapsed)"
    )
