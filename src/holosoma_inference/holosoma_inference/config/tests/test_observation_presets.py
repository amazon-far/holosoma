"""Invariants for the built-in observation presets.

Guards against silent drift that breaks a deployed ONNX checkpoint:

* ``wbt`` stays at 154 x history_length 1 (the stock six-term preset).
* ``wbt-dense`` stays at 157 x history_length 4 -> 628-dim obs, which
  matches the g1-wbt dense training config. Regressions here manifest
  at runtime as
  ``InvalidArgument: Got invalid dimensions for input: obs``.
"""

from __future__ import annotations

from holosoma_inference.config.config_values.observation import DEFAULTS


def _flat_actor_obs_dim(cfg) -> int:
    """Sum of the per-term dims listed in ``actor_obs`` (pre-history)."""
    terms = cfg.obs_dict["actor_obs"]
    return sum(cfg.obs_dims[t] for t in terms)


class TestStockWbtPreset:
    """``wbt`` — 6 actor terms, history 1, 154-dim obs."""

    def test_actor_obs_terms(self) -> None:
        cfg = DEFAULTS["wbt"]
        assert cfg.obs_dict["actor_obs"] == [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]

    def test_actor_obs_dim_sums_to_154(self) -> None:
        assert _flat_actor_obs_dim(DEFAULTS["wbt"]) == 154

    def test_history_length_is_one(self) -> None:
        assert DEFAULTS["wbt"].history_length_dict["actor_obs"] == 1


class TestWbtDensePreset:
    """``wbt-dense`` — 7 actor terms incl. ``projected_gravity``, history 4."""

    def test_actor_obs_terms(self) -> None:
        cfg = DEFAULTS["wbt-dense"]
        assert cfg.obs_dict["actor_obs"] == [
            "motion_command",
            "motion_ref_ori_b",
            "projected_gravity",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]

    def test_actor_obs_dim_sums_to_157(self) -> None:
        assert _flat_actor_obs_dim(DEFAULTS["wbt-dense"]) == 157

    def test_history_length_is_four(self) -> None:
        assert DEFAULTS["wbt-dense"].history_length_dict["actor_obs"] == 4

    def test_flattened_obs_matches_model_input(self) -> None:
        cfg = DEFAULTS["wbt-dense"]
        flat = _flat_actor_obs_dim(cfg) * cfg.history_length_dict["actor_obs"]
        assert flat == 628

    def test_projected_gravity_scale_present(self) -> None:
        cfg = DEFAULTS["wbt-dense"]
        assert cfg.obs_scales.get("projected_gravity") == 1.0


class TestWbtInferencePreset:
    """``g1-29dof-wbt-dense`` wires robot + dense obs + wbt task."""

    def test_inference_default_exposes_dense(self) -> None:
        from holosoma_inference.config.config_values.inference import get_defaults

        defaults = get_defaults()
        assert "g1-29dof-wbt-dense" in defaults
        cfg = defaults["g1-29dof-wbt-dense"]
        assert cfg.observation is DEFAULTS["wbt-dense"]
