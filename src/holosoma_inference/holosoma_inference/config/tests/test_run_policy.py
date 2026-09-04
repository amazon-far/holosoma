"""Unit tests for the run_policy entrypoint config flow."""

from __future__ import annotations

from holosoma_inference import run_policy as run_policy_module
from holosoma_inference.config.config_values.inference import get_annotated_inference_config
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy


def test_main_defers_default_inference_config_factory(monkeypatch):
    seen = {}
    parsed_config = object()

    def fake_parse_config(config, *args, **kwargs):
        seen["config"] = config
        seen["args"] = args
        seen["kwargs"] = kwargs
        return parsed_config

    def fake_run_policy(config):
        seen["run_config"] = config

    monkeypatch.setattr(run_policy_module.sys, "argv", ["run_policy.py"])
    monkeypatch.setattr(run_policy_module, "parse_config", fake_parse_config)
    monkeypatch.setattr(run_policy_module, "run_policy", fake_run_policy)

    run_policy_module.main()

    assert seen["config"] is get_annotated_inference_config
    assert seen["run_config"] is parsed_config


def _record_parse_config_calls(monkeypatch, *, argv, secondary=None):
    """Run main() with a fake parse_config that rejects a colliding ``config`` kwarg.

    parse_config's first positional parameter is named ``config``; a call site that also
    passes ``config=...`` as a keyword collides and raises TypeError. Mirroring that exact
    signature here means any regression of that bug surfaces as a TypeError, not a silent pass.
    """
    calls = []

    def fake_parse_config(config, *, args=None, default=None, **tyro_kwargs):
        calls.append({"config": config, "args": args, "default": default, "tyro_kwargs": tyro_kwargs})
        # First call returns the primary config; hand back an object exposing .secondary
        # so the downstream _replace/secondary branches have something to act on.
        import types

        return types.SimpleNamespace(secondary=secondary)

    def fake_run_policy(config):
        calls.append({"run": config})

    monkeypatch.setattr(run_policy_module.sys, "argv", ["run_policy.py", *argv])
    monkeypatch.setattr(run_policy_module, "parse_config", fake_parse_config)
    monkeypatch.setattr(run_policy_module, "run_policy", fake_run_policy)
    # dataclasses.replace is called on the SimpleNamespace stand-in; stub it to a no-op merge.
    import dataclasses

    def fake_replace(obj, **changes):
        import types

        merged = dict(obj.__dict__)
        merged.update(changes)
        return types.SimpleNamespace(**merged)

    monkeypatch.setattr(dataclasses, "replace", fake_replace)

    run_policy_module.main()
    return calls


def test_main_secondary_preset_path_does_not_pass_config_kwarg(monkeypatch):
    # --secondary-preset selects a registry preset, and a --secondary.* override forces the
    # secondary parse_config call (run_policy.py:207).
    calls = _record_parse_config_calls(
        monkeypatch,
        argv=["--secondary-preset", "g1-29dof-loco", "--secondary.control_dt", "0.02"],
    )
    parse_calls = [c for c in calls if "config" in c]
    assert len(parse_calls) == 2  # primary + secondary
    for c in parse_calls:
        assert "config" not in c["tyro_kwargs"], "parse_config must not receive a colliding config= kwarg"
    # The secondary call parses InferenceConfig with the preset as its default.
    assert parse_calls[1]["default"] is not None


def test_main_secondary_override_on_default_secondary(monkeypatch):
    # No preset, but the primary config has a non-None secondary and a --secondary.* override
    # is present, so the config.secondary parse_config call fires (run_policy.py:214).
    import types

    default_secondary = types.SimpleNamespace(name="default_secondary")
    calls = _record_parse_config_calls(
        monkeypatch,
        argv=["--secondary.control_dt", "0.02"],
        secondary=default_secondary,
    )
    parse_calls = [c for c in calls if "config" in c]
    assert len(parse_calls) == 2
    for c in parse_calls:
        assert "config" not in c["tyro_kwargs"]
    assert parse_calls[1]["default"] is default_secondary


def _captured_control_guide(monkeypatch, policy_class):
    lines = []

    class _Logger:
        def info(self, message=""):
            lines.append(str(message))

    monkeypatch.setattr(run_policy_module, "logger", _Logger())
    run_policy_module._print_control_guide(policy_class, use_joystick=False)
    return lines


def test_control_guide_covers_wbt_subclasses(monkeypatch):
    class _ExtensionWBTPolicy(WholeBodyTrackingPolicy):
        pass

    lines = _captured_control_guide(monkeypatch, _ExtensionWBTPolicy)

    assert any("Start motion clip" in line for line in lines)
    assert not any("Switch walking/standing mode" in line for line in lines)


def test_control_guide_keeps_locomotion_controls(monkeypatch):
    lines = _captured_control_guide(monkeypatch, LocomotionPolicy)

    assert any("Switch walking/standing mode" in line for line in lines)
    assert not any("Start motion clip" in line for line in lines)
