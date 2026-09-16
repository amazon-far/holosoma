"""Top-level inference configuration types for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from .observation import ObservationConfig
from .robot import RobotConfig
from .task import TaskConfig


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level configuration for policy inference.

    Combines robot, observation, and task configurations
    for running policies on real robots or in simulation.
    """

    robot: RobotConfig
    """Robot hardware and control configuration."""

    observation: ObservationConfig
    """Observation space configuration."""

    task: TaskConfig
    """Task execution configuration."""

    name: str = ""
    """Registered-policy identity for multi-policy serving (the launcher MODE:
    ``loco`` / ``wbt`` / ``stair`` / ``contextual_loco``). This is the stable key a
    client uses to switch to a specific policy and the label published in the
    policy-status topic. Empty falls back to ``task.policy_type`` (then an index)
    when the multi-mode registry is built. Distinct from ``task.policy_type``,
    which selects the policy CLASS via the ``holosoma.policies.by_type`` group."""

    secondary: InferenceConfig | None = None
    """DEPRECATED (kept for back-compat): secondary policy config for the original
    two-policy dual-mode (X-button switch). Prefer ``policies`` for N-policy
    serving. When set, the service node normalizes it into the policy registry as
    if it were the first entry of ``policies``. Set to None to disable.
    Override any field: --secondary.task.model-path, --secondary.task.rl-rate, etc."""

    policies: tuple[InferenceConfig, ...] = ()
    """Additional registered policies for N-policy serving (this top-level config
    is index 0 / the hardware owner; each entry here is another switchable policy).
    All must be hardware-COMPATIBLE with index 0 (same robot_type / num DOFs / SDK)
    — the service node validates and fails loud on a mismatch. Each policy keeps
    its OWN ``observation`` / ``task`` (incl. ``task.depth``), so per-policy sensor
    configs differ while the robot hardware + input providers are shared. A client
    switches between them by ``name`` (see the policy-status / select-policy
    topics); per-policy checkpoint switching (``task.model_path`` list) is
    unchanged. Nested ``policies`` / ``secondary`` on entries are ignored."""

    def registered_policies(self) -> list[InferenceConfig]:
        """All policies to register, index 0 = hardware owner, in switch order.

        Normalizes the legacy ``secondary`` into the same list as ``policies`` so
        the service node / ``MultiModePolicy`` have a single code path. Setting
        BOTH ``secondary`` and ``policies`` is ambiguous and rejected. A
        single-policy service returns ``[self]`` (length 1), which the service
        node builds directly (no MultiModePolicy wrapper needed).
        """
        if self.secondary is not None and self.policies:
            raise ValueError(
                "Set either `secondary` (legacy two-policy dual-mode) or `policies` "
                "(N-policy serving), not both."
            )
        extras = [self.secondary] if self.secondary is not None else list(self.policies)
        return [self, *extras]

    def resolved_names(self) -> list[str]:
        """Switch-key name for each registered policy, in order, de-duplicated.

        Name precedence per policy: explicit ``name`` -> ``task.policy_type`` ->
        ``"policy"``. Collisions get a ``#N`` suffix so every returned name is
        unique within the service (names are the switch key + status label, so
        they must be unambiguous). Mirrors ``registered_policies`` order.
        """
        names: list[str] = []
        seen: dict[str, int] = {}
        for cfg in self.registered_policies():
            base = cfg.name or getattr(cfg.task, "policy_type", "") or "policy"
            if base in seen:
                seen[base] += 1
                names.append(f"{base}#{seen[base]}")
            else:
                seen[base] = 1
                names.append(base)
        return names
