"""Lead-in / lead-out trajectories whose stored velocities are the derivative of their positions.

``hermite_segment`` gives an analytic derivative for free variables (joint angles, root position);
the ``*_velocity_from_*`` helpers difference poses derived through kinematics (body poses).
"""

from __future__ import annotations

import math

import torch

from holosoma.utils.rotations import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_to_angle_axis,
    quat_unit,
)


def hermite_segment(
    p0: torch.Tensor,
    v0: torch.Tensor,
    p1: torch.Tensor,
    v1: torch.Tensor,
    duration_s: float,
    num_intervals: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cubic Hermite trajectory between two (position, velocity) endpoints.

    Args:
        p0, v0: position and velocity at t=0. Any shape, all four arguments must match.
        p1, v1: position and velocity at t=duration_s.
        duration_s: segment duration; must equal ``num_intervals * dt`` or velocities are mis-scaled.
        num_intervals: number of steps; the result has ``num_intervals + 1`` frames spanning
            both endpoints inclusive.

    Returns:
        ``(position, velocity)``, each shaped ``(num_intervals + 1, *p0.shape)``. ``velocity``
        is the exact time derivative of ``position``.
    """
    if duration_s <= 0.0:
        raise ValueError(f"Transition duration must be positive, got {duration_s}")
    if num_intervals < 1:
        raise ValueError(f"Transition must contain at least one interval, got {num_intervals}")
    if not (p0.shape == v0.shape == p1.shape == v1.shape):
        raise ValueError(f"Endpoint shapes must all match, got {p0.shape}, {v0.shape}, {p1.shape}, {v1.shape}")

    out_dtype = p0.dtype
    # float64 keeps the endpoint round-trip exact; this runs once at setup.
    p0_64, v0_64 = p0.double(), v0.double()
    p1_64, v1_64 = p1.double(), v1.double()

    tau = torch.linspace(0.0, 1.0, num_intervals + 1, device=p0.device, dtype=torch.float64)
    tau = tau.reshape((-1,) + (1,) * p0.ndim)

    a0 = p0_64
    a1 = duration_s * v0_64
    c0 = p1_64 - a0 - a1
    c1 = duration_s * v1_64 - a1
    a2 = 3.0 * c0 - c1
    a3 = -2.0 * c0 + c1

    position = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3
    velocity = (a1 + 2.0 * a2 * tau + 3.0 * a3 * tau**2) / duration_s
    return position.to(out_dtype), velocity.to(out_dtype)


def hermite_rotation_series(
    q0: torch.Tensor,
    omega0: torch.Tensor,
    q1: torch.Tensor,
    omega1: torch.Tensor,
    duration_s: float,
    num_intervals: int,
) -> torch.Tensor:
    """Rotate from ``q0`` to ``q1`` hitting both endpoint *angular velocities* as well.

    Args:
        q0, q1: single xyzw quaternions, shape ``(4,)``.
        omega0, omega1: world-frame angular velocity at each endpoint, shape ``(3,)``.
        duration_s, num_intervals: as in :func:`hermite_segment`.

    Returns:
        ``(num_intervals + 1, 4)``, unit norm, endpoints included.

    Runs the cubic in rotation-vector space, mapping ``omega1`` through the inverse left Jacobian
    of SO(3) so the endpoint rate is matched exactly, not only to first order.
    """
    for name, value, expected in (("q0", q0, 4), ("q1", q1, 4), ("omega0", omega0, 3), ("omega1", omega1, 3)):
        if value.shape != (expected,):
            raise ValueError(f"hermite_rotation_series expects {name} of shape ({expected},), got {value.shape}")

    # World-frame relative rotation, as a vector whose magnitude is the angle to turn through.
    relative = quat_mul(q1.unsqueeze(0), quat_conjugate(q0.unsqueeze(0), w_last=True), w_last=True)
    total = _rotation_vector(relative)[0]

    # omega = Jl(r) * dr/dt, so the endpoint derivatives are the angular velocities pulled back
    # through Jl. At r = 0 that is the identity, hence omega0 passes straight through.
    rotation_vector, _ = hermite_segment(
        torch.zeros_like(total),
        omega0,
        total,
        _left_jacobian_inverse(total) @ omega1,
        duration_s,
        num_intervals,
    )
    angle = rotation_vector.norm(dim=-1)
    axis = rotation_vector / angle.clamp_min(1e-12).unsqueeze(-1)
    delta = quat_from_angle_axis(angle, axis, w_last=True)
    return quat_unit(quat_mul(delta, q0.expand_as(delta).contiguous(), w_last=True))


def _left_jacobian_inverse(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Inverse left Jacobian of SO(3) at ``rotation_vector``, shape ``(3, 3)``.

    Maps a world-frame angular velocity to the rotation-vector derivative that produces it:
    ``dr/dt = Jl^-1(r) * omega``. Reduces to the identity at ``r = 0``.
    """
    theta = float(rotation_vector.norm())
    skew = torch.zeros(3, 3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    rx, ry, rz = rotation_vector
    skew[0, 1], skew[0, 2] = -rz, ry
    skew[1, 0], skew[1, 2] = rz, -rx
    skew[2, 0], skew[2, 1] = -ry, rx

    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    if theta < 1e-6:
        # The quadratic coefficient tends to 1/12; below this the sin(theta) division loses more
        # precision than the truncation costs.
        coefficient = 1.0 / 12.0
    else:
        coefficient = 1.0 / theta**2 - (1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta))
    return identity - 0.5 * skew + coefficient * (skew @ skew)


def linear_velocity_from_positions(pos: torch.Tensor, dt: float) -> torch.Tensor:
    """Differentiate a position sequence along its leading (time) axis.

    Central differences in the interior, one-sided at the two ends, matching how the retargeting
    converter derives its joint and base velocities.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if pos.shape[0] < 2:
        return torch.zeros_like(pos)
    return torch.gradient(pos, spacing=dt, dim=0)[0]


def angular_velocity_from_quats(quat: torch.Tensor, dt: float) -> torch.Tensor:
    """World-frame angular velocity of a quaternion sequence along its leading (time) axis.

    Args:
        quat: ``(T, ..., 4)`` in **xyzw** order.
        dt: frame spacing.

    Returns:
        ``(T, ..., 3)``. Central SO(3) differences in the interior; the two end frames repeat
        their nearest interior value.

    The relative rotation is applied on the left (``q_next * q_prev^-1``), which yields
    world-frame omega to match the world-frame body velocities in a motion file.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with trailing dim 4, got {quat.shape}")
    quat = quat.contiguous()  # quat_mul reshapes, which needs a contiguous input

    num_frames = quat.shape[0]
    zeros_shape = quat.shape[:-1] + (3,)
    if num_frames < 2:
        return quat.new_zeros(zeros_shape)

    if num_frames == 2:
        relative = quat_mul(quat[1:], quat_conjugate(quat[:1], w_last=True), w_last=True)
        omega = _rotation_vector(relative) / dt
        return torch.cat([omega, omega], dim=0)

    relative = quat_mul(quat[2:], quat_conjugate(quat[:-2], w_last=True), w_last=True)
    omega = _rotation_vector(relative) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def _rotation_vector(quat: torch.Tensor) -> torch.Tensor:
    """Axis scaled by angle, in radians, for xyzw quaternions."""
    # quat_to_angle_axis's second return is already scaled by the angle.
    _, rotation_vector = quat_to_angle_axis(quat)
    return rotation_vector
