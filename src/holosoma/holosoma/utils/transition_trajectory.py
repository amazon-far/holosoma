"""Building blocks for synthesising a transition segment between two motion frames.

These are used to splice a lead-in / lead-out around a reference motion clip (see
``managers/command/terms/wbt.py``). The contract that matters is *derivative consistency*:
whatever a caller stores as a velocity must be the derivative of what it stores as a position,
because a tracking policy is commanded on both and penalised for disagreeing with either.

Two ways to honour that contract live here, and the choice is not stylistic:

* ``hermite_segment`` returns a position and its **exact analytic** derivative from one
  polynomial. Use it wherever the quantity is genuinely a free variable of the trajectory
  (joint angles, the root position).
* ``linear_velocity_from_positions`` / ``angular_velocity_from_quats`` **difference** a pose
  sequence. Use them for quantities that are *derived* from the free variables through
  kinematics (body poses), where no closed-form derivative is available without a Jacobian.
  Difference the same reference point the consumer expects -- a link-origin position sequence
  yields a link-origin velocity, which is not the centre-of-mass velocity a motion file stores.
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
    slerp,
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
        duration_s: segment duration. Must be the *effective* duration
            ``num_intervals * dt``, not a nominal config value, or the returned velocities are
            scaled wrong relative to the frame spacing the consumer replays them at.
        num_intervals: number of steps; the result has ``num_intervals + 1`` frames spanning
            both endpoints inclusive.

    Returns:
        ``(position, velocity)``, each shaped ``(num_intervals + 1, *p0.shape)``. ``velocity``
        is the exact time derivative of ``position``, and both endpoints are hit exactly in
        position *and* velocity, which is what makes the seam into the clip continuous.

    A cubic is the exact fit to the four constraints we have. A quintic would need two more, and
    the only ones available are the endpoint accelerations: zero is correct at a resting endpoint
    but wrong at a clip frame, which is generally accelerating. Imposing it anyway also raises peak
    mid-segment speed by 25% for the same endpoints.
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


def slerp_series(q0: torch.Tensor, q1: torch.Tensor, num_intervals: int) -> torch.Tensor:
    """Spherically interpolate from ``q0`` to ``q1`` over a uniform grid.

    Args:
        q0, q1: quaternions of matching shape ``(..., 4)``. A bare ``(4,)`` is accepted, so
            callers do not have to hand-roll ``unsqueeze``/``squeeze`` around a single rotation.
        num_intervals: as in :func:`hermite_segment`.

    Returns:
        ``(num_intervals + 1, *q0.shape)``, unit norm, endpoints included. Quaternion order
        follows the inputs; :func:`holosoma.utils.rotations.slerp` is order-agnostic.

    Sign continuity across the series comes from ``slerp`` flipping ``q1`` into ``q0``'s
    hemisphere. Do not "clean up" the result with ``quat_normalize`` -- that forces w>=0
    per frame, which would tear the series in half wherever it crosses w=0.
    """
    if q0.shape != q1.shape:
        raise ValueError(f"Quaternion endpoint shapes must match, got {q0.shape}/{q1.shape}")
    if q0.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with trailing dim 4, got {q0.shape}")
    if num_intervals < 1:
        raise ValueError(f"Transition must contain at least one interval, got {num_intervals}")

    num_frames = num_intervals + 1
    num_quats = q0.numel() // 4
    tau = torch.linspace(0.0, 1.0, num_frames, device=q0.device, dtype=q0.dtype)

    flat0 = q0.reshape(1, -1, 4).expand(num_frames, -1, -1).reshape(-1, 4)
    flat1 = q1.reshape(1, -1, 4).expand(num_frames, -1, -1).reshape(-1, 4)
    blend = tau.repeat_interleave(num_quats).unsqueeze(-1)

    interpolated = quat_unit(slerp(flat0, flat1, blend))
    return interpolated.reshape((num_frames,) + tuple(q0.shape))


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

    :func:`slerp_series` turns at a constant rate, so it lands on ``q1``'s orientation with an
    angular velocity of its own choosing. Splicing that against a clip whose root is already
    turning leaves a step in the commanded angular velocity at the seam, of the clip's full
    turn rate. This runs the cubic in rotation-vector space instead, so the rate matches too.

    Matching a nonzero ``omega1`` while starting from rest necessarily means turning past the
    endpoint orientation and back -- there is no way to be at a fixed orientation, at rest, and
    then arrive at that same orientation already turning. That excursion is the honest cost of a
    continuous hand-off, and is the same trade the position channels already make.

    Angular velocity is not the plain derivative of a rotation vector away from the origin, so the
    endpoint derivative is mapped through the inverse left Jacobian of SO(3). Without that the
    endpoint rate is only matched to first order, and a rotation whose axis differs from ``omega1``
    keeps tens of percent of the error.
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
    """Axis scaled by angle, in radians, for xyzw quaternions.

    Despite the name, ``quat_to_angle_axis`` already scales its second return value by the angle,
    so multiplying by the returned angle would apply that scale twice.
    """
    _, rotation_vector = quat_to_angle_axis(quat)
    return rotation_vector
