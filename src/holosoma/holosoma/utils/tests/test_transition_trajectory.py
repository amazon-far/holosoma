import pytest
import torch

from holosoma.utils.rotations import quat_conjugate, quat_from_angle_axis, quat_mul, quat_to_angle_axis
from holosoma.utils.transition_trajectory import (
    _left_jacobian_inverse,
    angular_velocity_from_quats,
    hermite_rotation_series,
    hermite_segment,
    linear_velocity_from_positions,
    slerp_series,
)

# The root conftest applies no_sim as a fallback, but be explicit like the sibling test modules.
pytestmark = pytest.mark.no_sim

DT = 0.02
NUM_INTERVALS = 100
DURATION = NUM_INTERVALS * DT


def test_hermite_hits_both_endpoints_in_position_and_velocity():
    p0 = torch.tensor([0.0, 1.0, -2.0])
    v0 = torch.zeros(3)
    p1 = torch.tensor([0.5, -1.0, 3.0])
    v1 = torch.tensor([1.5, 0.25, -0.75])

    pos, vel = hermite_segment(p0, v0, p1, v1, DURATION, NUM_INTERVALS)

    assert pos.shape == (NUM_INTERVALS + 1, 3)
    assert vel.shape == (NUM_INTERVALS + 1, 3)
    torch.testing.assert_close(pos[0], p0)
    torch.testing.assert_close(pos[-1], p1)
    torch.testing.assert_close(vel[0], v0)
    torch.testing.assert_close(vel[-1], v1)


def test_hermite_velocity_integrates_back_to_its_own_position():
    """The defect this module exists to prevent: a velocity that is not d/dt of the position.

    The old path interpolated position and velocity independently, so the stored velocity was a
    ramp where the true derivative was a constant. Simpson's rule is exact for the quadratic
    velocity of a cubic, so this is an equality rather than an approximation.
    """
    torch.manual_seed(0)
    p0, p1 = torch.randn(29), torch.randn(29)
    v0, v1 = torch.zeros(29), torch.randn(29) * 0.5

    pos, vel = hermite_segment(p0, v0, p1, v1, DURATION, NUM_INTERVALS)

    simpson = (DT / 3.0) * (vel[:-2] + 4.0 * vel[1:-1] + vel[2:])
    torch.testing.assert_close(simpson, pos[2:] - pos[:-2], rtol=1e-5, atol=1e-6)


def test_hermite_velocity_matches_what_a_consumer_would_measure():
    """A consumer differencing the stored positions must recover the stored velocities.

    Agreement is limited by the central difference's own O(dt^2) truncation error, so the bound
    is stated relative to peak speed rather than pointwise -- the analytic velocity crosses zero,
    where a relative comparison is meaningless.
    """
    torch.manual_seed(0)
    p0, p1 = torch.randn(29), torch.randn(29)
    v0, v1 = torch.zeros(29), torch.randn(29) * 0.5

    pos, vel = hermite_segment(p0, v0, p1, v1, DURATION, NUM_INTERVALS)
    # Interior only: torch.gradient is one-sided at the ends, which is a coarser estimate.
    numeric = linear_velocity_from_positions(pos, DT)[1:-1]

    error = (vel[1:-1] - numeric).abs().max()
    assert error < 1e-3 * vel.abs().max()


def test_hermite_peak_speed_is_cubic_not_quintic():
    """Guards the interpolant choice: a quintic would peak at 1.875x, a cubic at 1.5x."""
    displacement = 2.0
    p0 = torch.zeros(1)
    p1 = torch.full((1,), displacement)
    zero = torch.zeros(1)

    _, vel = hermite_segment(p0, zero, p1, zero, DURATION, NUM_INTERVALS)

    peak = vel.abs().max().item()
    assert peak == pytest.approx(1.5 * displacement / DURATION, rel=1e-3)


def test_hermite_reduces_to_a_straight_line_for_matched_endpoint_velocities():
    """With v0 = v1 = (p1 - p0) / T the cubic terms vanish, so the result is exactly linear."""
    p0 = torch.tensor([0.0, 0.0])
    p1 = torch.tensor([1.0, -2.0])
    v = (p1 - p0) / DURATION

    pos, vel = hermite_segment(p0, v, p1, v, DURATION, NUM_INTERVALS)

    tau = torch.linspace(0.0, 1.0, NUM_INTERVALS + 1).unsqueeze(-1)
    torch.testing.assert_close(pos, p0 + (p1 - p0) * tau)
    torch.testing.assert_close(vel, v.expand_as(vel).contiguous())


@pytest.mark.parametrize("shape", [(29,), (3,), (51, 3), (4, 5, 3)])
def test_hermite_broadcasts_over_trailing_dims(shape):
    torch.manual_seed(1)
    p0, v0, p1, v1 = (torch.randn(shape) for _ in range(4))

    pos, vel = hermite_segment(p0, v0, p1, v1, DURATION, NUM_INTERVALS)

    assert pos.shape == (NUM_INTERVALS + 1, *shape)
    assert vel.shape == (NUM_INTERVALS + 1, *shape)
    torch.testing.assert_close(pos[0], p0)
    torch.testing.assert_close(pos[-1], p1)


def test_hermite_preserves_input_dtype():
    p0 = torch.zeros(3, dtype=torch.float32)
    pos, vel = hermite_segment(p0, p0, p0 + 1.0, p0, DURATION, NUM_INTERVALS)
    assert pos.dtype == torch.float32
    assert vel.dtype == torch.float32


@pytest.mark.parametrize(
    ("duration", "num_intervals", "message"),
    [
        (0.0, 10, "duration must be positive"),
        (-1.0, 10, "duration must be positive"),
        (1.0, 0, "at least one interval"),
        (1.0, -3, "at least one interval"),
    ],
)
def test_hermite_rejects_degenerate_grids(duration, num_intervals, message):
    p = torch.zeros(3)
    with pytest.raises(ValueError, match=message):
        hermite_segment(p, p, p, p, duration, num_intervals)


def test_hermite_rejects_mismatched_endpoint_shapes():
    with pytest.raises(ValueError, match="shapes must all match"):
        hermite_segment(torch.zeros(3), torch.zeros(3), torch.zeros(4), torch.zeros(3), 1.0, 10)


def test_slerp_series_hits_endpoints_and_stays_unit_norm():
    axis = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    q0 = quat_from_angle_axis(torch.tensor([0.0, 0.1]), axis, w_last=True)
    q1 = quat_from_angle_axis(torch.tensor([1.2, -0.8]), axis, w_last=True)

    series = slerp_series(q0, q1, NUM_INTERVALS)

    assert series.shape == (NUM_INTERVALS + 1, 2, 4)
    torch.testing.assert_close(series[0], q0)
    torch.testing.assert_close(series[-1], q1, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(series.norm(dim=-1), torch.ones(NUM_INTERVALS + 1, 2))


def test_slerp_series_accepts_a_bare_quaternion():
    q0 = quat_from_angle_axis(torch.tensor([0.0]), torch.tensor([[0.0, 1.0, 0.0]]), w_last=True)[0]
    q1 = quat_from_angle_axis(torch.tensor([0.9]), torch.tensor([[0.0, 1.0, 0.0]]), w_last=True)[0]

    series = slerp_series(q0, q1, 10)

    assert series.shape == (11, 4)
    torch.testing.assert_close(series[0], q0)
    torch.testing.assert_close(series[-1], q1, atol=1e-6, rtol=1e-6)


def test_slerp_series_of_identical_endpoints_is_constant():
    q = quat_from_angle_axis(torch.tensor([0.4]), torch.tensor([[0.0, 0.0, 1.0]]), w_last=True)
    series = slerp_series(q, q, 10)
    torch.testing.assert_close(series, q.expand_as(series).contiguous())


def test_slerp_series_rejects_mismatched_shapes():
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="shapes must match"):
        slerp_series(q, q.expand(2, 4), 10)


def test_linear_velocity_recovers_a_constant_rate():
    velocity = torch.tensor([1.0, -2.0, 0.5])
    steps = torch.arange(20, dtype=torch.float32).unsqueeze(-1)
    pos = steps * velocity * DT

    derived = linear_velocity_from_positions(pos, DT)

    torch.testing.assert_close(derived, velocity.expand_as(derived).contiguous())


def test_linear_velocity_of_a_single_frame_is_zero():
    pos = torch.randn(1, 5, 3)
    torch.testing.assert_close(linear_velocity_from_positions(pos, DT), torch.zeros_like(pos))


def test_angular_velocity_recovers_a_constant_body_rate():
    axis = torch.tensor([0.0, 0.0, 1.0])
    rate = 1.3  # rad/s
    steps = torch.arange(20, dtype=torch.float32)
    quats = quat_from_angle_axis(steps * rate * DT, axis.expand(20, 3), w_last=True)

    omega = angular_velocity_from_quats(quats, DT)

    expected = (axis * rate).expand(20, 3)
    torch.testing.assert_close(omega, expected.contiguous(), rtol=1e-4, atol=1e-5)


def test_angular_velocity_is_world_frame_not_body_frame():
    """A world-frame omega must be independent of any constant pre-rotation of the sequence."""
    axis = torch.tensor([0.0, 0.0, 1.0])
    steps = torch.arange(12, dtype=torch.float32)
    spin = quat_from_angle_axis(steps * 0.9 * DT, axis.expand(12, 3), w_last=True)
    offset = quat_from_angle_axis(torch.tensor([0.7]), torch.tensor([[1.0, 0.0, 0.0]]), w_last=True)

    plain = angular_velocity_from_quats(spin, DT)
    # Right-multiplying by a constant re-parameterises the body frame; world omega is unchanged.
    rotated = angular_velocity_from_quats(quat_mul(spin, offset.expand(12, 4), w_last=True), DT)

    torch.testing.assert_close(plain, rotated, rtol=1e-4, atol=1e-5)


def test_angular_velocity_of_a_static_sequence_is_zero():
    q = quat_from_angle_axis(torch.tensor([0.3]), torch.tensor([[0.0, 1.0, 0.0]]), w_last=True)
    omega = angular_velocity_from_quats(q.expand(8, 4).contiguous(), DT)
    torch.testing.assert_close(omega, torch.zeros(8, 3), atol=1e-6, rtol=0.0)


def test_angular_velocity_handles_per_body_sequences():
    torch.manual_seed(2)
    axes = torch.nn.functional.normalize(torch.randn(51, 3), dim=-1)
    steps = torch.arange(15, dtype=torch.float32)
    angles = (steps.unsqueeze(-1) * 0.5 * DT).expand(15, 51)
    quats = quat_from_angle_axis(angles.reshape(-1), axes.expand(15, 51, 3).reshape(-1, 3), w_last=True)
    quats = quats.reshape(15, 51, 4)

    omega = angular_velocity_from_quats(quats, DT)

    assert omega.shape == (15, 51, 3)
    torch.testing.assert_close(omega, (axes * 0.5).expand(15, 51, 3).contiguous(), rtol=1e-4, atol=1e-5)


def test_angular_velocity_of_a_two_frame_sequence_uses_a_forward_difference():
    axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    quats = quat_from_angle_axis(torch.tensor([0.0, 0.6 * DT]), axis, w_last=True)

    omega = angular_velocity_from_quats(quats, DT)

    assert omega.shape == (2, 3)
    torch.testing.assert_close(omega[0], omega[1])
    torch.testing.assert_close(omega[0], torch.tensor([0.0, 0.0, 0.6]), rtol=1e-4, atol=1e-5)


def test_angular_velocity_of_a_single_frame_is_zero():
    q = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    torch.testing.assert_close(angular_velocity_from_quats(q, DT), torch.zeros(1, 1, 3))


@pytest.mark.parametrize("dt", [0.0, -0.01])
def test_velocity_helpers_reject_nonpositive_dt(dt):
    with pytest.raises(ValueError, match="dt must be positive"):
        linear_velocity_from_positions(torch.zeros(5, 3), dt)
    with pytest.raises(ValueError, match="dt must be positive"):
        angular_velocity_from_quats(torch.zeros(5, 4), dt)


def test_rotation_series_hits_both_endpoint_orientations():
    z = torch.tensor([0.0, 0.0, 1.0])
    q0 = quat_from_angle_axis(torch.tensor([0.2]), z.unsqueeze(0), w_last=True)[0]
    q1 = quat_from_angle_axis(torch.tensor([1.0]), z.unsqueeze(0), w_last=True)[0]

    series = hermite_rotation_series(q0, torch.zeros(3), q1, z * 0.8, DURATION, NUM_INTERVALS)

    assert series.shape == (NUM_INTERVALS + 1, 4)
    torch.testing.assert_close(series[0], q0, atol=1e-6, rtol=1e-6)
    # q and -q are the same rotation, so compare by angle.
    assert float(1.0 - (series[-1] * q1).sum().abs()) < 1e-6
    torch.testing.assert_close(series.norm(dim=-1), torch.ones(NUM_INTERVALS + 1), atol=1e-6, rtol=0.0)


def test_rotation_series_hits_both_endpoint_angular_velocities():
    """The reason this exists: slerp lands on the orientation but not the turn rate.

    A lead-in that arrives at the right orientation turning at the wrong rate leaves a step in the
    commanded angular velocity at the seam, of the clip's whole turn rate.
    """
    z = torch.tensor([0.0, 0.0, 1.0])
    q0 = quat_from_angle_axis(torch.tensor([0.0]), z.unsqueeze(0), w_last=True)[0]
    q1 = quat_from_angle_axis(torch.tensor([0.5]), z.unsqueeze(0), w_last=True)[0]
    omega1 = z * 1.2

    series = hermite_rotation_series(q0, torch.zeros(3), q1, omega1, DURATION, NUM_INTERVALS)
    measured = angular_velocity_from_quats(series, DT)

    # A central difference's end frames repeat their nearest interior value, so both measurements
    # are one step in from the endpoint -- and the cubic is still accelerating there. One step of
    # the endpoint acceleration is ~3% of omega1 here, which is what the tolerances allow.
    # test_rotation_series_beats_slerp_at_the_endpoint_rate is the discriminating comparison.
    assert float(measured[0].abs().max()) < 0.02 * float(omega1.norm())
    torch.testing.assert_close(measured[-1], omega1, atol=1e-3, rtol=0.05)


def test_rotation_series_endpoint_rate_converges_as_the_grid_refines():
    """Proves the endpoint rate is matched *exactly*, rather than to some tolerance.

    The measured rate is a difference over the final interval, so it lags the endpoint by one step of
    the cubic's endpoint acceleration. If the rate is genuinely matched that residual is O(dt) and
    vanishes as the grid refines; if it is not, it plateaus at a fixed error. Deliberately uses a
    rotation whose axis differs from omega1, which is the case the SO(3) Jacobian correction exists
    for -- without it this plateaus instead.
    """
    z = torch.tensor([0.0, 0.0, 1.0])
    q0 = quat_from_angle_axis(torch.tensor([0.35]), torch.tensor([[0.6, 0.0, 0.8]]), w_last=True)[0]
    q1 = quat_from_angle_axis(torch.tensor([0.5]), z.unsqueeze(0), w_last=True)[0]
    omega1 = z * 1.2
    duration = 2.0

    def endpoint_gap(num_intervals):
        series = hermite_rotation_series(q0, torch.zeros(3), q1, omega1, duration, num_intervals)
        relative = quat_mul(series[-1:], quat_conjugate(series[-2:-1], w_last=True), w_last=True)
        _, rotvec = quat_to_angle_axis(relative)
        return float((rotvec[0] / (duration / num_intervals) - omega1).abs().max())

    coarse, fine, finer = endpoint_gap(20), endpoint_gap(200), endpoint_gap(2000)
    assert fine < coarse / 5.0, f"gap did not shrink with dt: {coarse:.5f} -> {fine:.5f}"
    assert finer < fine / 5.0, f"gap did not shrink with dt: {fine:.5f} -> {finer:.5f}"
    assert finer < 1e-2


def test_left_jacobian_inverse_is_the_identity_at_zero():
    torch.testing.assert_close(_left_jacobian_inverse(torch.zeros(3)), torch.eye(3), atol=1e-9, rtol=0.0)


def test_rotation_series_beats_slerp_at_the_endpoint_rate():
    """Guards against a silent revert to slerp_series for the root rotation."""
    z = torch.tensor([0.0, 0.0, 1.0])
    q0 = quat_from_angle_axis(torch.tensor([0.0]), z.unsqueeze(0), w_last=True)[0]
    q1 = quat_from_angle_axis(torch.tensor([0.5]), z.unsqueeze(0), w_last=True)[0]
    omega1 = z * 1.2

    hermite_error = (
        (
            angular_velocity_from_quats(
                hermite_rotation_series(q0, torch.zeros(3), q1, omega1, DURATION, NUM_INTERVALS), DT
            )[-1]
            - omega1
        )
        .abs()
        .max()
    )
    slerp_error = (angular_velocity_from_quats(slerp_series(q0, q1, NUM_INTERVALS), DT)[-1] - omega1).abs().max()

    assert float(hermite_error) < 0.05 * float(slerp_error)


def test_rotation_series_of_equal_endpoints_at_rest_is_constant():
    q = quat_from_angle_axis(torch.tensor([0.4]), torch.tensor([[0.0, 1.0, 0.0]]), w_last=True)[0]
    series = hermite_rotation_series(q, torch.zeros(3), q, torch.zeros(3), DURATION, NUM_INTERVALS)
    torch.testing.assert_close(series, q.expand_as(series).contiguous(), atol=1e-6, rtol=1e-6)


def test_rotation_series_rejects_batched_input():
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="shape"):
        hermite_rotation_series(q, torch.zeros(3), q, torch.zeros(3), DURATION, NUM_INTERVALS)
