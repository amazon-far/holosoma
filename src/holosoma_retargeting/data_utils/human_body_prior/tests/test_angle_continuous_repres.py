import torch


def test_bgs_runs_on_cpu():
    from human_body_prior.tools.angle_continuous_repres import bgs

    # one sample with two 3D vectors (shape: Bx3x2)
    d6 = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]], dtype=torch.float32)
    out = bgs(d6)
    # Expect a rotation matrix: shape (B, 3, 3)
    assert out.shape == (1, 3, 3)

