from __future__ import annotations

import numpy as np
import torch
from holosoma_retargeting.data_conversion.convert_data_format_mj import MotionLoader
from holosoma_retargeting.xsens.tennis_racket import TennisRacketMotion, load_tennis_racket_attachment
from scipy.spatial.transform import Rotation


def test_motion_loader_resamples_racket_pose_diagnostics_and_metadata(tmp_path) -> None:
    frame_count = 4
    qpos = np.zeros((frame_count, 36))
    qpos[:, 3] = 1.0
    quaternions_xyzw = Rotation.from_euler("z", np.array([0.0, 30.0, 60.0, 90.0])[:, None], degrees=True).as_quat()
    racket = TennisRacketMotion(
        position_m=np.column_stack((np.arange(frame_count), np.zeros((frame_count, 2)))),
        quaternion_wxyz=quaternions_xyzw[:, [3, 0, 1, 2]],
        tracking_state=np.array(["hand", "racket", "wrist_limit", "racket"]),
        symmetry_branch=np.array([-1, 0, -1, 1]),
        target_error_rad=np.linspace(0.0, 0.3, frame_count),
        source_origin_deviation_m=np.linspace(0.0, 0.03, frame_count),
        min_wrist_limit_margin_rad=np.linspace(0.2, 0.1, frame_count),
        attachment=load_tennis_racket_attachment(),
        tracking_mode="filtered",
    )
    source_path = tmp_path / "raw.npz"
    np.savez(source_path, qpos=qpos, fps=10, **racket.as_npz_payload())

    loader = MotionLoader(
        str(source_path),
        input_fps=10,
        output_fps=20,
        device=torch.device("cpu"),
        line_range=None,
        has_dynamic_object=False,
        use_omniretarget_data=False,
    )

    assert loader.tennis_racket_output is not None
    output = loader.tennis_racket_output
    assert output["tennis_racket_position_m"].shape == (loader.output_frames, 3)
    np.testing.assert_allclose(
        np.linalg.norm(output["tennis_racket_quaternion_wxyz"], axis=1),
        1.0,
        atol=1e-6,
    )
    assert output["tennis_racket_tracking_state"][0] == "hand"
    assert output["tennis_racket_tracking_mode"].item() == "filtered"
    np.testing.assert_allclose(
        output["tennis_racket_attachment_position_m"],
        racket.attachment.position_m,
    )
