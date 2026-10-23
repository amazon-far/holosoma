"""Lab 3 renderer view adapter for the shared Isaac Sim recorder."""

from typing import Any
import torch
from holosoma.simulator.isaacsim.video_recorder import IsaacSimVideoRecorder as _SharedRecorder


class IsaacSimVideoRecorder(_SharedRecorder):
    def _create_camera_view(self, prim_path: str) -> Any:
        # Recording uses the Kit RTX renderer even when Newton owns the physics.
        # A free recording camera is a renderer frame, not a Newton body/site.
        from isaaclab_physx.sim.views import FabricFrameView

        return FabricFrameView(prim_path, device=self.simulator.device)

    def _set_camera_pose(self, positions: torch.Tensor, orientations: torch.Tensor) -> None:
        with self._view.xform_world_space_writer() as writer:
            writer.set_poses(positions=positions, orientations=orientations)

    def _render_recording_frame(self) -> None:
        # Lab 3 camera sensors drive RTX on demand. A standalone Replicator product
        # must request the same update when no camera sensor or viewer is active.
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        ensure_isaac_rtx_render_update(force=True)
