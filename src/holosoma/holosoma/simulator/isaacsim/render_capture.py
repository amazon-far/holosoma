"""Render applied state through Isaac Lab with physics held stopped."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.simulator.base_simulator.state import DERIVED

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim

_PLAY_SIMULATIONS = "/app/player/playSimulations"


class RenderCapture:
    """Keep Isaac physics stopped while Isaac Lab's render performs FK and Fabric sync.

    Owns lazy camera initialization. Isaac Lab's ``render()`` re-enables physics playback, so every
    render here turns it off again.
    """

    def __init__(self, simulator: IsaacSim) -> None:
        import carb

        if simulator.sensor_manager.lidars:
            raise ValueError("Isaac renders applied state for cameras only")
        self.simulator = simulator
        self.settings = carb.settings.get_settings()
        for camera in simulator.tiled_cameras.values():
            camera.cfg.update_latest_camera_pose = True
        self.settings.set_bool(_PLAY_SIMULATIONS, False)

    def initialize_cameras(self) -> None:
        """Complete lazy camera initialization and take the stopped-physics clock; call before :meth:`render`."""
        simulator = self.simulator
        try:
            simulator.render()
            self.settings.set_bool(_PLAY_SIMULATIONS, False)
            for camera in simulator.tiled_cameras.values():
                camera.update(simulator.sim_dt, force_recompute=True)
                _ = camera.data.output
            torch.cuda.synchronize(simulator.sim_device)
        finally:
            self.settings.set_bool(_PLAY_SIMULATIONS, False)
        self.clock = self._native_clock()

    def _native_clock(self) -> tuple[float, int]:
        context = self.simulator.sim
        return context.current_time, context.current_time_step_index

    def render(self) -> None:
        """Flush applied state, render it and refresh camera data; raise if physics moved."""
        simulator = self.simulator
        if self._native_clock() != self.clock:
            raise RuntimeError("Isaac physics advanced between rendered frames")
        self.settings.set_bool(_PLAY_SIMULATIONS, False)
        try:
            with simulator.state.mutation(DERIVED):
                simulator.write_state_updates()
                simulator.render()
                self.settings.set_bool(_PLAY_SIMULATIONS, False)
                # Isaac Lab recomputes timestamp-cached asset data only when its clock advances.
                simulator.scene.update(dt=simulator.sim_dt)
            if self._native_clock() != self.clock:
                raise RuntimeError("Rendering applied state advanced Isaac physics")
        finally:
            self.settings.set_bool(_PLAY_SIMULATIONS, False)
