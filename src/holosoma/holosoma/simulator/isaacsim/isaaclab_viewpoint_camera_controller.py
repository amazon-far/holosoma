# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import omni.kit.app
import omni.timeline
import torch

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ViewerCfg

    from holosoma.simulator.isaacsim.isaacsim import IsaacSim


class ViewportCameraController:
    """This class handles controlling the camera associated with a viewport in the simulator.

    It can be used to set the viewpoint camera to track different origin types:

    - **world**: the center of the world (static)
    - **env**: the center of an environment (static)
    - **asset_root**: the root of an asset in the scene (e.g. tracking a robot moving in the scene)

    On creation, the camera is set to track the origin type specified in the configuration.

    For the :attr:`asset_root` origin type, the camera is updated at each rendering step to track the asset's
    root position. For this, it registers a callback to the post update event stream from the simulation app.
    """

    def __init__(self, env: IsaacSim, cfg: ViewerCfg) -> None:
        """Initialize the ViewportCameraController.

        Args:
            env: The environment.
            cfg: The configuration for the viewport camera controller.

        Raises:
            ValueError: If origin type is configured to be "env" but :attr:`cfg.env_index` is out of bounds.
            ValueError: If origin type is configured to be "asset_root" but :attr:`cfg.asset_name` is unset.

        """
        # store inputs
        self._env = env
        self._cfg = copy.deepcopy(cfg)
        # cast viewer eye and look-at to numpy arrays
        self.default_cam_eye = np.array(self._cfg.eye)
        self.default_cam_lookat = np.array(self._cfg.lookat)

        # set the camera origins
        if self.cfg.origin_type == "env":
            # check that the env_index is within bounds
            self.set_view_env_index(self.cfg.env_index)
            # set the camera origin to the center of the environment
            self.update_view_to_env()
        elif self.cfg.origin_type == "asset_root":
            # note: we do not yet update camera for tracking an asset origin, as the asset may not yet be
            # in the scene when this is called. Instead, we subscribe to the post update event to update the camera
            # at each rendering step.
            if self.cfg.asset_name is None:
                raise ValueError(f"No asset name provided for viewer with origin type: '{self.cfg.origin_type}'.")
        else:
            # set the camera origin to the center of the world
            self.update_view_to_world()

        # subscribe to post update event so that camera view can be updated at each rendering step
        app_interface = omni.kit.app.get_app_interface()
        app_event_stream = app_interface.get_post_update_event_stream()
        self._viewport_camera_update_handle = app_event_stream.create_subscription_to_pop(
            lambda event, obj=weakref.proxy(self): obj._update_tracking_callback(event)  # noqa: B008
        )

    def __del__(self) -> None:
        """Unsubscribe from the callback."""
        # use hasattr to handle case where __init__ has not completed before __del__ is called
        if hasattr(self, "_viewport_camera_update_handle") and self._viewport_camera_update_handle is not None:
            self._viewport_camera_update_handle.unsubscribe()
            self._viewport_camera_update_handle = None

    """
    Properties
    """

    @property
    def cfg(self) -> ViewerCfg:
        """The configuration for the viewer."""
        return self._cfg

    """
    Public Functions
    """

    def set_view_env_index(self, env_index: int) -> None:
        """Sets the environment index for the camera view.

        Args:
            env_index: The index of the environment to set the camera view to.

        Raises:
            ValueError: If the environment index is out of bounds. It should be between 0 and num_envs - 1.
        """
        # check that the env_index is within bounds
        if env_index < 0 or env_index >= self._env.scene.num_envs:
            raise ValueError(
                f"Out of range value for attribute 'env_index': {env_index}."
                f" Expected a value between 0 and {self._env.scene.num_envs - 1} for the current environment."
            )
        # update the environment index
        self.cfg.env_index = env_index
        # update the camera view if the origin is set to env type (since, the camera view is static)
        # note: for assets, the camera view is updated at each rendering step
        if self.cfg.origin_type == "env":
            self.update_view_to_env()

    def update_view_to_world(self) -> None:
        """Updates the viewer's origin to the origin of the world which is (0, 0, 0)."""
        # set origin type to world
        self.cfg.origin_type = "world"
        # update the camera origins
        self.viewer_origin = torch.zeros(3)
        # update the camera view
        self.update_view_location()

    def update_view_to_env(self) -> None:
        """Updates the viewer's origin to the origin of the selected environment."""
        # set origin type to world
        self.cfg.origin_type = "env"
        # update the camera origins
        self.viewer_origin = self._env.scene.env_origins[self.cfg.env_index]
        # update the camera view
        self.update_view_location()

    def update_view_to_asset_root(self, asset_name: str) -> None:
        """Updates the viewer's origin based upon the root of an asset in the scene.

        Args:
            asset_name: The name of the asset in the scene. The name should match the name of the
                asset in the scene.

        Raises:
            ValueError: If the asset is not in the scene.
        """
        # check if the asset is in the scene
        if self.cfg.asset_name != asset_name:
            asset_entities = [*self._env.scene.rigid_objects.keys(), *self._env.scene.articulations.keys()]
            if asset_name not in asset_entities:
                raise ValueError(f"Asset '{asset_name}' is not in the scene. Available entities: {asset_entities}.")
        # update the asset name
        self.cfg.asset_name = asset_name
        # set origin type to asset_root
        self.cfg.origin_type = "asset_root"
        # update the camera origins
        self.viewer_origin = self._env.scene[self.cfg.asset_name].data.root_pos_w[self.cfg.env_index]
        # update the camera view
        self.update_view_location()

    def update_view_location(self, eye: Sequence[float] | None = None, lookat: Sequence[float] | None = None) -> None:
        """Updates the camera view pose based on the current viewer origin and the eye and lookat positions.

        Args:
            eye: The eye position of the camera. If None, the current eye position is used.
            lookat: The lookat position of the camera. If None, the current lookat position is used.
        """
        # store the camera view pose for later use
        if eye is not None:
            self.default_cam_eye = np.asarray(eye)
        if lookat is not None:
            self.default_cam_lookat = np.asarray(lookat)
        # set the camera locations
        viewer_origin = self.viewer_origin.detach().cpu().numpy()
        cam_eye = viewer_origin + self.default_cam_eye
        cam_target = viewer_origin + self.default_cam_lookat

        # set the camera view
        self._env.sim.set_camera_view(eye=cam_eye, target=cam_target)

    def capture_current_camera_offset(self) -> None:
        """Capture current camera position relative to asset origin.

        This method is called when toggling camera tracking ON to preserve
        the current camera angle/distance as the new tracking offset.
        """
        from loguru import logger
        import omni.kit.viewport.utility as vp_utils
        from pxr import Gf

        # Get the active viewport camera transform
        viewport_api = vp_utils.get_active_viewport()
        if viewport_api is None:
            logger.warning("No active viewport found, using default camera offset")
            return

        # Get camera position and target from viewport
        camera_path = viewport_api.camera_path
        if not camera_path:
            logger.warning("No camera path found, using default camera offset")
            return

        # Get the camera's transform
        from holosoma.simulator.isaacsim.prim_utils import get_current_stage

        stage = get_current_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)

        if not camera_prim.IsValid():
            logger.warning("Camera prim not valid, using default camera offset")
            return

        # Get the camera's world transform
        from pxr import UsdGeom

        xformable = UsdGeom.Xformable(camera_prim)
        world_transform = xformable.ComputeLocalToWorldTransform(0)

        # Extract position (translation) from transform matrix
        cam_pos_gf = world_transform.ExtractTranslation()
        cam_eye = np.array([cam_pos_gf[0], cam_pos_gf[1], cam_pos_gf[2]])

        # Calculate camera forward direction to get target
        # Camera looks down -Z in its local frame
        forward_local = Gf.Vec3d(0, 0, -1)
        forward_world = world_transform.TransformDir(forward_local)

        # Set target at a reasonable distance along viewing direction
        view_distance = 5.0  # Default viewing distance
        cam_target = cam_eye + np.array([forward_world[0], forward_world[1], forward_world[2]]) * view_distance

        # Calculate offset relative to current viewer_origin
        viewer_origin_np = self.viewer_origin.detach().cpu().numpy()
        self.default_cam_eye = cam_eye - viewer_origin_np
        self.default_cam_lookat = cam_target - viewer_origin_np

        logger.info(f"Captured camera offset: eye={self.default_cam_eye}, lookat={self.default_cam_lookat}")

    """
    Private Functions
    """

    def _update_tracking_callback(self, event: Any) -> None:
        """Updates the camera view at each rendering step."""
        # update the camera view if the origin is set to asset_root
        # in other cases, the camera view is static and does not need to be updated continuously
        if self.cfg.origin_type == "asset_root" and self.cfg.asset_name is not None:
            self.update_view_to_asset_root(self.cfg.asset_name)
