"""Simulator plugins: runtime plugin classes selected on the CLI as ``plugin.<key>:<variant>``.

A plugin is any class constructed as ``cls(cfg, simulator)`` that registers hooks on
``simulator.hooks`` — there is no base class to inherit (see
:mod:`holosoma.config_types.plugin`). This package holds the in-tree camera-frame egress
plugins and their shared substrate:

- :class:`~holosoma.simulator.plugins.camera_consumer.CameraConsumerPlugin` — an optional shared
  base for plugins that consume rendered camera frames each control step (ROS2 image publish, viz
  window, mp4 recording); its ROS/cv2 impls live in ``ros2/`` and ``viz/`` and are imported only via
  a config's ``get_cls``, so importing this package pulls in no transport dependency.

The ``none`` no-op and the ROS2 example plugins (``clock_publish``/``gantry_control``/``odometry``)
live under ``simulator/shared`` instead.
"""

from holosoma.simulator.plugins.camera_consumer import (
    CameraConsumerPlugin,
    CameraIntrinsics,
    FramePacket,
)

__all__ = ["CameraConsumerPlugin", "CameraIntrinsics", "FramePacket"]
