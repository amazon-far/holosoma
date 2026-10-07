"""Kinematic motion-playback plugin (clip-driven robot/object state for rendering-only replay).

Imported only via ``MotionPlaybackPluginConfig.get_cls``.
"""

from holosoma.simulator.plugins.playback.playback import MotionPlaybackPlugin, PlaybackClip

__all__ = ["MotionPlaybackPlugin", "PlaybackClip"]
