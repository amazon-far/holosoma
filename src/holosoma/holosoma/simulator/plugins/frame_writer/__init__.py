"""Frame-writer camera-egress plugin (image sequences / per-stream mp4s to disk).

Imports cv2 at module top — loaded only when ``FrameWriterPluginConfig.get_cls`` fires, so the
plugins package stays cv2-free otherwise.
"""

from holosoma.simulator.plugins.frame_writer.frame_writer import FrameWriterPlugin

__all__ = ["FrameWriterPlugin"]
