"""Local-visualization camera-egress plugin (cv2 window / mp4).

Imports cv2 + video utils at module top — loaded only when a ``CameraVizPluginConfig.get_cls``
fires (i.e. when the viz plugin is selected), so the plugins package stays cv2-free otherwise.
"""
