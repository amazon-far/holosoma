"""Optional local visualizers for camera frames and LiDAR point clouds.

Camera visualization imports cv2 only when ``CameraVizPluginConfig.get_cls`` is selected; LiDAR
visualization imports Matplotlib only when ``LidarVizPluginConfig.get_cls`` is selected. Importing
the plugins package itself keeps both optional visualization dependencies unloaded.
"""
