"""Configuration values for data conversion."""

from __future__ import annotations

from holosoma_retargeting.config_types.data_conversion import DataConversionConfig


def get_default_data_conversion_config(
    input_file: str,
    robot: str = "g1",
    data_format: str = "smplh",
    object_name: str | None = None,
    input_fps: int = 30,
    output_fps: int = 50,
    line_range: tuple[int, int] | None = None,
    has_dynamic_object: bool = False,
    output_name: str | None = None,
    once: bool = False,
) -> DataConversionConfig:
    """Get default data conversion configuration.

    Args:
        input_file: Path to input motion file.
        robot: Robot model to use.
        data_format: Motion data format.
        object_name: Override object name.
        input_fps: FPS of the input motion.
        output_fps: FPS of the output motion.
        line_range: Line range (start, end) for loading data.
        has_dynamic_object: Whether the motion has a dynamic object.
        output_name: Name of the output motion npz file.
        once: Run the motion once and exit.

    Returns:
        DataConversionConfig: Default configuration instance.
    """
    return DataConversionConfig(
        input_file=input_file,
        robot=robot,
        data_format=data_format,
        object_name=object_name,
        input_fps=input_fps,
        output_fps=output_fps,
        line_range=line_range,
        has_dynamic_object=has_dynamic_object,
        output_name=output_name,
        once=once,
    )


__all__ = ["get_default_data_conversion_config"]
