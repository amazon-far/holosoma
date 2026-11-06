"""Regression checks for the Lab 3 boundary; no viewer or simulator launch."""

import xml.etree.ElementTree as ET
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest

from holosoma.config_values.simulator import isaacsim6_newton, isaacsim6_ovphysx
from holosoma.simulator.isaacsim.backends import _active
from holosoma.simulator.isaacsim.backends.lab2 import Lab2Backend
from holosoma.simulator.isaacsim.backends.lab3 import Lab3Backend

try:
    native_available = version("isaaclab").startswith("28.")
except PackageNotFoundError:
    native_available = False
pytestmark = [
    pytest.mark.isaacsim6,
    pytest.mark.skipif(not native_available, reason="Requires the pinned Isaac Lab 3 environment"),
]


@pytest.fixture
def native_backend():
    backend = Lab3Backend(isaacsim6_newton.config.isaacsim)
    token = _active.set(backend)
    try:
        yield backend
    finally:
        _active.reset(token)


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("angle", [0.0, 90.0])
def test_custom_usd_rotation_matches_physical_axes(tmp_path, monkeypatch, native_backend, modern, angle):
    """Assert the resulting transform, independently of quaternion storage order."""
    import math

    from pxr import Gf, Usd, UsdGeom

    from holosoma.simulator.isaacsim.spawners import from_files

    source = Usd.Stage.CreateNew(str(tmp_path / "asymmetric.usda"))
    root = UsdGeom.Xform.Define(source, "/asset")
    source.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cube.Define(source, "/asset/box").AddScaleOp().Set(Gf.Vec3f(2, 1, 0.5))
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(from_files, "stage_utils", SimpleNamespace(get_current_stage=lambda: stage))
    backend = native_backend if modern else Lab2Backend()
    token = _active.set(backend)
    half_angle = math.radians(angle) / 2
    try:
        prim = from_files.create_prim(
            "/object",
            source.GetRootLayer().identifier,
            orientation=backend.config_quaternion((math.cos(half_angle), 0, 0, math.sin(half_angle))),
        )
    finally:
        _active.reset(token)
    transform = UsdGeom.Xformable(prim).GetLocalTransformation()
    expected = (1, 0, 0) if angle == 0 else (0, 1, 0)
    assert tuple(transform.TransformDir(Gf.Vec3d(1, 0, 0))) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("preserve_by_name", [False, True])
def test_selective_merge_preserves_mass_inertia_and_required_frames(tmp_path, preserve_by_name):
    from holosoma.simulator.isaacsim.backends.urdf import merge_fixed_joints

    source = tmp_path / "original.urdf"
    destination = tmp_path / "merged.urdf"
    source.write_text(
        f"""<robot name="mass_probe">
        <link name="base"><inertial><mass value="2"/>
        <inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/></inertial></link>
        <link name="payload"><inertial><mass value="1"/>
        <inertia ixx="0.1" iyy="0.2" izz="0.3" ixy="0" ixz="0" iyz="0"/></inertial>
        <visual><geometry><mesh filename="meshes/probe.stl"/></geometry></visual></link>
        <joint name="payload_fixed" type="fixed"><parent link="base"/><child link="payload"/>
        <origin xyz="2 0 0" rpy="0 0 1.5707963267948966"/></joint>
        <link name="sensor_frame"/>
        <joint name="sensor_fixed" type="fixed"><parent link="base"/><child link="sensor_frame"/>
        <origin xyz="0 0 0.25"/></joint>
        <link name="foot"><inertial><mass value="0.5"/>
        <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
        <joint name="foot_fixed" type="fixed" dont_collapse="{str(not preserve_by_name).lower()}">
        <parent link="sensor_frame"/><child link="foot"/><origin xyz="0 0 0.75"/></joint>
        </robot>"""
    )
    original = source.read_bytes()
    merge_fixed_joints(str(source), str(destination), ("foot",) if preserve_by_name else ())
    assert source.read_bytes() == original
    robot = ET.parse(destination).getroot()  # noqa: S314 - generated local fixture
    assert {link.get("name") for link in robot.findall("link")} == {"base", "foot"}
    (joint,) = robot.findall("joint")
    assert joint.get("type") == "fixed"
    assert joint.find("parent").get("link") == "base"
    assert [float(x) for x in joint.find("origin").get("xyz").split()] == pytest.approx([0, 0, 1])
    inertial = robot.find("link[@name='base']/inertial")
    assert float(inertial.find("mass").get("value")) == 3
    assert [float(x) for x in inertial.find("origin").get("xyz").split()] == pytest.approx([2 / 3, 0, 0])
    # Rz(90 deg) rotates the payload's inertia axes; parallel-axis shift is 8/3.
    inertia = inertial.find("inertia")
    assert [float(inertia.get(axis)) for axis in ("ixx", "iyy", "izz")] == pytest.approx(
        [1.2, 2.1 + 8 / 3, 3.3 + 8 / 3]
    )
    assert [float(inertia.get(axis)) for axis in ("ixy", "ixz", "iyz")] == pytest.approx([0, 0, 0], abs=1e-12)
    assert robot.find(".//mesh").get("filename") == str(tmp_path / "meshes/probe.stl")


def test_native_robot_config_keeps_merge_enabled_and_named_bodies(native_backend):
    from isaaclab.sim import UrdfFileCfg

    cfg = native_backend.urdf_spawn_config(
        UrdfFileCfg(asset_path="/tmp/robot.urdf", fix_base=False, merge_fixed_joints=True),
        preserve_fixed_bodies=("foot",),
    )
    assert cfg.merge_fixed_joints is True
    assert cfg.preserve_fixed_bodies == ("foot",)


def test_explicit_renderer_options_override_legacy_quality_preset():
    from holosoma.config_types.isaac import IsaacSimRenderConfig
    from holosoma.simulator.isaacsim.backends.native_config import renderer_config

    config = IsaacSimRenderConfig(
        runtime="6.x",
        rendering_mode="quality",
        renderer_options={"global_settings": {"enable_reflections": False}},
    )
    settings = renderer_config(config).global_settings
    assert settings.enable_reflections is False
    assert settings.enable_translucency is True
    assert settings.dlss_mode == 2
    assert settings.carb_settings["/rtx/reflections/denoiser/enabled"] is True


def test_native_recording_selects_configured_environment():
    import numpy as np
    import torch

    from holosoma.simulator.isaacsim.backends.recording import NativeVideoRecorder

    pixels = torch.zeros((2, 4, 6, 3), dtype=torch.uint8)
    pixels[0, ..., 0] = 255
    pixels[1, ..., 1] = 255
    positioned = []
    camera = SimpleNamespace(
        data=SimpleNamespace(output={"rgb": pixels}),
        set_world_poses_from_view=lambda _eyes, _targets, env_ids: positioned.extend(env_ids),
        update=lambda _dt, **_kwargs: None,
    )
    recorder = object.__new__(NativeVideoRecorder)
    recorder.config = SimpleNamespace(record_env_id=1)
    recorder.camera = camera
    recorder.device = "cpu"
    frames = []
    recorder._get_camera_parameters = lambda: SimpleNamespace(position=(0, 0, 1), target=(0, 0, 0))
    recorder._apply_command_overlay = lambda frame: frame
    recorder._add_frame = frames.append
    recorder._capture_frame_impl()
    assert positioned == [1]
    np.testing.assert_array_equal(frames[0], pixels[1].numpy())


def test_relocated_urdf_resolves_package_meshes_from_original_directory(tmp_path):
    from holosoma.simulator.isaacsim.backends.urdf import merge_fixed_joints

    mesh = tmp_path / "meshes" / "box.stl"
    mesh.parent.mkdir()
    mesh.touch()
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="box"><link name="base"><visual><geometry>'
        '<mesh filename="package://robot/meshes/box.stl"/>'
        "</geometry></visual></link></robot>"
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    destination = scratch / "robot.urdf"
    merge_fixed_joints(str(source), str(destination))
    root = ET.parse(destination).getroot()  # noqa: S314 - generated local fixture
    assert root.find(".//mesh").get("filename") == str(mesh)


@pytest.mark.parametrize("physics", ["newton", "ovphysx"])
def test_standalone_debug_draw_does_not_import_kit(monkeypatch, physics):
    import builtins

    monkeypatch.setattr("isaaclab.utils.version.has_kit", lambda: False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("isaacsim.util.debug_draw")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    preset = isaacsim6_newton if physics == "newton" else isaacsim6_ovphysx
    assert Lab3Backend(preset.config.isaacsim).acquire_debug_draw() is None


@pytest.mark.parametrize("override", [None, "newton_renderer", "isaacsim_rtx", "ovrtx"])
def test_fisheye_renderer_capability(native_backend, override):
    from holosoma.config_types.isaac import IsaacCameraRendererConfig
    from holosoma.config_types.sensor import CameraSensorConfig, IsaacSimCameraConfig, SensorMountConfig

    camera = CameraSensorConfig(
        mount=SensorMountConfig(target_kind="world"),
        isaacsim=IsaacSimCameraConfig(
            projection_type="fisheyePolynomial",
            renderer=IsaacCameraRendererConfig(backend=override),
        ),
    )
    if override in (None, "newton_renderer"):
        with pytest.raises(ValueError, match="F-theta"):
            native_backend.camera_options(camera)
    else:
        assert native_backend.camera_options(camera)["renderer_cfg"] is not None


def test_newton_spawn_material_rejects_independent_friction(native_backend):
    from holosoma.config_types.scene import IsaacSimPhysicsConfig, PhysicsConfig
    from holosoma.simulator.isaacsim.object_spawner import physics_material_cfg

    with pytest.raises(ValueError, match="one friction coefficient"):
        physics_material_cfg(PhysicsConfig(isaacsim=IsaacSimPhysicsConfig(static_friction=0.9, dynamic_friction=0.1)))
    material = physics_material_cfg(
        PhysicsConfig(isaacsim=IsaacSimPhysicsConfig(static_friction=0.4, dynamic_friction=0.4))
    )
    assert material.static_friction == material.dynamic_friction == 0.4
    # The version adapter must not impose Newton's limitation on PhysX.
    Lab2Backend().validate_friction(0.9, 0.1)
    Lab3Backend(isaacsim6_ovphysx.config.isaacsim).validate_friction(0.9, 0.1)


def test_newton_body_damping_does_not_silently_write_unused_usd():
    import torch

    from holosoma.simulator.isaacsim.backends.properties import NativeProperties

    properties = NativeProperties(SimpleNamespace(), "newton")
    with pytest.raises(NotImplementedError, match="per-body linear/angular damping"):
        properties.set_body_damping("linear", torch.tensor([0.5]), torch.tensor([0]))


@pytest.mark.parametrize("mesh_type", ["plane", "trimesh", "load_obj"])
def test_newton_terrain_rejects_independent_friction(native_backend, mesh_type):
    from holosoma.simulator.isaacsim.scene_fixtures import build_terrain_fixtures

    terrain = SimpleNamespace(mesh_type=mesh_type, static_friction=0.9, dynamic_friction=0.1)
    manager = SimpleNamespace(get_state=lambda _name: terrain)
    with pytest.raises(ValueError, match="one friction coefficient"):
        build_terrain_fixtures(manager, None, "base")


def test_ovphysx_rejects_kit_settings_before_launch(monkeypatch):
    from holosoma.config_types.run_sim import RunSimConfig
    from holosoma.simulator.isaacsim.backends.lab3 import launch

    config = RunSimConfig(simulator=isaacsim6_ovphysx)
    config = replace(
        config,
        training=replace(config.training, headless=True),
        simulator=replace(
            config.simulator,
            config=replace(
                config.simulator.config,
                isaacsim=replace(config.simulator.config.isaacsim, rendering_mode="quality"),
            ),
        ),
    )
    monkeypatch.setattr("sys.argv", ["test"])
    monkeypatch.setattr("isaaclab.app.add_launcher_args", lambda _parser: None)
    monkeypatch.setattr("isaaclab.app.launch_simulation", lambda *_args: pytest.fail("Must fail before launch"))
    with pytest.raises(ValueError, match="(?i)(kit|ovphysx)"):
        launch(config, "cpu")
