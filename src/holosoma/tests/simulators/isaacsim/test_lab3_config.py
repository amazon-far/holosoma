"""Native constructors and compatibility checks; these do not launch a simulator."""

from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version

import pytest

from holosoma.config_values.simulator import isaacsim6_newton
from holosoma.simulator.isaacsim.backends.native_config import SOLVERS, physics_config, visualizer_configs

pytestmark = pytest.mark.isaacsim6

try:
    native_available = version("isaaclab").startswith("28.")
except PackageNotFoundError:
    native_available = False
pytestmark = [pytestmark, pytest.mark.skipif(not native_available, reason="Requires Isaac Lab 3 environment")]


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_selection_reaches_native_manager(solver):
    cfg = replace(isaacsim6_newton.config, isaacsim=replace(isaacsim6_newton.config.isaacsim, solver=solver))
    native = physics_config(cfg)
    assert type(native.solver_cfg).__name__ == SOLVERS[solver]
    assert native.class_type == native.solver_cfg.class_type


def test_native_options_and_constructor_validation():
    cfg = replace(
        isaacsim6_newton.config,
        isaacsim=replace(
            isaacsim6_newton.config.isaacsim,
            physics_options={"num_substeps": 3, "deterministic_mode": "run_to_run"},
            solver_options={"iterations": 23},
        ),
    )
    native = physics_config(cfg)
    assert native.num_substeps == 3
    assert native.solver_cfg.iterations == 23
    bad = replace(cfg, isaacsim=replace(cfg.isaacsim, physics_options={"deterministic_mode": "typo"}))
    with pytest.raises(ValueError, match="deterministic_mode"):
        physics_config(bad)


@pytest.mark.parametrize("options", [{"num_substepps": 3}, {"class_type": "anything"}, {"solver_cfg": {}}])
def test_invalid_native_fields_are_rejected(options):
    cfg = replace(isaacsim6_newton.config, isaacsim=replace(isaacsim6_newton.config.isaacsim, physics_options=options))
    with pytest.raises(ValueError, match="physics_options"):
        physics_config(cfg)


def test_web_visualizer_defaults_to_local_access():
    selection = replace(isaacsim6_newton.config.isaacsim, visualizers=["viser"])
    (native,) = visualizer_configs(selection, headless=False)
    assert native.bind_address == "127.0.0.1"
    assert native.share is False
    assert native.open_browser is False


@pytest.mark.parametrize("capture", ["sensor", "video"])
def test_launch_declares_rtx_requirements_with_newton_physics(monkeypatch, capture):
    from contextlib import nullcontext

    from isaaclab import app

    from holosoma.config_types.isaac import IsaacCameraRendererConfig
    from holosoma.config_types.run_sim import RunSimConfig
    from holosoma.config_types.sensor import CameraSensorConfig, IsaacSimCameraConfig, SensorMountConfig
    from holosoma.simulator.isaacsim.backends.lab3 import launch

    config = RunSimConfig(simulator=isaacsim6_newton)
    config = replace(config, training=replace(config.training, headless=True))
    if capture == "sensor":
        camera = CameraSensorConfig(
            mount=SensorMountConfig(target_kind="world"),
            isaacsim=IsaacSimCameraConfig(renderer=IsaacCameraRendererConfig(backend="isaacsim_rtx")),
        )
        config = replace(config, sensor={"camera": camera})
    else:
        simulator = replace(
            config.simulator,
            config=replace(
                config.simulator.config, isaacsim=replace(config.simulator.config.isaacsim, renderer="isaacsim_rtx")
            ),
        )
        config = replace(
            config, simulator=simulator, logger=replace(config.logger, video=replace(config.logger.video, enabled=True))
        )
    observed = []

    def inspect_launch(tree, args):
        scan, sources = app.validate_launch(tree, args)
        observed.append(scan.has_kit_camera)
        assert not scan.has_kit_physics
        assert sources
        return nullcontext()

    monkeypatch.setattr("sys.argv", ["test"])
    monkeypatch.setattr(app, "add_launcher_args", lambda _parser: None)
    monkeypatch.setattr(app, "launch_simulation", inspect_launch)
    session = launch(config, "cpu")
    session.close()
    assert observed == [True]


def test_legacy_capsule_dimensions_preserve_visuals_and_inertia():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from holosoma.simulator.isaacsim.backends.urdf import replace_collision_cylinders

    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/body")
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(2.0)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.2, 0.3))
    collision = UsdGeom.Cylinder.Define(stage, "/body/collision")
    collision.CreateHeightAttr(0.2)
    collision.CreateRadiusAttr(0.05)
    collision.CreateAxisAttr("Z")
    collision.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    UsdGeom.Cylinder.Define(stage, "/body/visual")

    replace_collision_cylinders(stage)

    capsule = UsdGeom.Capsule(stage.GetPrimAtPath("/body/collision"))
    assert capsule
    # Measured 5.1 behavior: height remains the cylinder's height; the caps
    # extend the collision bounds by one radius at each end.
    assert capsule.GetHeightAttr().Get() == pytest.approx(0.2)
    assert capsule.GetRadiusAttr().Get() == pytest.approx(0.05)
    assert tuple(capsule.GetExtentAttr().Get()[1]) == pytest.approx((0.05, 0.05, 0.15))
    assert tuple(capsule.GetLocalTransformation().ExtractTranslation()) == (1, 2, 3)
    assert stage.GetPrimAtPath("/body/visual").IsA(UsdGeom.Cylinder)
    assert mass.GetMassAttr().Get() == 2.0
    assert tuple(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx((0.1, 0.2, 0.3))
