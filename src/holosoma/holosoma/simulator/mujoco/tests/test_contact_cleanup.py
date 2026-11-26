"""Contact query teardown with the simulator and readers still retained."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from weakref import ref

import numpy as np
import pytest
from typing_extensions import Self

mujoco = pytest.importorskip("mujoco")
wp = pytest.importorskip("warp")

from holosoma.config_types.sensor import ContactSensorConfig  # noqa: E402
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase  # noqa: E402
from holosoma.simulator.mujoco import contact_setup, contact_warp  # noqa: E402
from holosoma.simulator.mujoco.mujoco import MuJoCo  # noqa: E402
from holosoma.simulator.shared.sensor_manager import SensorManager  # noqa: E402

pytestmark = pytest.mark.no_sim


def test_close_releases_retained_readers_before_backend_close(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Backend:
        num_envs = 1
        mjw_device = "cpu"
        wp_stream = None
        _contact_ids = np.zeros(1)
        _contact_wrenches = None
        mjw_data = SimpleNamespace(nacon=None, contact=SimpleNamespace(worldid=None, geom=None, frame=None))

        def close(self) -> None:
            events.append("backend")

    class Graph:
        pass

    class Capture:
        def __init__(self, **_kwargs: Any) -> None:
            self.graph = Graph()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    # Replace only the CUDA capture boundary. Arrays and their Torch alias are real CPU allocations.
    monkeypatch.setattr(contact_setup, "WarpBackend", Backend)
    monkeypatch.setattr(wp, "ScopedStream", lambda _stream: nullcontext())
    monkeypatch.setattr(wp, "ScopedCapture", Capture)
    monkeypatch.setattr(wp, "load_module", lambda **_kwargs: None)
    monkeypatch.setattr(wp, "launch", lambda *_args, **_kwargs: None)

    sim = cast("Any", object.__new__(MuJoCo))
    sim._closed = False
    sim.hooks = HookRegistry()
    sim.backend = Backend()
    sim.hooks.add(Phase.CLOSE, sim._close_backend, name="mujoco.backend.close")
    sim.root_model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="robot_body"><freejoint/>'
        '<geom name="robot_geom" type="sphere" size=".1"/></body></worldbody></mujoco>'
    )
    sim.robot_config = SimpleNamespace(body_names=["robot_body"])
    sim.body_ids = [1]
    sim.sensor_config = {"whole": ContactSensorConfig()}
    sim.contact_target_map = contact_setup.ContactTargetMap(geom_names={"robot": ["robot_geom"]})
    sim.sensor_manager = SensorManager("cpu", 50.0, physics_hz=200.0)
    with wp.ScopedDevice("cpu"):
        contact_setup.register_contacts(sim)
    readers = [
        cast("contact_warp.WarpContactReader", record.capture) for record in sim.sensor_manager._contacts.values()
    ]
    resources = [
        ref(getattr(reader, attribute))
        for reader in readers
        for attribute in ("graph", "sources", "counterparts", "output", "tensor")
    ]

    def synchronize(device: str) -> None:
        assert device == "cpu"
        for reader in readers:
            if reader.backend is not None:
                assert all(
                    getattr(reader, attribute) is not None
                    for attribute in ("graph", "sources", "counterparts", "output", "tensor")
                )
        events.append("sync")

    monkeypatch.setattr(contact_warp.wp, "synchronize_device", synchronize)
    sim.close()
    assert all(resource() is None for resource in resources)
    assert all(reader.backend is None for reader in readers)
    assert events == ["sync", "sync", "backend"]
    sim.close()
    for reader in readers:
        reader.close()
        assert reader.backend is None
        with pytest.raises(RuntimeError, match="contact reader is closed"):
            reader()

    assert sim.sensor_manager._contacts["contact_sensor"].capture is readers[0]
    assert sim.sensor_manager._contacts["whole"].capture is readers[1]
    assert events == ["sync", "sync", "backend"]
