"""Round trips over the co-simulation wire protocol, in one process over an ipc:// endpoint."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from holosoma.simulator.plugins.cosim.protocol import CoSimError, Replier, Requester


@pytest.fixture
def pair(tmp_path: Path) -> Iterator[tuple[Requester, Replier]]:
    endpoint = f"ipc://{tmp_path}/cosim"
    replier = Replier(endpoint)
    requester = Requester(endpoint)
    yield requester, replier
    requester.close()
    replier.close()


def test_header_and_arrays_round_trip(pair: tuple[Requester, Replier]) -> None:
    requester, replier = pair
    root = np.arange(2 * 7, dtype=np.float32).reshape(2, 7)
    objects = np.zeros((0, 2, 7), dtype=np.float32)
    requester.send("step", {"cams": ["head"], "rows": [1]}, {"root": root, "objects": objects})

    request = replier.recv(timeout_s=5)
    assert (request.type, request.header) == ("step", {"cams": ["head"], "rows": [1]})
    np.testing.assert_array_equal(request.arrays["root"], root)
    assert request.arrays["objects"].shape == (0, 2, 7)

    rgb = np.random.default_rng(0).integers(0, 255, size=(1, 3, 5, 3), dtype=np.uint8)
    depth = np.full((1, 3, 5, 1), np.inf, dtype=np.float32)
    replier.send("frames", arrays={"head/rgb": rgb, "head/depth": depth})

    reply = requester.recv(timeout_s=5)
    np.testing.assert_array_equal(reply.arrays["head/rgb"], rgb)
    np.testing.assert_array_equal(reply.arrays["head/depth"], depth)
    assert reply.arrays["head/rgb"].flags.writeable


def test_peer_error_and_version_mismatch_raise(pair: tuple[Requester, Replier]) -> None:
    requester, replier = pair
    requester.send("step")
    replier.recv(timeout_s=5)
    replier.send("error", {"msg": "camera 'head' missing"})
    with pytest.raises(CoSimError, match="camera 'head' missing"):
        requester.recv(timeout_s=5)

    requester.send("step")
    replier.recv(timeout_s=5)
    replier._socket.send_multipart([json.dumps({"v": 99, "type": "ok", "arrays": []}).encode()])
    with pytest.raises(CoSimError, match="version 99"):
        requester.recv(timeout_s=5)


def test_recv_times_out(pair: tuple[Requester, Replier]) -> None:
    _, replier = pair
    with pytest.raises(TimeoutError):
        replier.recv(timeout_s=0.05)
