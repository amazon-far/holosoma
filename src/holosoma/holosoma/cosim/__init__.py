"""Optional GPU co-simulation: share a simulator's state with another process on the same GPU.

- :mod:`.bridge` and :mod:`.cuda`: the lockstep exchange and its shared CUDA buffers.
- :mod:`.base`: bases for a source plugin and for a peer, which need not be a Holosoma simulator.
- :mod:`.render`: the ``cosim-render`` plugin and a Holosoma renderer for remote cameras.

Import the CUDA helpers only after the simulator provider has initialized. Scenes and process or container
deployment are the caller's.
"""
