"""Writes depth_resize_golden.npz: torch's antialiased bicubic resize, which training uses.

Run with any Python that has torch: ``python gen_depth_resize_golden.py depth_resize_golden.npz``.
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

near, far = 0.3, 2.0
rng = np.random.default_rng(20261112)
noise = rng.uniform(near, far, (63, 112)).astype(np.float32)
ramp = np.tile(np.linspace(near, far, 112, dtype=np.float32), (63, 1))
step = np.full((63, 112), far, np.float32)
step[30:, 40:75] = 0.6  # an obstacle edge
up = rng.uniform(near, far, (40, 60)).astype(np.float32)  # upsampling as well
inputs = {"noise": noise, "ramp": ramp, "step": step, "up": up}
out = {}
for k, v in inputs.items():
    t = torch.from_numpy(v)[None, None]
    out[k] = F.interpolate(t, size=(58, 87), mode="bicubic", antialias=True, align_corners=False)[0, 0].numpy()
np.savez_compressed(
    sys.argv[1],
    **{f"in_{k}": v for k, v in inputs.items()},
    **{f"out_{k}": v for k, v in out.items()},
    torch_version=np.array(torch.__version__),
)
print(torch.__version__)
