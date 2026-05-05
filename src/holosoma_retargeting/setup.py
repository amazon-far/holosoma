from setuptools import find_packages, setup  # type: ignore[import-untyped]

setup(
    name="holosoma-retargeting",
    version="0.1.0",
    description="holosoma-retargeting: retargeting components for converting human motions to robot motions",
    author="Amazon FAR Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Needs to ping numpy to 2.3.5;
        # reason: later numpy version such as 2.4 will trigger
        # "TypeError: only 0-dimensional arrays can be converted to Python scalars"
        # in yourdf/urdf.py::1078 when converting float(q)
        "numpy==2.3.5",
        "torch",
        "tqdm",
        "scipy",
        "matplotlib",
        "trimesh",
        "smplx",
        "jinja2",
        "mujoco",
        "viser",
        "imageio[ffmpeg]",
        "robot_descriptions",
        "yourdfpy",
        "cvxpy",
        "libigl",
        "tyro",
    ],
)
