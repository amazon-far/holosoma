from setuptools import find_packages, setup  # type: ignore[import-untyped]

setup(
    name="holosoma-retargeting",
    version="0.1.0",
    description="holosoma-retargeting: retargeting components for converting human motions to robot motions",
    author="Amazon FAR Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "scipy",
        "matplotlib",
        "trimesh",
        "smplx",
        "jinja2",
        "mujoco",
        "viser",
        "robot_descriptions",
        "yourdfpy",
        "cvxpy",
        "libigl",
        "tyro",
    ],
)
