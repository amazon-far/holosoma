import platform

from setuptools import find_packages, setup

UNITREE_VERSION = "0.1.1"
UNITREE_REPO = "https://github.com/amazon-far/unitree_sdk2"
BOOSTER_VERSION = "0.1.0"
BOOSTER_REPO = "https://github.com/amazon-far/booster_robotics_sdk"

PLATFORM_MAP = {
    "x86_64": "linux_x86_64",
    "aarch64": "linux_aarch64",
}

platform_tag = PLATFORM_MAP.get(platform.machine(), "linux_x86_64")

unitree_extras = []
unitree_url = (
    f"{UNITREE_REPO}/releases/download/{UNITREE_VERSION}/unitree_sdk2-{UNITREE_VERSION}-cp310-cp310-{platform_tag}.whl"
)
unitree_extras.append(f"unitree_sdk2 @ {unitree_url}")

booster_extras = []
booster_url = f"{BOOSTER_REPO}/releases/download/{BOOSTER_VERSION}/booster_robotics_sdk-{BOOSTER_VERSION}-cp310-cp310-{platform_tag}.whl"  # noqa: E501
booster_extras.append(f"booster_robotics_sdk @ {booster_url}")


setup(
    name="holosoma-inference",
    version="0.1.0",
    description="holosoma-inference: inference components for humanoid robot policies",
    long_description="",
    long_description_content_type="text/markdown",
    author="Amazon FAR Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pydantic",
        "loguru",
        "netifaces",
        "onnx",
        "onnxruntime",
        "scipy",
        "sshkeyboard",
        "termcolor",
        "pyyaml",
        "tyro>=0.10.0a4",
        "wandb",
        "zmq",
        "defusedxml",
        "evdev",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "unitree": unitree_extras,
        "booster": booster_extras,
    },
    entry_points={
        "holosoma.sdk": [
            "unitree = holosoma_inference.sdk.unitree.unitree_interface:UnitreeInterface",
            "booster = holosoma_inference.sdk.booster.booster_interface:BoosterInterface",
        ],
        "holosoma.config.robot": [
            "g1-29dof = holosoma_inference.config.config_values.robot:g1_29dof",
            "t1-29dof = holosoma_inference.config.config_values.robot:t1_29dof",
        ],
        "holosoma.config.inference": [
            "g1-29dof-loco = holosoma_inference.config.config_values.inference:g1_29dof_loco",
            "t1-29dof-loco = holosoma_inference.config.config_values.inference:t1_29dof_loco",
            "g1-29dof-wbt = holosoma_inference.config.config_values.inference:g1_29dof_wbt",
        ],
    },
    keywords="humanoid robotics inference policy onnx",
    include_package_data=True,
    package_data={
        "holosoma_inference": ["configs/**/*.yaml", "py.typed"],
    },
)
