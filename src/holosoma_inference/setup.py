import platform
import sys

from setuptools import find_packages, setup

UNITREE_VERSION = "0.1.2"
UNITREE_REPO = "https://github.com/amazon-far/unitree_sdk2"
BOOSTER_VERSION = "0.1.0"
BOOSTER_REPO = "https://github.com/amazon-far/booster_robotics_sdk"


def get_platform_tag():
    """Get the correct platform tag for wheel installation."""
    machine = platform.machine()
    system = platform.system()

    if system == "Linux":
        if machine == "x86_64":
            return "linux_x86_64"
        if machine in ("aarch64", "arm64"):
            return "linux_aarch64"
    elif system == "Darwin":
        # Get macOS version for proper wheel tag
        mac_ver = platform.mac_ver()[0]  # Returns version like "14.1.0"
        major_version = mac_ver.split(".")[0] if mac_ver else "11"
        assert major_version == "26"
        return f"macosx_{major_version}_0_{machine}"
    # Fallback
    return "linux_x86_64"


platform_tag = get_platform_tag()
pyvers = f"cp{sys.version_info.major}{sys.version_info.minor}"
unitree_extras = []
unitree_url = f"{UNITREE_REPO}/releases/download/{UNITREE_VERSION}/unitree_sdk2-{UNITREE_VERSION}-{pyvers}-{pyvers}-{platform_tag}.whl"  # noqa: E501
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
    keywords="humanoid robotics inference policy onnx",
    include_package_data=True,
    package_data={
        "holosoma_inference": ["configs/**/*.yaml", "py.typed"],
    },
)
