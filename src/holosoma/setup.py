import platform
import sys

from setuptools import setup

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


pyvers = f"cp{sys.version_info.major}{sys.version_info.minor}"
platform_str = get_platform_tag()

unitree_url = f"{UNITREE_REPO}/releases/download/{UNITREE_VERSION}/unitree_sdk2-{UNITREE_VERSION}-{pyvers}-{pyvers}-{platform_str}.whl"  # noqa: E501
booster_url = f"{BOOSTER_REPO}/releases/download/{BOOSTER_VERSION}/booster_robotics_sdk-{BOOSTER_VERSION}-{pyvers}-{pyvers}-{platform_str}.whl"  # noqa: E501

setup(
    extras_require={
        "unitree": [f"unitree_sdk2 @ {unitree_url}"],
        "booster": [f"booster_robotics_sdk @ {booster_url}"],
    },
)
