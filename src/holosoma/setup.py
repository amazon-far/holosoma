from setuptools import setup

# Robot SDKs are published to PyPI (FAR forks). The import names are unchanged
# (`unitree_interface`, `booster_robotics_sdk`) — only the distribution names
# differ from the historical GitHub-release wheels.
UNITREE_VERSION = "0.1.4"
BOOSTER_VERSION = "0.1.0"

setup(
    extras_require={
        "unitree": [f"far-unitree-sdk=={UNITREE_VERSION}"],
        "booster": [f"far-booster-sdk=={BOOSTER_VERSION}"],
    },
    # Entry points are declared in pyproject.toml [project.entry-points.*]
)
