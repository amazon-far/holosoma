from setuptools import find_packages, setup

# Robot SDKs are published to PyPI (FAR forks). The import names are unchanged
# (`unitree_interface`, `booster_robotics_sdk`) — only the distribution names
# differ from the historical GitHub-release wheels. pip resolves the correct
# wheel for the running interpreter/platform from the PyPI index.
UNITREE_VERSION = "0.1.5"
BOOSTER_VERSION = "0.1.1"

unitree_extras = [f"far-unitree-sdk=={UNITREE_VERSION}"]
booster_extras = [f"far-booster-sdk=={BOOSTER_VERSION}"]


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
        "importlib_metadata>=4.6; python_version<'3.12'",
        "eval_type_backport; python_version<'3.10'",
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
            "unitree_mp = holosoma_inference.sdk.unitree.unitree_interface_mp:UnitreeInterfaceMP",
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
