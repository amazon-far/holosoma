from glob import glob

from setuptools import find_packages, setup

package_name = "holosoma_service"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),  # noqa: PTH207
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Tomasz Lewicki",
    maintainer_email="jtomasle@amazon.com",
    description="ROS2 service nodes + launch for holosoma teleop.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_service_node = holosoma_service.policy_control.service_node:main",
            "retargeter_node = holosoma_service.retargetting.retargeter_node:main",
            "unitree_split_controller = holosoma_service.unitree_control.unitree_split_controller:_cli",
            "wasd_controller_node = holosoma_service.unitree_control.wasd_controller_node:main",
        ],
        # Retargeter implementations, selectable at launch via retargeter:=<name>.
        # Extensions register their own here (see FAR-pi holosoma_extensions).
        "holosoma.retargeter": [
            "g1-smpl = holosoma_service.retargetting.smpl_retargeter:G1SmplRetargeter",
        ],
    },
)
