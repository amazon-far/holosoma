"""Native editable installs need implementation source in the dependency stage."""

import ast
import importlib.util
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("filename", "install_marker", "expected_installs"),
    [
        ("docker/Dockerfile", "bash scripts/setup_all.sh", 1),
        ("docker/thor/Dockerfile", "RUN uv pip install -e", 2),
        ("src/holosoma_inference/docker/Dockerfile", "RUN uv pip install --system -e", 1),
    ],
)
def test_every_native_install_has_cmake_sources_in_its_stage(filename, install_marker, expected_installs):
    lines = (ROOT / filename).read_text().replace("\\\n", " ").splitlines()
    copied_native = False
    installs = 0
    for line in lines:
        if line.startswith("FROM "):
            copied_native = False
        if line.startswith("COPY "):
            words = shlex.split(line)
            copied_native |= "src/holosoma_inference/holosoma_cpp" in words[1:-1]
        if install_marker in line:
            installs += 1
            assert copied_native, f"{filename}: native install precedes its CMake source COPY"
    assert installs == expected_installs, "fixture must cover every editable native install"


def test_cpp_image_provisions_the_fixture_generator_before_configuring():
    cpp = ROOT / "src/holosoma_inference/holosoma_cpp"
    instructions = (cpp / "docker/Dockerfile.test").read_text().replace("\\\n", " ").splitlines()
    before_cmake = []
    for instruction in instructions:
        if instruction.startswith("RUN cmake "):
            break
        if instruction.startswith("RUN "):
            before_cmake.append(shlex.split(instruction[4:]))
    else:
        pytest.fail("C++ test image must configure the native tests")
    assert any("apt-get" in words and "install" in words and "python3" in words for words in before_cmake)
    # Derive dependencies from the executed generator rather than keeping a
    # second package list that can silently drift when the generator changes.
    tree = ast.parse((cpp / "tests/gen_live_stub_models.py").read_text())
    imports = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imports.update(node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    required = set()
    for name in imports:
        spec = importlib.util.find_spec(name)
        if spec is None or {"site-packages", "dist-packages"}.intersection(Path(spec.origin or "").parts):
            required.add(name)
    assert required, "fixture generator must exercise third-party dependencies"
    installed = {
        word.split("==")[0] for words in before_cmake if "pip" in words and "install" in words for word in words
    }
    assert required <= installed


@pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")
def test_install_copies_only_onnx_runtime_libraries_from_its_directory(tmp_path):
    # Run the shipped install rule against a system-like library directory.
    cmake_lists = (ROOT / "src/holosoma_inference/holosoma_cpp/CMakeLists.txt").read_text()
    rule = re.search(r'install\(DIRECTORY "\$\{ONNXRuntime_LIBRARY_DIR\}/".*?\)\n', cmake_lists, re.DOTALL)
    assert rule, "ONNX Runtime install rule not found"
    libdir = tmp_path / "usr-lib"
    libdir.mkdir()
    shipped = ["libonnxruntime.so", "libonnxruntime.so.1", "libonnxruntime_providers_shared.so"]
    for name in [*shipped, "libc.so.6", "libstdc++.so.6", "libcudart.so.12"]:
        (libdir / name).write_text(name)
    project = tmp_path / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(rule NONE)\n"
        f'set(ONNXRuntime_LIBRARY_DIR "{libdir}")\nset(CMAKE_INSTALL_LIBDIR lib)\n' + rule.group(0)
    )
    build, prefix = tmp_path / "build", tmp_path / "prefix"
    subprocess.run(["cmake", "-S", project, "-B", build], check=True, capture_output=True)
    subprocess.run(["cmake", "--install", build, "--prefix", prefix], check=True, capture_output=True)
    assert sorted(path.name for path in (prefix / "lib").iterdir()) == sorted(shipped)
