import logging
import os
import shutil
import json
from plumbum import local
from pathlib import Path
from .model import MODELS, DEFAULT_MODEL

REPO_ROOT: Path = Path(os.getenv("TFLITE_BUILDER_ROOT") or Path(__file__).parent)
CLONES: Path = REPO_ROOT / "clones"
BUILDS: Path = REPO_ROOT / "builds"
SDKS: Path = REPO_ROOT / "sdks"
RUNNER_FILE: Path = REPO_ROOT / "runner/main.cpp"

BAZEL_TARGET: str = "//tensorflow/lite:libtensorflowlite.so"


def get_config(build_system: str, arch: str):
    config = REPO_ROOT / f"configs/{arch}.json"
    with open(config) as fd:
        metadata = json.load(fd)

    return metadata[build_system]


def clone_repository(branch_or_tag: str) -> None:
    """Clone TensorFlow repository for the specified branch/tag."""
    repo_url = "https://github.com/tensorflow/tensorflow.git"
    target_dir: Path = CLONES / branch_or_tag

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Cloning TensorFlow repository ({branch_or_tag}) into {target_dir}/")

    try:
        git = local["git"]
        git[
            "clone", "--depth", "1", "--branch", branch_or_tag, repo_url, target_dir
        ].run_fg()
        logging.info(f"Successfully cloned into {target_dir}/")
    except Exception as e:
        raise RuntimeError(f"Failed to clone repository: {e}") from e


def build_with_cmake(repo_dir: Path, build_target: str) -> None:
    """Build TensorFlow using CMake."""
    logging.info(f"Building TensorFlow with CMake (target: {build_target})")

    branch_or_tag = repo_dir.name
    build_dir = BUILDS / f"cmake-build-{branch_or_tag}"
    build_dir.mkdir(parents=True, exist_ok=True)

    metadata = get_config("cmake", "x86-64")
    defines = metadata["defines"]

    try:
        # Configure for x86_64 CPU-only build
        cmake = local["cmake"]
        cmake[
            "-S",
            repo_dir / "tensorflow/lite",
            "-B",
            build_dir,
            f"-DTENSORFLOW_SOURCE_DIR={repo_dir}",
            *defines,
            f"-DML_DTYPES_SOURCE_DIR={repo_dir / 'tensorflow/lite/tools/cmake/modules/ml_dtypes'}",
            f"-DCMAKE_INSTALL_PREFIX={build_dir / 'install'}",
        ].run_fg()

        cmake["--build", build_dir, "--target", build_target].run_fg()
        logging.info("CMake build completed successfully")
    except Exception as e:
        raise RuntimeError(f"CMake build failed: {e}") from e

    install_dst = SDKS / f"{branch_or_tag}--cmake"
    if not install_dst.exists():
        install_dst.mkdir(parents=True)

    # Install tensorflow headers

    # Install tensorflow library
    tflite_lib = build_dir / "libtensorflow-lite.so"
    lib = install_dst / "lib"
    if not lib.exists():
        lib.mkdir(parents=True)
    shutil.copy2(tflite_lib, lib)

    # Install flatbuffers headers
    flatbuffers_src = build_dir / "flatbuffers/include/flatbuffers"
    flatbuffers_dst = install_dst / "include/flatbuffers"
    flatbuffers_dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(flatbuffers_src), str(flatbuffers_dst), dirs_exist_ok=True)
    logging.info(f"Copied flatbuffers headers to {flatbuffers_dst}")


def build_with_bazel(repo_dir: Path, build_target: str) -> None:
    """Build TensorFlow using Bazel."""
    logging.info(f"Building TensorFlow with Bazel (target: {build_target})")

    # Set up bazel command
    bazel = local["bazel"]
    metadata = get_config("bazel", "x86-64")
    config = metadata.get("config", None)
    args = [
        "-c",
        "opt",
    ]
    arch = config or "x86-64"
    if config:
        args.append(f"--config={config}")

    branch_or_tag = repo_dir.name
    bazel_output_base = BUILDS / f"bazel-build-{branch_or_tag}"

    # Run bazel command
    try:
        bazel = local["bazel"]
        bazel[
            f"--output_base={bazel_output_base!s}", "build", BAZEL_TARGET, *args
        ].with_cwd(repo_dir).run_fg()
        logging.info("Bazel build completed successfully")

        # Stage the built artifacts into a single install folder
        install_dir = SDKS / f"{branch_or_tag}--{arch}--bazel"
        lib_dir = install_dir / "lib"
        include_dir = install_dir / "include"
        lib_dir.mkdir(parents=True, exist_ok=True)
        include_dir.mkdir(parents=True, exist_ok=True)

        # Copy the .so from bazel-bin
        args = []
        if config:
            args.append(f"--config={config}")

        bazel_bin = Path(
            bazel[f"--output_base={bazel_output_base}", "info", *args, "bazel-bin"]
            .with_cwd(repo_dir)()
            .strip()
        )
        logging.debug(f"bazel-bin: {bazel_bin}")
        lib_so = bazel_bin / "tensorflow/lite/libtensorflowlite.so"

        if not lib_so.exists():
            raise RuntimeError(f"Expected library not found: {lib_so}")

        shutil.copy2(lib_so, lib_dir)
        logging.info(f"Copied {lib_so.name} to {lib_dir}")

        # Copy TFLite headers
        src_include = repo_dir / "tensorflow" / "lite"
        if src_include.exists():
            for root, _dirs, files in os.walk(src_include):
                rel = Path(root).relative_to(src_include)
                dest_dir = include_dir / "tensorflow" / "lite" / rel
                dest_dir.mkdir(parents=True, exist_ok=True)
                for f in files:
                    if f.endswith(".h") or f.endswith(".hpp"):
                        shutil.copy2(os.path.join(root, f), str(dest_dir / f))

        # Copy tensorflow core headers (e.g. tensorflow/core/public/version.h)
        core_src = repo_dir / "tensorflow"
        if core_src.exists():
            for root, _dirs, files in os.walk(core_src):
                rel = Path(root).relative_to(core_src)
                dest_dir = include_dir / "tensorflow" / rel
                dest_dir.mkdir(parents=True, exist_ok=True)
                for f in files:
                    if f.endswith(".h") or f.endswith(".hpp"):
                        shutil.copy2(os.path.join(root, f), str(dest_dir / f))

        # Copy flatbuffers headers from bazel download cache
        flatbuffers = (
            BUILDS
            / f"bazel-build-{branch_or_tag}/external/flatbuffers/include/flatbuffers"
        )
        shutil.copytree(flatbuffers, include_dir / "flatbuffers")

        logging.info(f"✓ TensorFlow Lite staged into: {install_dir}")
        logging.info(f"  - Library: {lib_dir}")
        logging.info(f"  - Headers: {include_dir}")
    except Exception as e:
        raise RuntimeError(f"Bazel build failed: {e}") from e


def runner(install_path: Path) -> None:
    """
    Compile and run a C++ application linked against TensorFlow Lite.

    Args:
        install_path: Path to the TensorFlow Lite install directory
    """
    install_dir = Path(install_path).resolve()

    if not RUNNER_FILE.exists():
        raise RuntimeError(f"C++ source file not found: {RUNNER_FILE}")

    if not install_dir.exists():
        raise RuntimeError(f"Install directory not found: {install_dir}")

    lib_dir = install_dir / "lib"
    include_dir = install_dir / "include"

    if not lib_dir.exists() or not include_dir.exists():
        raise RuntimeError("Invalid install directory structure")

    # Determine output executable path
    output_exe = RUNNER_FILE.stem
    logging.info(f"Compiling {RUNNER_FILE.name}...")
    logging.info(f"  Include: {include_dir}")
    logging.info(f"  Library: {lib_dir}")
    logging.info(f"  Output: {output_exe}")

    try:
        # Compile with g++ linking against TensorFlow Lite
        gpp = local["g++"]
        gpp[
            "-std=c++17",
            f"-I{include_dir}",
            str(RUNNER_FILE),
            "-o",
            str(RUNNER_FILE.with_name(output_exe)),
            f"-L{lib_dir}",
            "-ltensorflow-lite"
            if "--cmake" in install_path.stem
            else "-ltensorflowlite",
            "-Wl,-rpath," + str(lib_dir),  # Set RPATH for runtime library discovery
        ].run_fg()
        logging.info(f"✓ Successfully compiled to: {output_exe}")

        # Run the executable
        logging.info(f"Running {output_exe}...")
        logging.info("=" * 60)
        execute = local[str(RUNNER_FILE.with_name(output_exe))]
        execute[MODELS / DEFAULT_MODEL].run_fg()
        logging.info("=" * 60)
        logging.info("✓ Program completed successfully")
    except Exception as e:
        raise RuntimeError(f"Compilation or execution failed: {e}") from e
