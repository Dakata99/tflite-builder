import logging
import os
import shutil
from plumbum import local
from pathlib import Path
from .model import MODELS, DEFAULT_MODEL

REPO_ROOT: Path = Path(os.getenv("TFLITE_BUILDER_ROOT"))
CLONES: Path = REPO_ROOT / "clones"
BUILDS: Path = REPO_ROOT / "builds"
SDKS: Path = REPO_ROOT / "sdks"
RUNNER_FILE: Path = REPO_ROOT / "runner/main.cpp"

def clone_repository(branch_or_tag: str) -> str:
    """Clone TensorFlow repository for the specified branch/tag."""
    repo_url = "https://github.com/tensorflow/tensorflow.git"
    target_dir: Path = CLONES / branch_or_tag

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Cloning TensorFlow repository ({branch_or_tag}) into {target_dir}/")

    try:
        git = local["git"]
        git["clone", "--depth", "1", "--branch", branch_or_tag, repo_url, target_dir].run_fg()
        logging.info(f"Successfully cloned into {target_dir}/")
    except Exception as e:
        raise RuntimeError(f"Failed to clone repository: {e}") from e


def build_with_cmake(repo_dir: Path, build_target: str) -> None:
    """Build TensorFlow using CMake."""
    logging.info(f"Building TensorFlow with CMake (target: {build_target})")

    branch_or_tag = repo_dir.name
    build_dir = repo_dir / f"cmake-build-{branch_or_tag}"
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Configure for x86_64 CPU-only build
        cmake = local["cmake"]
        cmake[
            '-S', repo_dir / "tensorflow/lite",
            '-B', build_dir,
            f"-DTENSORFLOW_SOURCE_DIR={repo_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_SYSTEM_PROCESSOR=x86_64",
            "-DTFLITE_ENABLE_GPU=OFF",
            "-DTFLITE_ENABLE_NNAPI=OFF",
            "-DTFLITE_ENABLE_METAL=OFF",
            "-DTFLITE_ENABLE_FLEX=OFF",
            "-DTFLITE_ENABLE_XNNPACK=ON",
            "-DBUILD_SHARED_LIBS=ON",
            "-DTFLITE_ENABLE_INSTALL=OFF",
            f"-DML_DTYPES_SOURCE_DIR={repo_dir / 'tensorflow/lite/tools/cmake/modules/ml_dtypes'}",
            f"-DCMAKE_INSTALL_PREFIX={build_dir / 'install'}",
        ].run_fg()

        cmake["--build", build_dir, "--target", build_target].run_fg()
        logging.info("CMake build completed successfully")
    except Exception as e:
        raise RuntimeError(f"CMake build failed: {e}") from e


def build_with_bazel(repo_dir: Path, build_target: str) -> None:
    """Build TensorFlow using Bazel."""
    logging.info(f"Building TensorFlow with Bazel (target: {build_target})")

    branch_or_tag = repo_dir.name
    bazel_output_base = repo_dir / f"bazel-build-{branch_or_tag}"

    try:
        bazel = local["bazel"]
        bazel[
            f"--output_base={bazel_output_base!s}",
            "build",
            '-c', 'opt',
            build_target
        ].with_cwd(repo_dir).run_fg()
        logging.info("Bazel build completed successfully")

        # Stage the built artifacts into a single install folder
        install_dir = repo_dir / f"bazel-install-{branch_or_tag}"
        lib_dir = install_dir / "lib"
        include_dir = install_dir / "include"
        lib_dir.mkdir(parents=True, exist_ok=True)
        include_dir.mkdir(parents=True, exist_ok=True)

        # Copy the .so from bazel-bin
        bazel_bin = (
            bazel_output_base
            / "execroot"
            / "org_tensorflow"
            / "bazel-out"
            / "k8-opt"
            / "bin"
            / "tensorflow"
            / "lite"
        )
        lib_so = bazel_bin / "libtensorflowlite.so"
        if lib_so.exists():
            shutil.copy2(str(lib_so), str(lib_dir))
            logging.info(f"Copied {lib_so.name} to {lib_dir}")
        else:
            logging.warning(f"Expected library not found: {lib_so}")

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
        flatbuffers_search = repo_dir / f"bazel-build-{branch_or_tag}" / "_deps"
        if flatbuffers_search.exists():
            for flatbuffers_src in flatbuffers_search.glob("flatbuffers-src/include"):
                if flatbuffers_src.exists():
                    for root, _dirs, files in os.walk(flatbuffers_src):
                        rel = Path(root).relative_to(flatbuffers_src)
                        dest_dir = include_dir / rel
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        for f in files:
                            if f.endswith(".h") or f.endswith(".hpp"):
                                shutil.copy2(os.path.join(root, f), str(dest_dir / f))
                    logging.info("Copied flatbuffers headers")
                    break

        logging.info(f"✓ TensorFlow Lite staged into: {install_dir}")
        logging.info(f"  - Library: {lib_dir}")
        logging.info(f"  - Headers: {include_dir}")
    except Exception as e:
        logging.error(f"Bazel build failed: {e}")
        raise RuntimeError(f"Bazel build failed: {e}") from e


def install_artifacts(source_type: str, branch_or_tag: str, install_path: str) -> None:
    """
    Install staged artifacts to a custom location.

    Args:
        source_type: "cmake" or "bazel" to specify source build system
        branch_or_tag: Branch/tag name (e.g., v2.16.1)
        install_path: Destination directory to install to
    """
    logging.info(f"Installing TensorFlow Lite artifacts (source: {source_type}, tag: {branch_or_tag})")

    # Determine source directory based on build system
    if source_type == "cmake":
        src_dir = CLONES / branch_or_tag / f"cmake-build-{branch_or_tag}"
        build_dir = CLONES / branch_or_tag / f"cmake-build-{branch_or_tag}"
    elif source_type == "bazel":
        src_dir = CLONES / branch_or_tag / f"bazel-install-{branch_or_tag}"
        build_dir = CLONES / branch_or_tag / f"bazel-build-{branch_or_tag}"
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    if not src_dir.exists():
        raise RuntimeError(f"Source install directory not found: {src_dir}")

    # Create destination and copy artifacts
    install_dst = install_path or BUILDS / f"{branch_or_tag}--{source_type}"
    include_dst = install_dst / "include"
    install_dst.mkdir(parents=True, exist_ok=True)

    if source_type == "cmake":
        tflite_lib = src_dir / "libtensorflow-lite.so"
    else:
        tflite_lib = src_dir / "lib/libtensorflowlite.so"
    include_src = src_dir / "include"

    if not tflite_lib.exists():
        raise RuntimeError(f"TensorFlow Lite library not found: {tflite_lib}")

    lib_dst = install_dst / "lib"
    lib_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(tflite_lib), str(lib_dst))
    logging.info(f"Copied {tflite_lib.name} to {lib_dst}")

    if include_src.exists():
        include_dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(include_src), str(include_dst), dirs_exist_ok=True)
        logging.info(f"Copied headers to {include_dst}")

    if build_dir:
        # Copy flatbuffers headers if they exist
        if source_type == "cmake":
            flatbuffers_src = build_dir / "flatbuffers/include/flatbuffers"
        else:
            flatbuffers_src = build_dir / "external/flatbuffers/include/flatbuffers"
        if flatbuffers_src.exists():
            flatbuffers_dst = install_dst / "include/flatbuffers"
            flatbuffers_dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(flatbuffers_src), str(flatbuffers_dst), dirs_exist_ok=True)
            logging.info(f"Copied flatbuffers headers to {flatbuffers_dst}")

    logging.info(f"✓ Successfully installed to: {install_dst}")
    logging.info(f"  - Library: {lib_dst}")
    logging.info(f"  - Headers: {include_dst}")


def runner(build_system: str, install_path: str) -> None:
    """
    Compile and run a C++ application linked against TensorFlow Lite.

    Args:
        build_system: Build system to use (cmake or bazel)
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
            "-ltensorflow-lite" if build_system == "cmake" else "-ltensorflowlite",
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
