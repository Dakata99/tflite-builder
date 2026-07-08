import argparse
import logging
from pathlib import Path

import argcomplete

from .core import (
    CLONES,
    build_with_bazel,
    build_with_cmake,
    clone_repository,
    install_artifacts,
    runner,
)

CMAKE_TARGET: str = "tensorflow-lite"
BAZEL_TARGET: str = "//tensorflow/lite:libtensorflowlite.so"

# Set logging config
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")


def get_clones() -> list[str]:
    """Get a list of cloned TensorFlow repositories."""
    return [d.name for d in CLONES.iterdir() if d.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tf", description="Clone and build TensorFlow"
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Clone subcommand
    clone_parser = subparsers.add_parser("clone", help="Clone TensorFlow repository")
    clone_parser.add_argument(
        "-b",
        "--branch-tag",
        type=str,
        required=True,
        help="Git branch or tag to clone (e.g., v2.16.1, master)",
    )

    # Build subcommand
    build_parser = subparsers.add_parser("build", help="Build TensorFlow")
    build_parser.add_argument(
        "-b",
        "--branch-tag",
        type=str,
        required=True,
        help="Repository directory name (branch/tag)",
    ).completer = get_clones  # type: ignore[attr-defined]

    # Build system subcommands
    build_systems = build_parser.add_mutually_exclusive_group(required=True)

    # CMake subcommand (target is hardcoded to the TFLite C++ library)
    _ = build_systems.add_argument(
        "--cmake",
        action="store_true",
        help="Build with CMake (builds TFLite C++ library)",
    )

    # Bazel subcommand (target is hardcoded to the TFLite C++ API)
    _ = build_systems.add_argument(
        "--bazel", action="store_true", help="Build with Bazel (builds TFLite C++ API)"
    )

    # Install subcommand
    install_parser = subparsers.add_parser(
        "install", help="Install staged artifacts to a custom location"
    )
    # TODO: make get_builds?
    install_parser.add_argument(
        "-b",
        "--branch-tag",
        type=str,
        required=True,
        help="Repository tag/branch name (e.g., v2.16.1)",
    ).completer = get_clones  # type: ignore[attr-defined]

    install_parser.add_argument(
        "-s",
        "--build-system",
        type=str,
        required=True,
        choices=["cmake", "bazel"],
        help="Build system to use (cmake or bazel)",
    )
    install_parser.add_argument(
        "-i", "--install-path", type=Path, help="Custom destination install path."
    )

    # Run subcommand
    run_parser = subparsers.add_parser(
        "run", help="Compile and run main.cpp with TensorFlow Lite"
    )
    run_parser.add_argument(
        "-s",
        "--build-system",
        type=str,
        required=True,
        choices=["cmake", "bazel"],
        help="Build system to use (cmake or bazel)",
    )
    run_parser.add_argument(
        "-i",
        "--install-path",
        type=Path,
        required=True,
        help="Custom destination install path.",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.command == "clone":
        clone_repository(args.branch_tag)
    elif args.command == "build":
        if args.cmake:
            # Hardcode CMake target for TFLite C++ library
            build_with_cmake(CLONES / args.branch_tag, CMAKE_TARGET)
        elif args.bazel:
            # Hardcode Bazel target for the full TFLite shared library
            build_with_bazel(CLONES / args.branch_tag, BAZEL_TARGET)
    elif args.command == "install":
        install_artifacts(args.build_system, args.branch_tag, args.install_path)
    elif args.command == "run":
        runner(args.build_system, args.install_path)
