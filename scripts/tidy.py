#!/usr/bin/env python3

import json
import shlex
import sys
from pathlib import Path


def cuda_args(args: list[str]) -> list[str]:
    # NVCC validates newer CUDA headers.
    output = [
        "clang++",
        "--cuda-path=/usr/local/cuda",
        "--cuda-gpu-arch=sm_86",
        "-Wno-unknown-cuda-version",
    ]
    index = 1
    while index < len(args):
        arg = args[index]
        if (
            arg == "-forward-unknown-to-host-compiler"
            or arg.startswith(("--generate-code=", "-arch="))
            or (arg.startswith("-t") and arg[2:].isdigit())
        ):
            index += 1
            continue
        if arg in {"-Xcompiler", "--options-file"}:
            index += 2
            continue
        if arg.startswith(("-Xcompiler=", "--options-file=")):
            index += 1
            continue
        if arg == "-x" and index + 1 < len(args) and args[index + 1] == "cu":
            output.extend(("-x", "cuda"))
            index += 2
            continue
        output.append(arg)
        index += 1
    return output


def main() -> None:
    source = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    commands = json.loads(source.read_text())
    result = []
    for command in commands:
        file = Path(command["file"])
        if not str(file).startswith("/app/src/"):
            continue
        args = command.get("arguments") or shlex.split(command["command"])
        if file.suffix == ".cu":
            args = cuda_args(args)
        result.append(
            {"directory": command["directory"], "file": str(file), "arguments": args}
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "compile_commands.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
