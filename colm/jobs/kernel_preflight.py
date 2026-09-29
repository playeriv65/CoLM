"""Load a configured hub kernel in a separate, offline interpreter before queue execution."""

import json
import sys


def check(requirement: dict) -> None:
    from kernels import get_kernel

    repo = requirement["repo"]
    version = requirement["version"]
    kernel = get_kernel(repo, version=version)
    missing = [
        name for name in requirement.get("symbols", []) if not callable(getattr(kernel, name, None))
    ]
    if missing:
        raise RuntimeError(f"{repo} version {version} is missing callable symbols {missing}")


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    requirement = json.loads(args[0])
    try:
        check(requirement)
    except Exception as exc:
        repo, version = requirement["repo"], requirement["version"]
        print(
            f"{repo} version {version} is unavailable or cannot load offline: {exc}. "
            "The pinned local snapshot must contain a complete build for the current Torch, "
            "CUDA and GPU architecture; prewarm it online with kernels.get_kernel.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
