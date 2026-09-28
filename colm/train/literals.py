from typing import Literal, get_args, get_origin


def check_literals(obj) -> None:
    """Reject values outside the `Literal` type of a field (JSON values are not checked by argparse)."""
    for name, hint in type(obj).__dict__.get("__annotations__", {}).items():
        if get_origin(hint) is Literal and getattr(obj, name) not in get_args(hint):
            raise ValueError(f"{name}={getattr(obj, name)!r} is not one of {get_args(hint)}")
