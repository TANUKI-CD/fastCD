from typing import Any

def field(
    char: int, deg: int, modulus: tuple[int, ...] | None
) -> tuple[Any, tuple[int, ...]]: ...
def context(field: Any, dimension: int, scalar: bool) -> Any: ...
def apply(
    context: Any,
    operation: int,
    out: Any,
    x: Any,
    y: Any,
    z: Any,
    algorithm: int,
    index: int,
) -> None: ...
def backend(context: Any) -> str: ...
