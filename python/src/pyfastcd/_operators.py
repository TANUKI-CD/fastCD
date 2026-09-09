"""Private NumPy ufunc dispatch for pyfastcd scalar and array objects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

_Handler = Callable[[Any, tuple[Any, ...]], Any]


def _supported_operand(value: Any) -> bool:
    """Return whether *value* belongs to the intentional ufunc surface."""

    from ._array import ElementArray, _is_real_scalar
    from ._core import Element

    return isinstance(value, (Element, ElementArray)) or _is_real_scalar(value)


def _extract_out(kwargs: dict[str, Any]) -> Any | None:
    """Return one supported ufunc output target or decline extra options."""

    options = dict(kwargs)
    out = options.pop("out", None)
    if options:
        return NotImplemented
    if out is None:
        return None
    if not isinstance(out, tuple) or len(out) != 1:
        return NotImplemented
    return out[0]


def _unary(name: str) -> _Handler:
    """Map a unary ufunc onto a receiver method or operator."""

    def apply(receiver: Any, inputs: tuple[Any, ...]) -> Any:
        del inputs
        return getattr(receiver, name)()

    return apply


def _binary(forward: str, reflected: str) -> _Handler:
    """Map a binary ufunc onto the forward or reflected operator."""

    def apply(receiver: Any, inputs: tuple[Any, ...]) -> Any:
        if inputs[0] is receiver:
            return getattr(receiver, forward)(inputs[1])
        return getattr(receiver, reflected)(inputs[0])

    return apply


def _forward_only(forward: str) -> _Handler:
    """Map a binary ufunc that has no reflected form onto its operator."""

    def apply(receiver: Any, inputs: tuple[Any, ...]) -> Any:
        if inputs[0] is not receiver:
            return NotImplemented
        return getattr(receiver, forward)(inputs[1])

    return apply


_MULTIPLY_OPERATOR = _binary("__mul__", "__rmul__")


def _multiply(receiver: Any, inputs: tuple[Any, ...], out: Any | None) -> Any:
    """Route ``np.multiply`` through the operators or, with ``out``, the parent.

    The operator path matters for NumPy scalar operands: ``Algebra.multiply``
    evaluates ``x * y``, and a NumPy scalar on the left would re-enter the
    ufunc machinery and recurse.
    """

    if out is None:
        return _MULTIPLY_OPERATOR(receiver, inputs)
    left, right = inputs
    result = receiver.algebra.multiply(left, right, out=out)
    return out if isinstance(out, np.ndarray) else result


# The supported ufunc surface: each entry names the operator or method that
# already implements the operation for both Element and ElementArray.  Only
# np.multiply accepts an out= target; it is dispatched separately.
_HANDLERS: dict[Any, _Handler] = {
    np.add: _binary("__add__", "__radd__"),
    np.subtract: _binary("__sub__", "__rsub__"),
    np.true_divide: _forward_only("__truediv__"),
    np.power: _forward_only("__pow__"),
    np.negative: _unary("__neg__"),
    np.positive: _unary("__pos__"),
    np.square: _unary("square"),
    np.conjugate: _unary("conjugate"),
    np.absolute: _unary("__abs__"),
    np.equal: _binary("__eq__", "__eq__"),
    np.not_equal: _binary("__ne__", "__ne__"),
}


def _dispatch(
    receiver: Any,
    ufunc: Any,
    method: str,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Implement the supported NumPy ufunc subset for one receiver."""

    if method != "__call__":
        return NotImplemented
    out = _extract_out(kwargs)
    if out is NotImplemented:
        return NotImplemented
    if not all(_supported_operand(value) for value in inputs):
        return NotImplemented
    if ufunc is np.multiply:
        return _multiply(receiver, inputs, out)
    if out is not None:
        return NotImplemented
    handler = _HANDLERS.get(ufunc)
    if handler is None:
        return NotImplemented
    return handler(receiver, inputs)


def _element_array_ufunc(
    self: Any, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
) -> Any:
    """Implement ``Element.__array_ufunc__``; called from the C extension."""

    return _dispatch(self, ufunc, method, inputs, kwargs)


def _dispatch_array_ufunc(
    self: Any,
    ufunc: Any,
    method: str,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Implement ``ElementArray.__array_ufunc__``."""

    return _dispatch(self, ufunc, method, inputs, kwargs)
