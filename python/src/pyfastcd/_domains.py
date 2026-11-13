"""Field-aware adapters for complex and finite-field parents.

Real parents use the _core extension directly. Over C and finite fields,
_core runs Element arithmetic whose operands share one parent, and scalings
by builtin scalars, without calling Python. These adapters handle every
other operand combination, all array arithmetic, and coercion; their
coefficient loops execute in _domains_core.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from . import _core, _domains_core
from ._core import Element
from ._field import FiniteField
from ._validation import DIFFERENT_ALGEBRA_MESSAGE

_OPS = {
    name: i
    for i, name in enumerate(
        (
            "multiply",
            "add",
            "subtract",
            "negative",
            "scale",
            "divide",
            "conjugate",
            "imag_part",
            "inner",
            "quadratic_norm",
            "reciprocal",
            "square",
            "imag_cross",
            "commutator",
            "anticommutator",
            "associator",
            "basis_left",
            "basis_right",
            "left_operator_commutator",
            "right_operator_commutator",
        )
    )
}


_AUTO = _core._algorithm("auto", 1)
_ADD = _OPS["add"]
_SUBTRACT = _OPS["subtract"]
_MULTIPLY = _OPS["multiply"]
_SCALE = _OPS["scale"]
_DIVIDE = _OPS["divide"]
_INNER = _OPS["inner"]
_QUADRATIC_NORM = _OPS["quadratic_norm"]

# Array methods that map one complete record to another of the same parent
_ARRAY_UNARY = {
    "__neg__": _OPS["negative"],
    "negative": _OPS["negative"],
    "conjugate": _OPS["conjugate"],
    "imag_part": _OPS["imag_part"],
    "imag": _OPS["imag_part"],
    "square": _OPS["square"],
    "reciprocal": _OPS["reciprocal"],
}
# Binary array methods, each with its operation and whether the array is the
# right operand
_ARRAY_BINARY = {
    "__add__": (_ADD, False),
    "__radd__": (_ADD, True),
    "__sub__": (_SUBTRACT, False),
    "__rsub__": (_SUBTRACT, True),
    "__mul__": (_MULTIPLY, False),
    "__rmul__": (_MULTIPLY, True),
    "__truediv__": (_DIVIDE, False),
    "imag_cross": (_OPS["imag_cross"], False),
    "commutator": (_OPS["commutator"], False),
    "anticommutator": (_OPS["anticommutator"], False),
    "inner": (_INNER, False),
}
# Operator names of element_binary, as array methods in either operand order
_FORWARD = {
    "add": "__add__",
    "subtract": "__sub__",
    "multiply": "__mul__",
    "divide": "__truediv__",
}
_REFLECTED = {"add": "__radd__", "subtract": "__rsub__", "multiply": "__rmul__"}
_ARRAY_OPERATORS = {
    "__add__": "add",
    "__radd__": "add",
    "__sub__": "subtract",
    "__rsub__": "subtract",
    "__mul__": "multiply",
    "__rmul__": "multiply",
    "__truediv__": "divide",
    "__eq__": "eq",
    "__ne__": "ne",
}
_UFUNC_BINARY = {
    np.add: "add",
    np.subtract: "subtract",
    np.multiply: "multiply",
    np.true_divide: "divide",
    np.equal: "eq",
    np.not_equal: "ne",
}
_UFUNC_UNARY = {
    np.negative: "negative",
    np.positive: "positive",
    np.square: "square",
    np.conjugate: "conjugate",
    np.absolute: "norm",
}
# Returned by a native shortcut that does not apply to its operands
_MISS = object()
_ONE_BYTE = bytes(1)
_INT = frozenset((int,))


def extended(parent: Any) -> bool:
    """Return whether a parent explicitly selected a nonlegacy coefficient field."""

    return bool(parent.char != 0 or parent.deg != 1)


def field_parent(parent: Any) -> Any:
    """Return the finite coefficient parent, or None for complex coefficients."""

    return parent.field


def field_shape(parent: Any) -> tuple[int, ...]:
    """Return storage axes of one coefficient, excluding the CD coordinate axis."""

    return parent.field.coefficient_shape if parent.char else ()


def is_field(parent: Any) -> bool:
    """Distinguish a coefficient scalar parent from a containing CD algebra."""

    return isinstance(parent, FiniteField)


def algorithm_code(name: Any) -> int:
    """Validate an algorithm name for a complex or finite-field parent."""

    if type(name) is str and name == "auto":
        return _AUTO
    # Reuse the real API's name parsing at dimension one, where every real
    # algorithm is valid, so only Cariow--Cariowa needs a field-specific check
    code = _core._algorithm(name, 1)
    if name == "cariow_cariowa":
        raise ValueError("algorithm 'cariow_cariowa' is unavailable for this field")
    return code


def scalar(value: Any, parent: Any) -> bool:
    """Recognize coefficient scalars without treating full CD records as scalars."""

    if isinstance(value, Element):
        return value.algebra is parent.field
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 0 and value.dtype.kind in (
            "biu" if parent.char else "biufc"
        )
    if parent.char:
        return isinstance(value, (int, np.integer, bool, np.bool_)) or hasattr(
            value, "__index__"
        )
    return (
        isinstance(value, (int, float, complex, np.number))
        or hasattr(value, "__complex__")
        or hasattr(value, "__float__")
    )


def _compatible(a: Any, parent: Any) -> bool:
    """Check native dtype, alignment and contiguous trailing record axes.

    Leading batch strides may be negative or zero; polynomial and CD axes
    must remain complete. Residue canonicality is checked separately.
    """

    f = len(parent.coefficient_shape)
    if (
        a.dtype != parent.dtype
        or not a.dtype.isnative
        or not a.flags.aligned
        or a.ndim < f
    ):
        return False
    if f and a.shape[-f:] != parent.coefficient_shape:
        return False
    if a.flags.c_contiguous:
        # C order makes the record axes contiguous and every stride a
        # multiple of the item size
        return True
    stride = a.itemsize
    for k in range(a.ndim - 1, a.ndim - f - 1, -1):
        if a.size and a.shape[k] > 1 and a.strides[k] != stride:
            return False
        stride *= a.shape[k]
    return all(
        extent <= 1 or s % a.dtype.alignment == 0
        for extent, s in zip(
            a.shape[: a.ndim - f], a.strides[: a.ndim - f], strict=True
        )
    )


def _finite_values(parent: Any, values: Any) -> Any:
    """Unwrap nested field scalars before NumPy inspects their buffer protocol.

    A degree-one scalar exports a zero-dimensional uint32 buffer, which
    NumPy can otherwise mistake for a byte string inside a Python sequence.
    Keep native numeric arrays untouched so zero-copy inputs remain views.
    """

    if isinstance(values, Element):
        if values.algebra is not parent and values.algebra is not parent.field:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
        return values.coefficients
    if isinstance(values, (list, tuple)):
        return [_finite_values(parent, value) for value in values]
    if isinstance(values, np.ndarray) and values.dtype.kind == "O":
        return _finite_values(parent, values.tolist())
    return values


def coefficients(
    parent: Any, values: Any, *, copy: bool | None, single: bool = False
) -> Any:
    """Validate and construct storage under the requested copy policy.

    Finite reduction precedes narrowing, so large Python integers reduce to
    their exact residues. A shared mutable array must already be canonical;
    the native boundary checks it again before every arithmetic call.
    """

    ElementArray = _array.ElementArray
    if copy is not None and not isinstance(copy, bool):
        raise TypeError("copy must be True, False, or None")
    if not single and type(values) is np.ndarray and values.dtype == parent.dtype:
        # Storage of the parent's own dtype needs no conversion, only the
        # shape, layout and residue checks
        f = len(parent.coefficient_shape)
        if values.ndim < f or (f and values.shape[-f:] != parent.coefficient_shape):
            raise ValueError(
                f"expected trailing coefficient shape {parent.coefficient_shape}"
            )
        canonical = _canonical(parent, values)
        good = canonical and _compatible(values, parent)
        if copy is False:
            if not good:
                raise ValueError(
                    "copy=False requires canonical native storage "
                    "with contiguous coefficient axes"
                )
            return values
        if good and copy is None:
            return values
        if not canonical:
            values = values % parent.char
        return np.array(values, dtype=parent.dtype, copy=True, order="C")
    original = values
    if isinstance(values, (Element, ElementArray)):
        if values.algebra is not parent:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
        values = values.coefficients
    else:
        values = _array._from_dlpack_if_needed(values, copy=copy)
    if isinstance(values, (set, dict, str, bytes)):
        raise TypeError("coefficients must be an ordered numeric input")
    if single and not hasattr(values, "__array__") and not scalar(values, parent):
        try:
            values = list(values)
        except TypeError:
            pass
    normalized = _finite_values(parent, values) if parent.char else values
    repacked = normalized is not values
    values = normalized
    a = np.asarray(values)
    if parent.char:
        if a.dtype.kind not in "biuO" and not isinstance(
            values, (np.ndarray, memoryview)
        ):
            # Mixed large positive and negative Python integers can make NumPy
            # infer float64, so recover their exact values before reduction
            a = np.asarray(values, dtype=object)
        if a.dtype.kind not in "biuO":
            raise TypeError("finite-field coefficients must be integers")
        if a.dtype.kind == "O":
            entries = [operator.index(x) for x in a.flat]
            a = np.asarray(entries, dtype=object).reshape(a.shape)
    else:
        if single and scalar(values, parent):
            a = np.asarray(complex(values), dtype=np.complex128)
        if a.dtype.kind not in "biufc":
            raise TypeError("complex coefficients must be numeric")
    f = len(parent.coefficient_shape)
    if single and a.ndim == 0 and f:
        if copy is False:
            raise ValueError("scalar embedding requires a copy")
        result = np.zeros(parent.coefficient_shape, dtype=parent.dtype)
        result.flat[0] = (
            operator.index(a.item()) % parent.char if parent.char else complex(a.item())
        )
        return result
    if a.ndim < f or (f and a.shape[-f:] != parent.coefficient_shape):
        raise ValueError(
            f"expected trailing coefficient shape {parent.coefficient_shape}"
        )
    if single and a.shape != parent.coefficient_shape:
        raise ValueError(f"expected one element of shape {parent.coefficient_shape}")
    canonical = (
        not parent.char or not a.size or bool(np.all((a >= 0) & (a < parent.char)))
    )
    good = _compatible(a, parent) and canonical
    shareable = not repacked and (
        isinstance(original, (np.ndarray, Element, ElementArray, memoryview))
        or hasattr(original, "__dlpack__")
    )
    if copy is False:
        if not good or not shareable:
            raise ValueError(
                "copy=False requires canonical native storage "
                "with contiguous coefficient axes"
            )
        return a
    if good and copy is None:
        return a
    if parent.char and not canonical:
        a = a % parent.char
    return np.array(a, dtype=parent.dtype, copy=True, order="C")


def _canonical(parent: Any, a: Any) -> bool:
    """Report whether storage of the parent's dtype holds canonical residues."""

    if not parent.char or not a.size:
        return True
    if a.flags.c_contiguous and a.flags.aligned:
        return bool(_domains_core.canonical(parent._ctx, a))
    return bool(np.all(a < parent.char))


def _integer_lists(values: Any, depth: int) -> bool:
    """Report whether lists or tuples nested to a depth hold only exact ints."""

    if depth == 1:
        return set(map(type, values)) <= _INT
    return all(
        (type(value) is list or type(value) is tuple)
        and _integer_lists(value, depth - 1)
        for value in values
    )


def element(parent: Any, value: Any = 0) -> Any:
    """Copy one record into an immutable native Element."""

    kind = type(value)
    if kind is Element:
        owner = value.algebra
        if owner is parent:
            return value.copy()
        if owner is parent.field:
            a = np.zeros(parent.coefficient_shape, dtype=parent.dtype)
            a[0] = value.coefficients
            return _core._extended_element(parent, a)
    elif kind is np.ndarray:
        # A record already in native storage needs no coercion; the native
        # constructor copies it and reduces finite residues
        if (
            value.dtype == parent.dtype
            and value.shape == parent.coefficient_shape
            and value.flags.c_contiguous
        ):
            return _core._extended_element(parent, value)
    elif kind is int and parent.char:
        a = np.zeros(parent.coefficient_shape, dtype=parent.dtype)
        a.flat[0] = value % parent.char
        return _core._extended_element(parent, a)
    elif not parent.char and (kind is int or kind is float or kind is complex):
        a = np.zeros(parent.coefficient_shape, dtype=parent.dtype)
        a.flat[0] = complex(value)
        return _core._extended_element(parent, a)
    elif kind is list or kind is tuple:
        # A record written as numbers converts as in the general path, which
        # is still needed for field scalars, which NumPy must not read as
        # buffers, and for every other kind of content
        shape = parent.coefficient_shape
        if not parent.char or (shape and _integer_lists(value, len(shape))):
            a = np.asarray(value)
            if a.shape == shape and a.dtype.kind in ("iu" if parent.char else "iufc"):
                if parent.char:
                    a = a % parent.char
                return _core._extended_element(
                    parent, a.astype(parent.dtype, order="C")
                )
    return _core._extended_element(
        parent, coefficients(parent, value, copy=True, single=True)
    )


def array(parent: Any, values: Any, *, copy: bool | None) -> Any:
    """Wrap validated coefficient storage in an ElementArray."""

    return _array.ElementArray._from_coefficients(
        parent, coefficients(parent, values, copy=copy)
    )


def _result(parent: Any, a: Any, batched: bool) -> Any:
    """Choose a scalar or array wrapper without changing its coefficient parent."""

    return _array.ElementArray._wrap(parent, a) if batched else element(parent, a)


def _operand(
    parent: Any, value: Any, *, coerce: bool = False, allow_scalars: bool = False
) -> Any:
    """Accept a record, embedding coefficient scalars only when explicitly enabled.

    Raw packed records coerce only for constructors and explicit algebra
    methods. Named structural methods require records, as on the real path.
    Operators decline raw records so NumPy dispatch stays unambiguous.
    """

    ElementArray = _array.ElementArray
    if isinstance(value, (Element, ElementArray)):
        if value.algebra is parent:
            if type(value) is ElementArray:
                return value._coefficients
            return np.asarray(value)
        if allow_scalars and value.algebra is parent.field:
            if isinstance(value, Element):
                return np.asarray(element(parent, value))
            a = np.zeros(
                value.batch_shape + parent.coefficient_shape, dtype=parent.dtype
            )
            if parent.deg > 1:
                a[..., 0, :] = value.coefficients
            else:
                a[..., 0] = value.coefficients
            return a
        raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
    if allow_scalars and scalar(value, parent):
        return np.asarray(element(parent, value))
    if coerce:
        return coefficients(parent, value, copy=None)
    return None


def _broadcast(
    arrays: list[Any], shapes: list[tuple[int, ...]]
) -> tuple[list[Any], tuple[int, ...]]:
    """Broadcast leading axes while preserving each operand's complete record.

    A scale operand and a full CD record have different trailing shapes;
    including either feature axis in broadcast_shapes would mix coordinates.
    The native bridge repeats an operand that has no batch axes, so operands
    are expanded here only when two different batch shapes must be combined.
    """

    batches = [
        a.shape[: a.ndim - len(s)] for a, s in zip(arrays, shapes, strict=True)
    ]
    batch: tuple[int, ...] = ()
    for candidate in batches:
        if candidate:
            if batch and candidate != batch:
                break
            batch = candidate
    else:
        return arrays, batch
    batch = np.broadcast_shapes(*batches)
    return [
        np.broadcast_to(a, batch + s) for a, s in zip(arrays, shapes, strict=True)
    ], batch


def apply(
    parent: Any,
    op: str,
    *values: Any,
    out: Any = None,
    algorithm: Any = "auto",
    index: int = 0,
    coerce: bool = False,
    allow_scalars: bool = False,
    defer: bool = False,
) -> Any:
    """Prepare one native operation and restore its public result type.

    Broadcast only batch axes, validate out, and delegate coefficient loops
    to the buffer bridge. Finite reductions change the result parent to F;
    complex reductions return ordinary complex scalars or ndarrays. The
    bridge validates all records before writing any part of the batch.
    Only operator/ufunc dispatch sets defer=True; public named methods must
    raise for unsupported operands rather than returning NotImplemented.
    """

    code = algorithm_code(algorithm)
    scaling = op == "scale" or op == "divide"
    if out is None and not scaling:
        # Elements of this parent need no coercion, broadcasting or buffers
        for value in values:
            if type(value) is not Element or value.algebra is not parent:
                break
        else:
            return _core._apply_elements(_OPS[op], code, index, *values)
    ElementArray = _array.ElementArray
    reduction = op == "inner" or op == "quadratic_norm"
    arrays = []
    shapes = []
    for k, value in enumerate(values):
        if scaling and k == 1:
            if parent.char:
                if (
                    isinstance(value, (Element, ElementArray))
                    and value.algebra is parent.field
                ):
                    a = (
                        value._coefficients
                        if type(value) is ElementArray
                        else np.asarray(value)
                    )
                elif type(value) is int:
                    a = np.zeros(parent.field.coefficient_shape, dtype=parent.dtype)
                    a.flat[0] = value % parent.char
                else:
                    a = coefficients(parent.field, value, copy=None, single=True)
            else:
                if not scalar(value, parent):
                    if defer:
                        return NotImplemented
                    raise TypeError("scaling requires a coefficient-field scalar")
                a = np.asarray(complex(value), dtype=np.complex128)
            arrays.append(a)
            shapes.append(field_shape(parent))
        else:
            a = _operand(parent, value, coerce=coerce, allow_scalars=allow_scalars)
            if a is None:
                if defer:
                    return NotImplemented
                raise TypeError(
                    "array arithmetic requires Element or ElementArray operands"
                )
            arrays.append(a)
            shapes.append(parent.coefficient_shape)
    arrays, batch = _broadcast(arrays, shapes)
    output_parent = parent.field if reduction and parent.char else parent
    output_tail = field_shape(parent) if reduction else parent.coefficient_shape
    shape = batch + output_tail
    batched = bool(batch) or any(
        isinstance(v, ElementArray)
        or (coerce and not isinstance(v, Element) and not scalar(v, parent))
        for v in values
    )
    if out is None:
        destination = np.empty(shape, dtype=parent.dtype)
    else:
        if isinstance(out, ElementArray):
            if out.algebra is not output_parent:
                raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
            destination = out.coefficients
        elif isinstance(out, np.ndarray):
            destination = out
        else:
            raise TypeError("out must be an ElementArray or ndarray")
        if (
            destination.shape != shape
            or destination.dtype != parent.dtype
            or not destination.dtype.isnative
            or not destination.flags.aligned
            or not destination.flags.c_contiguous
            or not destination.flags.writeable
        ):
            raise ValueError(
                "out has an incompatible shape, dtype, layout, or writeability"
            )
    padded = arrays + [None] * (3 - len(arrays))
    _domains_core.apply(
        parent._ctx, _OPS[op], destination, padded[0], padded[1], padded[2], code, index
    )
    if out is not None:
        return (
            out
            if isinstance(out, ElementArray)
            else ElementArray._from_coefficients(output_parent, destination)
        )
    if reduction and not parent.char:
        return destination if batched else complex(destination.item())
    return _result(output_parent, destination, batched)


def multiply(
    parent: Any,
    x: Any,
    y: Any,
    *,
    out: Any = None,
    algorithm: Any = "auto",
    coerce: bool = True,
    defer: bool = False,
) -> Any:
    """Select scalar scaling or a complete CD product, including field arrays."""

    code = algorithm_code(algorithm)
    if (
        out is None
        and type(x) is Element
        and type(y) is Element
        and x.algebra is parent
        and y.algebra is parent
    ):
        # Field scalars multiply as coefficients, which the operator does
        if isinstance(parent, FiniteField):
            return x * y
        return _core._apply_elements(_MULTIPLY, code, 0, x, y)
    ElementArray = _array.ElementArray
    if (
        type(x) is ElementArray
        and type(y) is ElementArray
        and x._algebra is parent
        and y._algebra is parent
        and not isinstance(parent, FiniteField)
    ):
        return apply(
            parent,
            "multiply",
            x,
            y,
            out=out,
            algorithm=algorithm,
            coerce=coerce,
            allow_scalars=True,
            defer=defer,
        )
    if scalar(y, parent) or (isinstance(y, ElementArray) and y.algebra is parent.field):
        return apply(
            parent,
            "scale",
            x,
            y,
            out=out,
            algorithm=algorithm,
            coerce=coerce,
            allow_scalars=True,
            defer=defer,
        )
    if scalar(x, parent) or (isinstance(x, ElementArray) and x.algebra is parent.field):
        return apply(
            parent,
            "scale",
            y,
            x,
            out=out,
            algorithm=algorithm,
            coerce=coerce,
            allow_scalars=True,
            defer=defer,
        )
    return apply(
        parent,
        "multiply",
        x,
        y,
        out=out,
        algorithm=algorithm,
        coerce=coerce,
        allow_scalars=True,
        defer=defer,
    )


def _parent(value: Any) -> Any:
    """Return a wrapped value's parent without inferring one from raw arrays."""

    if isinstance(value, (Element, _array.ElementArray)):
        return value.algebra
    return None


def _binary_parent(left: Any, right: Any) -> Any:
    """Choose the containing CD parent when one operand is a matching scalar."""

    parent = _parent(left) or _parent(right)
    if parent is None:
        return None
    # A coefficient-field scalar can combine with a containing CD element
    other_parent = _parent(right)
    if other_parent is not None and other_parent.field is parent:
        parent = other_parent
    return parent


def element_binary(name: str, left: Any, right: Any) -> Any:
    """Implement shared element/array operators with strict parent selection."""

    ElementArray = _array.ElementArray
    if name in _FORWARD:
        result = _MISS
        if type(left) is ElementArray:
            result = _array_binary(
                _FORWARD[name], left._algebra, left._coefficients, right
            )
        elif type(right) is ElementArray and name in _REFLECTED:
            result = _array_binary(
                _REFLECTED[name], right._algebra, right._coefficients, left
            )
        if result is not _MISS:
            return result
    elif (
        type(left) is ElementArray
        and type(right) is ElementArray
        and left._algebra is right._algebra
        and left._coefficients.shape == right._coefficients.shape
    ):
        f = len(left._algebra.coefficient_shape)
        same = left._coefficients == right._coefficients
        if f:
            same = same.all(axis=tuple(range(-f, 0)))
        return np.logical_not(same) if name == "ne" else same
    parent = _binary_parent(left, right)
    if parent is None:
        return NotImplemented
    if name in ("eq", "ne"):
        return equal(parent, left, right, invert=name == "ne")
    if name == "multiply":
        return multiply(parent, left, right, coerce=False, defer=True)
    if name == "divide":
        if _parent(left) is None:
            if not is_field(parent):
                return NotImplemented
            left = parent(left)
        if isinstance(right, ElementArray) and right.algebra is parent.field:
            return apply(parent, "divide", left, right, allow_scalars=True, defer=True)
        if not scalar(right, parent):
            return NotImplemented
        return apply(parent, "divide", left, right, allow_scalars=True, defer=True)
    return apply(parent, name, left, right, allow_scalars=True, defer=True)


def equal(parent: Any, left: Any, right: Any, *, invert: bool = False) -> Any:
    """Compare exact values without arithmetic coercion or modular reduction.

    This preserves scalar equality and equal-hash contracts, including large
    integers that floating-point NumPy promotion would otherwise round.
    """

    ElementArray = _array.ElementArray
    receiver = left if _parent(left) is not None else right
    other = right if receiver is left else left
    a = receiver.coefficients
    tail = receiver.algebra.coefficient_shape
    f = len(tail)
    if isinstance(other, (Element, ElementArray)):
        if (receiver.algebra is parent and other.algebra is parent.field) or (
            receiver.algebra is parent.field and other.algebra is parent
        ):
            a = _operand(parent, receiver, allow_scalars=True)
            b = _operand(parent, other, allow_scalars=True)
            f = len(parent.coefficient_shape)
            result = np.all(a == b, axis=tuple(range(-f, 0))) if f else a == b
        elif other.algebra is not receiver.algebra:
            batch = np.broadcast_shapes(
                a.shape[:-f] if f else a.shape,
                other.shape[: -len(other.algebra.coefficient_shape)]
                if other.algebra.coefficient_shape
                else other.shape,
            )
            result = np.zeros(batch, dtype=bool)
        else:
            b = other.coefficients
            result = np.all(a == b, axis=tuple(range(-f, 0))) if f else a == b
    elif scalar(other, receiver.algebra):
        # Comparison is exact and never performs modular scalar coercion
        if receiver.algebra.char:
            try:
                value = operator.index(other)
            except TypeError:
                return NotImplemented
            coordinate = np.zeros(tail, dtype=object)
            coordinate.flat[0] = value
            result = (
                np.all(a == coordinate, axis=tuple(range(-f, 0))) if f else a == value
            )
        else:
            # Python's complex comparison keeps equality and hash semantics for
            # large integers, Decimal and Fraction, which NumPy's numeric
            # promotion would round before comparison
            b = np.zeros(tail, dtype=object)
            b.flat[0] = (
                other.item() if isinstance(other, (np.ndarray, np.generic)) else other
            )
            result = (
                np.all(a.astype(object) == b, axis=tuple(range(-f, 0)))
                if f
                else a.astype(object) == b
            )
    else:
        return NotImplemented
    if invert:
        result = np.logical_not(result)
    if isinstance(left, ElementArray) or isinstance(right, ElementArray):
        return result
    return bool(result)


def _power(value: Any, exponent: Any, modulus: Any = None) -> Any:
    """Use reciprocal and repeated squaring with explicit nonassociative grouping."""

    ElementArray = _array.ElementArray
    if modulus is not None:
        raise TypeError(
            "modular exponentiation is not defined for Cayley--Dickson elements"
        )
    n = operator.index(exponent)
    if n < 0:
        return _power(apply(value.algebra, "reciprocal", value), -n)
    if n == 1:
        return value.copy()
    if n == 2:
        return apply(value.algebra, "square", value)
    result = value.algebra.one()
    if isinstance(value, ElementArray):
        result = value.algebra.array(np.broadcast_to(result.coefficients, value.shape))
    base = value
    while n:
        if n & 1:
            result = multiply(value.algebra, result, base)
        n >>= 1
        if n:
            base = apply(value.algebra, "square", base)
    return result


def element_dispatch(name: str, value: Any, extra: Any = None) -> Any:
    """Handle the Element slots that _core leaves to Python.

    Real elements never reach this function. Over C and finite fields, _core
    answers arithmetic on same-parent operands and plain coefficient access
    itself; representation, hashing, pickling, slices, powers, ufunc
    dispatch and every operand it does not recognize arrive here.
    """

    parent = value.algebra
    a = np.asarray(value)
    if name == "positive":
        return value.copy()
    if name == "repr":
        return f"Element({a.tolist()!r}, algebra={parent!r})"
    if name == "len":
        if is_field(parent):
            raise TypeError("field scalars have no length")
        return parent.dimension
    if name == "getitem":
        if is_field(parent):
            raise TypeError("field scalars are not sequences; use coefficients")
        if isinstance(extra, slice):
            return tuple(
                element_dispatch("getitem", value, i)
                for i in range(*extra.indices(parent.dimension))
            )
        if isinstance(extra, (bool, np.bool_)):
            raise TypeError("element index must be an integer, not bool")
        i = operator.index(extra)
        if i < 0:
            i += parent.dimension
        if not 0 <= i < parent.dimension:
            raise IndexError("element index is out of range")
        return parent.field(a[i]) if parent.char else complex(a[i])
    if name == "iter":
        return iter(
            tuple(
                element_dispatch("getitem", value, i)
                for i in range(element_dispatch("len", value))
            )
        )
    if name == "reduce":
        return (parent, (a.copy(),))
    if name == "dtype":
        return parent.dtype
    if name in ("norm", "absolute", "normalized"):
        raise TypeError(
            "Euclidean norm operations require real coefficients; use quadratic_norm()"
        )
    if name == "power":
        return _power(value, *extra)
    if name == "associator":
        if len(extra) != 2:
            raise TypeError("associator requires exactly two operands")
        if not all(isinstance(operand, Element) for operand in extra):
            raise TypeError("associator operands must be pyfastcd.Element objects")
        return apply(parent, name, value, *extra)
    if name == "inner" or name in ("imag_cross", "commutator", "anticommutator"):
        if not isinstance(extra, Element):
            raise TypeError("operand must be a pyfastcd.Element")
        return apply(parent, name, value, extra)
    return apply(parent, name, value)


def element_keywords(name: str, value: Any, args: Any, kwargs: Any) -> Any:
    """Bridge keyword-capable native slots to the matching Python adapter."""

    return (
        ufunc(value, *args, **(kwargs or {}))
        if name == "__array_ufunc__"
        else element_dispatch(name, value, args)
    )


def ufunc(value: Any, u: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
    """Support selected NumPy calls with the same parents and out contract."""

    if method != "__call__":
        return NotImplemented
    out = kwargs.pop("out", None)
    if kwargs:
        return NotImplemented
    if out is not None:
        if u is not np.multiply or not isinstance(out, tuple) or len(out) != 1:
            return NotImplemented
        target = out[0]
        result = multiply(
            _binary_parent(*inputs), *inputs, out=target, coerce=False, defer=True
        )
        if result is NotImplemented:
            return NotImplemented
        return target if isinstance(target, np.ndarray) else result
    name = _UFUNC_BINARY.get(u)
    if name is not None:
        return element_binary(name, *inputs)
    if u is np.power:
        return _power(*inputs)
    name = _UFUNC_UNARY.get(u)
    if name is not None:
        return (
            array_dispatch(name, value)
            if hasattr(value, "batch_shape")
            else element_dispatch(name, value)
        )
    return NotImplemented


def array_operator(name: str, value: Any, other: Any) -> Any:
    """Run a binary ElementArray method, natively when no operand needs coercion."""

    result = _array_binary(name, value._algebra, value._coefficients, other)
    if result is not _MISS:
        return result
    return array_dispatch(name, value, other)


def array_dispatch(name: str, value: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement array methods for complex and finite-field parents.

    Only leading axes index or broadcast as batches; each coefficient record
    remains intact, including an extension's polynomial coordinate axis.
    """

    parent = value._algebra
    a = value._coefficients
    if not kwargs:
        # Operands that need no coercion go straight to the native bridge,
        # which validates their storage on every call
        if not args:
            code = _ARRAY_UNARY.get(name)
            if code is not None:
                destination = np.empty(a.shape, dtype=a.dtype)
                _domains_core.apply(
                    parent._ctx, code, destination, a, None, None, _AUTO, 0
                )
                return _array.ElementArray._wrap(parent, destination)
            if name == "quadratic_norm" or name == "norm_squared":
                return _array_reduce(parent, _QUADRATIC_NORM, a, None)
        elif len(args) == 1 and name not in _ARRAY_OPERATORS:
            result = _array_binary(name, parent, a, args[0])
            if result is not _MISS:
                return result
    f = len(parent.coefficient_shape)
    batch = a.shape[:-f] if f else a.shape
    if name == "batch_shape":
        return batch
    if name == "real":
        if is_field(parent):
            return array(parent, a, copy=None)
        return (
            array(
                parent.field,
                a[..., 0, :] if parent.char and parent.deg > 1 else a[..., 0],
                copy=None,
            )
            if parent.char
            else a[..., 0]
        )
    if name == "imag":
        return apply(parent, "imag_part", value)
    if name == "__getitem__":
        key = args[0]
        if type(key) is int and batch:
            return _result(parent, a[key], len(batch) > 1)
        # A key applies only to batch axes, so coefficients stay complete; a
        # one-byte array with zero strides has the batch shape for any size
        dummy = np.ndarray(
            batch, dtype=np.uint8, buffer=_ONE_BYTE, strides=(0,) * len(batch)
        )
        selected = dummy[key]
        result = a[_batch_key(key, len(batch), f)]
        return _result(parent, result, np.ndim(selected) != 0)
    if name == "__setitem__":
        key, item = args
        if not a.flags.writeable:
            raise ValueError("assignment destination is read-only")
        result = array_dispatch("__getitem__", value, key)
        target_shape = result.shape
        if scalar(item, parent):
            source = element(parent, item).coefficients
        else:
            source = _operand(parent, item, coerce=True, allow_scalars=True)
        np.broadcast_to(source, target_shape)
        a[_batch_key(key, len(batch), f)] = source
        return None
    if name == "__len__":
        if not batch:
            raise TypeError("an array with no batch axes has no length")
        return batch[0]
    if name == "__iter__":
        return iter(
            array_dispatch("__getitem__", value, i)
            for i in range(array_dispatch("__len__", value))
        )
    if name in ("copy", "__copy__", "__deepcopy__", "__pos__", "positive"):
        return array(parent, a, copy=True)
    if name == "__bool__":
        raise ValueError("the truth value of an ElementArray is ambiguous")
    if name == "__array_ufunc__":
        return ufunc(value, *args, **kwargs)
    if name in ("__pow__", "power"):
        return _power(value, *args, **kwargs)
    if name in ("norm", "normalized", "__abs__"):
        raise TypeError(
            "Euclidean norm operations require real coefficients; use quadratic_norm()"
        )
    operator_name = _ARRAY_OPERATORS.get(name)
    if operator_name is not None:
        left, right = (
            (args[0], value)
            if name in ("__radd__", "__rsub__", "__rmul__")
            else (value, args[0])
        )
        return element_binary(operator_name, left, right)
    if name in ("__neg__", "negative"):
        name = "negative"
    if name in ("norm_squared", "quadratic_norm"):
        name = "quadratic_norm"
    return apply(parent, name, value, *args, **kwargs)


def _array_reduce(parent: Any, code: int, a: Any, b: Any) -> Any:
    """Reduce complete records to coefficients over the batch axes."""

    batch = a.shape[: a.ndim - len(parent.coefficient_shape)]
    destination = np.empty(batch + field_shape(parent), dtype=a.dtype)
    _domains_core.apply(parent._ctx, code, destination, a, b, None, _AUTO, 0)
    if parent.char:
        return _array.ElementArray._wrap(parent.field, destination)
    return destination


def _array_binary(name: str, parent: Any, a: Any, other: Any) -> Any:
    """Run a binary array method natively when no operand needs coercion.

    ``a`` is the storage of an array of ``parent``, and ``other`` is the
    other operand as written. The shortcut covers an array of the same parent
    and shape, an Element of the parent, a scalar of its coefficient field,
    and scalars of exact builtin types. It returns ``_MISS`` for any other
    operand, which the general path then coerces or rejects.
    """

    entry = _ARRAY_BINARY.get(name)
    if entry is None:
        return _MISS
    code, reflected = entry
    kind = type(other)
    if kind is _array.ElementArray:
        if (
            reflected
            or code == _DIVIDE
            or other._algebra is not parent
            or other._coefficients.shape != a.shape
        ):
            return _MISS
        b = other._coefficients
        # Field scalars multiply as coefficients
        if code == _MULTIPLY and isinstance(parent, FiniteField):
            code = _SCALE
    elif kind is Element:
        owner = other.algebra
        b = other
        if owner is parent:
            if isinstance(parent, FiniteField):
                if code == _MULTIPLY:
                    code, reflected = _SCALE, False
            elif code == _DIVIDE:
                return _MISS
        elif parent.char and owner is parent.field:
            if code == _MULTIPLY:
                code, reflected = _SCALE, False
            elif code == _ADD or code == _SUBTRACT:
                b = element(parent, other)
            elif code != _DIVIDE:
                return _MISS
        else:
            return _MISS
    elif kind is int if parent.char else (
        kind is int or kind is float or kind is complex
    ):
        if code == _MULTIPLY or code == _DIVIDE:
            if code == _MULTIPLY:
                code, reflected = _SCALE, False
            if parent.char:
                b = np.zeros(parent.field.coefficient_shape, dtype=a.dtype)
                b.flat[0] = other % parent.char
            else:
                b = np.asarray(complex(other), dtype=a.dtype)
        elif code == _ADD or code == _SUBTRACT:
            b = element(parent, other)
        else:
            return _MISS
    else:
        return _MISS
    if code == _INNER:
        return _array_reduce(parent, code, a, b)
    destination = np.empty(a.shape, dtype=a.dtype)
    if reflected:
        _domains_core.apply(parent._ctx, code, destination, b, a, None, _AUTO, 0)
    else:
        _domains_core.apply(parent._ctx, code, destination, a, b, None, _AUTO, 0)
    return _array.ElementArray._wrap(parent, destination)


def _batch_key(key: Any, rank: int, features: int) -> tuple[Any, ...]:
    """Append complete coefficient slices to an index of the batch axes only."""

    expanded = key if isinstance(key, tuple) else (key,)
    consumed = 0
    for k in expanded:
        if k is None or k is Ellipsis:
            continue
        if isinstance(k, (bool, np.bool_)):
            continue
        index = np.asarray(k) if isinstance(k, (list, np.ndarray)) else None
        consumed += index.ndim if index is not None and index.dtype.kind == "b" else 1
    if not any(k is Ellipsis for k in expanded):
        expanded += (slice(None),) * (rank - consumed)
    return expanded + (slice(None),) * features



# The array module imports these adapters while it loads, so it is bound here
# only after every name above exists
from . import _array  # noqa: E402
