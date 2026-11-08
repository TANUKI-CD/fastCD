"""Field-aware adapters; all coefficient loops execute in _domains_core.

Real parents use the _core extension directly. These adapters are entered
only for explicitly selected complex and finite-field parents.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from . import _core, _domains_core
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

    from ._field import FiniteField

    return isinstance(parent, FiniteField)


def algorithm_code(name: Any) -> int:
    """Validate an algorithm name for a complex or finite-field parent."""

    # Reuse the real API's name parsing at dimension one, where every real
    # algorithm is valid, so only Cariow--Cariowa needs a field-specific check
    code = _core._algorithm(name, 1)
    if name == "cariow_cariowa":
        raise ValueError("algorithm 'cariow_cariowa' is unavailable for this field")
    return code


def scalar(value: Any, parent: Any) -> bool:
    """Recognize coefficient scalars without treating full CD records as scalars."""

    if isinstance(value, _core.Element):
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

    if isinstance(values, _core.Element):
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

    from ._array import ElementArray, _from_dlpack_if_needed

    if copy is not None and not isinstance(copy, bool):
        raise TypeError("copy must be True, False, or None")
    original = values
    if isinstance(values, (_core.Element, ElementArray)):
        if values.algebra is not parent:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
        values = values.coefficients
    else:
        values = _from_dlpack_if_needed(values, copy=copy)
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
        isinstance(original, (np.ndarray, _core.Element, ElementArray, memoryview))
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


def element(parent: Any, value: Any = 0) -> Any:
    """Copy one record into an immutable native Element."""

    if (
        isinstance(value, _core.Element)
        and value.algebra is parent.field
        and not is_field(parent)
    ):
        a = np.zeros(parent.coefficient_shape, dtype=parent.dtype)
        a[0] = value.coefficients
    else:
        a = coefficients(parent, value, copy=True, single=True)
    return _core._extended_element(
        parent, a.tobytes(), parent.coefficient_shape, not bool(parent.char)
    )


def array(parent: Any, values: Any, *, copy: bool | None) -> Any:
    """Wrap validated coefficient storage in an ElementArray."""

    from ._array import ElementArray

    return ElementArray._from_coefficients(
        parent, coefficients(parent, values, copy=copy)
    )


def _result(parent: Any, a: Any, batched: bool) -> Any:
    """Choose a scalar or array wrapper without changing its coefficient parent."""

    from ._array import ElementArray

    return ElementArray._from_coefficients(parent, a) if batched else element(parent, a)


def _operand(
    parent: Any, value: Any, *, coerce: bool = False, allow_scalars: bool = False
) -> Any:
    """Accept a record, embedding coefficient scalars only when explicitly enabled.

    Raw packed records coerce only for constructors and explicit algebra
    methods. Named structural methods require records, as on the real path.
    Operators decline raw records so NumPy dispatch stays unambiguous.
    """

    from ._array import ElementArray

    if isinstance(value, (_core.Element, ElementArray)):
        if value.algebra is parent:
            return value.coefficients
        if allow_scalars and value.algebra is parent.field:
            if isinstance(value, _core.Element):
                return element(parent, value).coefficients
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
        return element(parent, value).coefficients
    if coerce:
        return coefficients(parent, value, copy=None)
    return None


def _broadcast(
    arrays: list[Any], shapes: list[tuple[int, ...]]
) -> tuple[list[Any], tuple[int, ...]]:
    """Broadcast leading axes while preserving each operand's complete record.

    A scale operand and a full CD record have different trailing shapes;
    including either feature axis in broadcast_shapes would mix coordinates.
    """

    batch = np.broadcast_shapes(
        *(
            a.shape[: -len(s)] if s else a.shape
            for a, s in zip(arrays, shapes, strict=True)
        )
    )
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

    from ._array import ElementArray

    code = algorithm_code(algorithm)
    reduction = op in ("inner", "quadratic_norm")
    scaling = op in ("scale", "divide")
    arrays = []
    shapes = []
    for k, value in enumerate(values):
        if scaling and k == 1:
            if parent.char:
                if (
                    isinstance(value, (_core.Element, ElementArray))
                    and value.algebra is parent.field
                ):
                    a = value.coefficients
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
        or (coerce and not isinstance(v, _core.Element) and not scalar(v, parent))
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

    from ._array import ElementArray

    algorithm_code(algorithm)
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

    from ._array import ElementArray

    if isinstance(value, (_core.Element, ElementArray)):
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

    from ._array import ElementArray

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

    from ._array import ElementArray

    receiver = left if _parent(left) is not None else right
    other = right if receiver is left else left
    a = receiver.coefficients
    tail = receiver.algebra.coefficient_shape
    f = len(tail)
    if isinstance(other, (_core.Element, ElementArray)):
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

    from ._array import ElementArray

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
    """Handle native Element slots for complex and finite-field parents.

    Real elements never reach this function. Coefficient access, hashes and
    reductions keep the selected field.
    """

    parent = value.algebra
    a = np.asarray(value)
    if name == "copy" or name == "positive":
        return element(parent, a)
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
    if name == "bool":
        return bool(np.any(a != 0))
    if name == "hash":
        if not parent.char and bool(np.any(np.isnan(a))):
            # Fresh NaN scalar objects hash by identity, but the value's hash
            # must be stable over repeated calls
            return parent.dimension
        flat = tuple(a.flat)
        if parent.char:
            if not is_field(parent) and bool(np.all(a[1:] == 0)):
                # A constant CD element compares equal to its matching field
                # scalar, including non-prime extension coefficients
                flat = tuple(np.asarray(a[0]).flat)
                parent = parent.field
            pure = all(v == 0 for v in flat[1:])
            return hash(int(flat[0])) if pure else hash((parent, flat))
        pure = all(v == 0 for v in flat[1:])
        return hash(complex(flat[0])) if pure else hash((parent, flat))
    if name == "reduce":
        return (parent, (a.copy(),))
    if name == "dimension" or name == "level" or name == "dtype":
        return getattr(parent, name)
    if name == "shape":
        return parent.coefficient_shape
    if name == "real":
        if is_field(parent):
            return element(parent, a)
        return parent.field(a[0]) if parent.char else complex(a[0])
    if name in ("norm", "absolute", "normalized"):
        raise TypeError(
            "Euclidean norm operations require real coefficients; use quadratic_norm()"
        )
    if name == "power":
        return _power(value, *extra)
    if name == "associator":
        if len(extra) != 2:
            raise TypeError("associator requires exactly two operands")
        if not all(isinstance(operand, _core.Element) for operand in extra):
            raise TypeError("associator operands must be pyfastcd.Element objects")
        return apply(parent, name, value, *extra)
    if name == "inner" or name in ("imag_cross", "commutator", "anticommutator"):
        if not isinstance(extra, _core.Element):
            raise TypeError("operand must be a pyfastcd.Element")
        return apply(parent, name, value, extra)
    if name in ("norm_squared", "quadratic_norm"):
        name = "quadratic_norm"
    if name == "imag":
        name = "imag_part"
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
    binary = {
        np.add: "add",
        np.subtract: "subtract",
        np.multiply: "multiply",
        np.true_divide: "divide",
        np.equal: "eq",
        np.not_equal: "ne",
    }
    if u in binary:
        return element_binary(binary[u], *inputs)
    if u is np.power:
        return _power(*inputs)
    unary = {
        np.negative: "negative",
        np.positive: "positive",
        np.square: "square",
        np.conjugate: "conjugate",
        np.absolute: "norm",
    }
    if u in unary:
        return (
            array_dispatch(unary[u], value)
            if hasattr(value, "batch_shape")
            else element_dispatch(unary[u], value)
        )
    return NotImplemented


def array_dispatch(name: str, value: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement array methods for complex and finite-field parents.

    Only leading axes index or broadcast as batches; each coefficient record
    remains intact, including an extension's polynomial coordinate axis.
    """

    parent = value.algebra
    a = value._coefficients
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
        # A key applies only to batch axes, so coefficients stay complete
        dummy = np.broadcast_to(np.empty((), dtype=np.uint8), batch)
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
    binary = {
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
    if name in binary:
        left, right = (
            (args[0], value)
            if name in ("__radd__", "__rsub__", "__rmul__")
            else (value, args[0])
        )
        return element_binary(binary[name], left, right)
    if name in ("__neg__", "negative"):
        name = "negative"
    if name in ("norm_squared", "quadratic_norm"):
        name = "quadratic_norm"
    return apply(parent, name, value, *args, **kwargs)


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
