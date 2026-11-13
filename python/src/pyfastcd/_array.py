"""NumPy-backed batches of standard Cayley--Dickson elements."""

from __future__ import annotations

import operator
from collections.abc import Iterator
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    SupportsFloat,
    SupportsIndex,
    cast,
    overload,
)

import numpy as np
from numpy.lib import NumpyVersion

from . import _core
from ._core import Element
from ._domains import array_dispatch as _domain_array_dispatch
from ._domains import array_operator as _domain_array_operator
from ._typing import _RealScalar
from ._validation import (
    DIFFERENT_ALGEBRA_MESSAGE,
    INTEGER_EXPONENT_MESSAGE,
    MODULAR_POWER_MESSAGE,
)

if TYPE_CHECKING:
    from ._algebra import Algebra


_NUMPY_HAS_EXTENDED_DLPACK = NumpyVersion(np.__version__) >= "2.1.0"
_NDARRAY_DLPACK_PARAMETERS = frozenset(
    {"self", "stream", "max_version", "dl_device", "copy"}
    if _NUMPY_HAS_EXTENDED_DLPACK
    else {"self", "stream"}
)
_FROM_DLPACK_PARAMETERS = frozenset(
    {"x", "device", "copy"} if _NUMPY_HAS_EXTENDED_DLPACK else {"x"}
)
_DLPACK_HOST_DEVICE_TYPES = frozenset({1, 3, 11, 13})
_DLPACK_CPU_CONVERTIBLE_DEVICE_TYPES = frozenset({3, 11, 13})


def _dlpack_pair(value: Any, *, name: str) -> tuple[int, int] | None:
    """Validate an optional DLPack version or device pair."""

    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be None or a pair of integers")
    try:
        pair = (operator.index(value[0]), operator.index(value[1]))
    except TypeError as error:
        raise TypeError(f"{name} must be None or a pair of integers") from error
    if pair[0] < 0 or pair[1] < 0:
        raise TypeError(f"{name} entries must be nonnegative")
    return pair


def _is_real_scalar(value: Any) -> bool:
    """Return whether a value has unambiguous real-scalar semantics."""

    # Decide the operand types of ordinary arithmetic before the protocol
    # check below, which inspects the value attribute by attribute
    kind = type(value)
    if kind is float or kind is int:
        return True
    if kind is ElementArray or kind is Element:
        return False
    if isinstance(value, (complex, np.complexfloating)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 0 and value.dtype.kind in "biuf"
    if not isinstance(value, SupportsFloat):
        return False

    # Some array containers also provide __float__, usually for their 0-D case,
    # so their array dimensionality is honored before the scalar protocol and a
    # dual-protocol coefficient vector cannot be mistaken for a real scalar
    if hasattr(value, "__dlpack__"):
        ndim = getattr(value, "ndim", None)
        if ndim is None:
            shape = getattr(value, "shape", None)
            if shape is None:
                return False
            try:
                ndim = len(shape)
            except TypeError:
                return False
        try:
            normalized_ndim = operator.index(ndim)
        except TypeError:
            return False
        if normalized_ndim != 0:
            return False
        kind = getattr(getattr(value, "dtype", None), "kind", None)
        return kind is None or kind in "biuf"
    if hasattr(value, "__array__"):
        array = np.asarray(value)
        return array.ndim == 0 and array.dtype.kind in "biuf"
    return True


def _validate_source_dtype(array: np.ndarray[Any, Any]) -> None:
    """Require an array to use a supported real numeric dtype."""

    if array.dtype.kind == "c":
        raise TypeError("complex coefficient arrays are unsupported")
    if array.dtype.kind == "O":
        raise TypeError("object coefficient arrays are unsupported")
    if array.dtype.kind not in "biuf":
        raise TypeError("coefficient arrays must contain real numeric values")


def _layout_is_compatible(array: np.ndarray[Any, Any], dimension: int) -> bool:
    """Return whether coefficients can be wrapped without copying."""

    alignment = np.dtype(np.float64).alignment
    return (
        array.dtype == np.dtype(np.float64)
        and array.dtype.isnative
        and array.flags.aligned
        and array.ndim >= 1
        and array.shape[-1] == dimension
        and (
            array.strides[-1] == array.itemsize
            or (dimension == 1 and array.strides[-1] == 0)
            or (array.size == 0 and array.flags.c_contiguous)
        )
        and (
            array.flags.c_contiguous
            or all(
                extent <= 1 or stride % alignment == 0
                for extent, stride in zip(
                    array.shape[:-1], array.strides[:-1], strict=True
                )
            )
        )
    )


def _from_dlpack_if_needed(values: Any, *, copy: bool | None) -> Any:
    """Import a non-NumPy DLPack producer under the requested copy policy."""

    if isinstance(values, (np.ndarray, Element, ElementArray)):
        return values
    if not hasattr(values, "__dlpack__"):
        return values
    if hasattr(values, "__dlpack_device__"):
        device_type, _device_id = values.__dlpack_device__()
        if device_type not in _DLPACK_HOST_DEVICE_TYPES:
            raise ValueError(
                "fastCD is CPU-only; move the array to CPU before conversion"
            )
    if copy is False:
        if "copy" not in _FROM_DLPACK_PARAMETERS:
            raise ValueError(
                "copy=False for a generic DLPack producer requires NumPy 2.1+"
            )
        return np.from_dlpack(values, copy=False)
    if "copy" in _FROM_DLPACK_PARAMETERS:
        return np.from_dlpack(values, copy=None)
    return np.from_dlpack(values)


def _coerce_coefficients(
    algebra: Algebra, values: Any, *, copy: bool | None
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return validated coefficient storage under an explicit copy policy."""

    if copy is not None and not isinstance(copy, bool):
        raise TypeError("copy must be True, False, or None")
    if isinstance(values, ElementArray):
        if values.algebra is not algebra:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
        if copy is True:
            return values._coefficients.copy(order="C")
        return values._coefficients
    if isinstance(values, Element):
        if values.algebra is not algebra:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
        source = np.asarray(values)
    else:
        source = _from_dlpack_if_needed(values, copy=copy)
        source = np.asarray(source)
    _validate_source_dtype(source)
    if source.ndim < 1 or source.shape[-1] != algebra.dimension:
        raise ValueError(
            f"coefficient array must have a final axis of length {algebra.dimension}"
        )

    compatible = _layout_is_compatible(source, algebra.dimension)
    zero_copy_source = isinstance(values, (np.ndarray, Element)) or hasattr(
        values, "__dlpack__"
    )
    if not zero_copy_source:
        try:
            view = memoryview(values)
        except TypeError:
            pass
        else:
            view.release()
            zero_copy_source = True
    if copy is False and (not compatible or not zero_copy_source):
        raise ValueError(
            "copy=False cannot satisfy native float64 alignment and layout"
        )
    if copy is True:
        return np.array(source, dtype=np.float64, order="C", copy=True)
    if compatible:
        return source
    return np.array(source, dtype=np.float64, order="C", copy=True)


def _array_from_values(
    algebra: Algebra, values: Any, *, copy: bool | None
) -> ElementArray:
    """Construct an ElementArray from validated array-like coefficients."""

    coefficients = _coerce_coefficients(algebra, values, copy=copy)
    return ElementArray._from_coefficients(algebra, coefficients)


class ElementArray:
    """A NumPy-backed batch of Cayley--Dickson elements.

    Parameters
    ----------
    *args, **kwargs
        Direct construction is intentionally unsupported.  Use
        :meth:`Algebra.array` for an owned copy or :meth:`Algebra.asarray` for
        copy-controlled conversion.

    Raises
    ------
    TypeError
        Always, when called directly.

    Notes
    -----
    Storage has shape ``batch_shape + algebra.coefficient_shape`` and the
    parent's native coefficient dtype. Default real calls use ``float64``.
    The wrapper is generally mutable: writeability and ownership are
    inherited from the backing ndarray.  Algebraic batch work executes in
    compiled, matrix-free kernels, with one Python wrapper per whole array
    rather than one object per record.

    Operators accept :class:`Element`, :class:`ElementArray`, and coefficient
    scalars. For the default real parent, these include Python integers and floats,
    :class:`decimal.Decimal`, :class:`fractions.Fraction`, real NumPy scalars,
    and real numeric zero-dimensional ndarrays. Complex parents also accept
    complex scalars; finite parents accept integers and matching field scalars.
    Finite arithmetic reduces integer operands but equality compares exactly.
    Nonscalar raw coefficient
    arrays must first be wrapped or passed to :meth:`Algebra.multiply`.
    Leading batch axes broadcast by NumPy rules, while complete coefficient
    axes never broadcast. Powers use ``**``; ordinary Python reserves ``^``
    for bitwise XOR.
    """

    __slots__ = ("_algebra", "_coefficients")
    _algebra: Algebra
    _coefficients: np.ndarray[Any, np.dtype[np.float64]]
    __array_priority__ = 1000.0
    __hash__ = None  # type: ignore[assignment]

    def __class_getitem__(cls, parameters: Any) -> Any:
        """Support optional domain-specific static type annotations."""
        return GenericAlias(cls, parameters)

    if TYPE_CHECKING:

        def __init__(self, _forbidden: NoReturn, /) -> None:
            """Describe the intentionally unavailable public constructor."""

    @classmethod
    def _from_coefficients(
        cls, algebra: Algebra, coefficients: np.ndarray[Any, Any]
    ) -> ElementArray:
        """Wrap validated coefficient storage without copying its data."""

        self = object.__new__(cls)
        self._algebra = algebra
        self._coefficients = coefficients.view()
        return self

    @classmethod
    def _wrap(
        cls, algebra: Algebra, coefficients: np.ndarray[Any, Any]
    ) -> ElementArray:
        """Wrap a newly created ndarray that no other object refers to."""

        self = object.__new__(cls)
        self._algebra = algebra
        self._coefficients = coefficients
        return self

    def __new__(cls, *args: Any, **kwargs: Any) -> ElementArray:
        """Reject direct construction.

        Use :meth:`Algebra.array` or :meth:`Algebra.asarray` so that the
        coefficient dimension, dtype, and ownership contract are validated.
        """

        del args, kwargs
        raise TypeError("ElementArray objects are created by Algebra.array/asarray")

    @property
    def algebra(self) -> Algebra:
        """Algebra: Cached parent algebra for every batch record."""

        return self._algebra

    @property
    def dimension(self) -> int:
        """int: Number of coefficients in every batch record."""

        return self.algebra.dimension

    @property
    def level(self) -> int:
        """int: Parent level ``n`` satisfying ``dimension == 2**n``."""

        return self.algebra.level

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """numpy.dtype: Parent's native coefficient dtype, ``float64`` by default."""

        return self._coefficients.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple[int, ...]: Complete coefficient-storage shape.

        This equals ``batch_shape + algebra.coefficient_shape``. For example,
        100 real quaternions have shape ``(100, 4)``; over F_(p^d), d>1,
        their storage has shape ``(100, 4, d)``.
        """

        return tuple(self._coefficients.shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """tuple[int, ...]: Leading shape that indexes complete elements."""

        if self._algebra._ctx is not None:
            return cast("tuple[int, ...]", _domain_array_dispatch("batch_shape", self))

        return tuple(self._coefficients.shape[:-1])

    @property
    def coefficients(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """numpy.ndarray: A metadata-insulating view of coefficient storage.

        No coefficient data are copied.  Mutations are visible through this
        wrapper and any source shared by :meth:`Algebra.asarray`, but changing
        the returned ndarray's own metadata does not alter this wrapper's
        internal ndarray object.  The view can be read-only when the source
        storage was read-only or was an immutable Element.
        """

        return self._coefficients.view()

    @property
    def real(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """View of coordinate zero over ``batch_shape``.

        Real/complex parents return an ndarray. Finite parents return an
        ElementArray over the matching coefficient field, retaining its
        polynomial axes. A field-scalar array is already its own real part.
        """

        if self._algebra._ctx is not None:
            return cast(
                "np.ndarray[Any, np.dtype[np.float64]]",
                _domain_array_dispatch("real", self),
            )

        return self._coefficients[..., 0:1].squeeze(axis=-1)

    @property
    def imag(self) -> ElementArray:
        """ElementArray: Imaginary parts in newly allocated storage.

        This is equivalent to :meth:`imag_part`; it is not a view.
        """

        return self.imag_part()

    def __repr__(self) -> str:
        """Return a representation showing shape, parent, and coefficients."""

        return (
            f"ElementArray(shape={self.shape}, algebra={self.algebra!r}, "
            f"coefficients={self._coefficients!r})"
        )

    def __array__(
        self, dtype: Any | None = None, copy: bool | None = None
    ) -> np.ndarray[Any, Any]:
        """Expose coefficient storage to NumPy.

        Parameters
        ----------
        dtype : data-type, optional
            Requested NumPy dtype.  A different dtype requires conversion.
        copy : bool or None, optional
            ``True`` returns independent storage.  ``False`` forbids a copy.
            ``None`` permits returning a zero-copy view.

        Returns
        -------
        numpy.ndarray
            A metadata-insulating view when no conversion or copy is
            requested; otherwise an ndarray copy.

        Raises
        ------
        TypeError
            If ``copy`` is not ``True``, ``False``, or ``None``.
        ValueError
            If conversion is requested together with ``copy=False``.
        """

        if copy is not None and not isinstance(copy, bool):
            raise TypeError("copy must be True, False, or None")
        if dtype is not None and np.dtype(dtype) != self.dtype:
            if copy is False:
                raise ValueError("dtype conversion requires a copy")
            return self._coefficients.astype(dtype, copy=True)
        if copy is True:
            return self._coefficients.copy()
        return self._coefficients.view()

    def __dlpack__(
        self,
        *,
        stream: Any = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        """Export coefficient storage through the DLPack protocol.

        Parameters
        ----------
        stream : object, optional
            Must be ``None`` because NumPy exports this host storage without
            device-stream synchronization.
        max_version : tuple[int, int], optional
            Newest DLPack version understood by the consumer.
        dl_device : tuple[int, int], optional
            Requested destination device.  The current host device is
            supported when NumPy can represent the backing array's
            writeability without a forbidden copy.  CPU export is also
            supported from CUDA- or ROCm-pinned host and CUDA-managed storage.
        copy : bool, optional
            ``False`` forbids copying, ``True`` requires independent storage,
            and ``None`` permits a copy when sharing is impossible.

        Returns
        -------
        object
            A DLPack capsule produced by the backing ndarray.

        Raises
        ------
        TypeError
            If a version, device, or copy argument is malformed.
        ValueError
            If ``stream`` is not ``None``.
        BufferError
            If the requested device or no-copy constraint cannot be
            satisfied.

        Notes
        -----
        Extended DLPack controls are delegated when the installed NumPy
        supports them.  On NumPy versions with the legacy stream-only
        producer, compatible device and version requests are validated here;
        ``copy=True`` and permitted read-only copies are implemented before
        export.  A legacy read-only pinned or managed import may be
        re-exportable only as CPU storage when copying is permitted.
        """

        if stream is not None:
            raise ValueError("host DLPack export requires stream=None")
        version = _dlpack_pair(max_version, name="max_version")
        device = _dlpack_pair(dl_device, name="dl_device")
        if copy is not None and not isinstance(copy, bool):
            raise TypeError("copy must be True, False, or None")
        current_device = self.__dlpack_device__()
        cross_host_device_request = (
            device is not None
            and current_device[0] in _DLPACK_CPU_CONVERTIBLE_DEVICE_TYPES
            and device == (1, 0)
        )
        if (
            device is not None
            and device != current_device
            and not cross_host_device_request
        ):
            raise BufferError("fastCD cannot export to the requested device")
        if (
            cross_host_device_request
            and "dl_device" not in _NDARRAY_DLPACK_PARAMETERS
            and copy is False
        ):
            raise BufferError(
                "legacy NumPy cannot guarantee no-copy pinned-host export"
            )

        coefficients = self._coefficients
        supports_versioned_readonly = (
            "max_version" in _NDARRAY_DLPACK_PARAMETERS
            and version is not None
            and version >= (1, 0)
        )
        manual_copy_required = (
            (copy is True and "copy" not in _NDARRAY_DLPACK_PARAMETERS)
            or (
                cross_host_device_request
                and "dl_device" not in _NDARRAY_DLPACK_PARAMETERS
            )
            or (
                copy is None
                and not coefficients.flags.writeable
                and not supports_versioned_readonly
            )
        )
        if (
            manual_copy_required
            and current_device[0] in _DLPACK_CPU_CONVERTIBLE_DEVICE_TYPES
            and device != (1, 0)
        ):
            raise BufferError("legacy NumPy cannot copy while preserving host device")
        if manual_copy_required:
            coefficients = coefficients.copy()

        arguments: dict[str, Any] = {"stream": stream}
        if "max_version" in _NDARRAY_DLPACK_PARAMETERS:
            arguments["max_version"] = version
        if "dl_device" in _NDARRAY_DLPACK_PARAMETERS:
            arguments["dl_device"] = device
        if "copy" in _NDARRAY_DLPACK_PARAMETERS:
            arguments["copy"] = copy
        return coefficients.__dlpack__(**arguments)

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the DLPack device tuple for coefficient storage.

        Returns
        -------
        tuple[int, int]
            The host device type and device identifier reported by NumPy.
            Ordinary CPU memory reports ``(1, 0)``; imported pinned or managed
            host-accessible storage can retain its DLPack device type.
        """

        return self._coefficients.__dlpack_device__()

    def copy(self) -> ElementArray:
        """Return an independent copy of the complete coefficient array.

        Returns
        -------
        ElementArray
            A new wrapper with writable, aligned, C-contiguous storage.
        """

        return self.algebra.array(self._coefficients)

    def __copy__(self) -> ElementArray:
        """Return an independent copy for :func:`copy.copy`.

        Although this is the shallow-copy protocol, coefficient storage is
        copied because sharing it would make mutation surprising.
        """

        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> ElementArray:
        """Return an independent copy for :func:`copy.deepcopy`.

        Parameters
        ----------
        memo : dict
            Deep-copy memo populated with the newly created wrapper.
        """

        result = self.copy()
        memo[id(self)] = result
        return result

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Return owning reconstruction data for :mod:`pickle`.

        Unpickling creates fresh coefficient storage and resolves the cached
        parent Algebra.
        """

        return (_element_array_from_pickle, (self.algebra, self._coefficients))

    def to_numpy(self, *, copy: bool = True) -> np.ndarray[Any, Any]:
        """Return coefficients as a NumPy array.

        Parameters
        ----------
        copy : bool, optional
            If ``True``, return independent writable storage.  If ``False``,
            return a metadata-insulating view that shares coefficient data.

        Returns
        -------
        numpy.ndarray
            Coefficients of shape ``batch_shape + coefficient_shape``.  The
            zero-copy result keeps the source strides and writeability.

        Raises
        ------
        TypeError
            If ``copy`` is not ``True`` or ``False``.
        """

        if not isinstance(copy, bool):
            raise TypeError("copy must be True or False")
        return self._coefficients.copy() if copy else self._coefficients.view()

    def __getitem__(self, key: Any) -> Element | ElementArray:
        """Index leading batch axes while retaining complete elements.

        Parameters
        ----------
        key : object
            A NumPy-compatible index for batch axes. Complete trailing
            coefficient axes, including extension polynomial axes, are appended.

        Returns
        -------
        Element or ElementArray
            Selecting one complete record returns an owned immutable Element.
            A result retaining batch axes returns an ElementArray; ordinary
            basic slicing shares storage and advanced indexing follows NumPy.
        """

        if self._algebra._ctx is not None:
            return cast(
                "Element | ElementArray",
                _domain_array_dispatch("__getitem__", self, key),
            )

        full_key = key if isinstance(key, tuple) else (key,)
        result = self._coefficients[full_key + (slice(None),)]
        if result.ndim == 1:
            return self.algebra(result)
        return ElementArray._from_coefficients(self.algebra, result)

    def __len__(self) -> int:
        """Return the extent of the first batch axis.

        Raises
        ------
        TypeError
            If this wrapper contains one scalar record and has empty
            ``batch_shape``.
        """

        if not self.batch_shape:
            raise TypeError("a scalar ElementArray has no len()")
        return self.batch_shape[0]

    def __iter__(self) -> Iterator[Element | ElementArray]:
        """Iterate over the first batch axis.

        Yields
        ------
        Element or ElementArray
            Owned Elements when one record remains, otherwise wrappers over
            the corresponding leading-axis slices.

        Raises
        ------
        TypeError
            If this wrapper contains one scalar record and has empty
            ``batch_shape``.
        """

        if not self.batch_shape:
            raise TypeError("iteration over a scalar ElementArray")
        for index in range(self.batch_shape[0]):
            yield self[index]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Assign complete elements through leading-axis indexing.

        Parameters
        ----------
        key : object
            A NumPy-compatible index for the leading batch axes.
        value : coefficient scalar, Element, ElementArray, or array-like
            Values broadcast by NumPy into the selected records.  A coefficient
            scalar is embedded in coordinate zero, with other coordinates
            zero.

        Raises
        ------
        ValueError
            If a fastCD value has another parent, shapes cannot broadcast, or
            the backing array is read-only.
        TypeError
            If coefficients are unsupported.

        Notes
        -----
        Assignment mutates the backing ndarray and any storage shared through
        :meth:`Algebra.asarray`.
        """

        if self._algebra._ctx is not None:
            _domain_array_dispatch("__setitem__", self, key, value)
            return

        full_key = key if isinstance(key, tuple) else (key,)
        destination_key = full_key + (slice(None),)
        if isinstance(value, Element):
            self.algebra._require_element(value)
            converted = np.asarray(value)
        elif isinstance(value, ElementArray):
            if value.algebra is not self.algebra:
                raise ValueError("ElementArray belongs to a different Algebra")
            converted = value._coefficients
        elif _is_real_scalar(value):
            converted = np.zeros(self.dimension, dtype=np.float64)
            converted[0] = float(value)
        else:
            converted = self.algebra.asarray(value)._coefficients
        self._coefficients[destination_key] = converted

    def __bool__(self) -> bool:
        """Reject ambiguous truth testing.

        Raises
        ------
        ValueError
            Always.  Inspect coordinates or use an explicit reduction.
        """

        raise ValueError("the truth value of an ElementArray is ambiguous")

    def __pos__(self) -> ElementArray:
        """Return an independent copy for unary ``+``."""

        return self.copy()

    def __neg__(self) -> ElementArray:
        """Return coordinatewise negation in newly allocated storage."""

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_dispatch("__neg__", self))

        return ElementArray._from_coefficients(
            self.algebra, np.negative(self._coefficients)
        )

    def __add__(self, other: Element | ElementArray | _RealScalar) -> ElementArray:
        """Add elements or an embedded coefficient scalar.

        Element and ElementArray operands broadcast over leading axes.  A coefficient
        scalar changes coordinate zero only; nonscalar raw ndarrays are
        unsupported.  A zero-dimensional ndarray must have a scalar dtype
        accepted by the parent.
        """

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_operator("__add__", self, other))

        if type(other) is ElementArray and other._algebra is self._algebra:
            left = self._coefficients
            right = other._coefficients
            if left.shape == right.shape:
                return ElementArray._wrap(self._algebra, np.add(left, right))
        return cast(
            ElementArray,
            _coordinate_add(self, other, subtract=False, reflected=False),
        )

    def __radd__(self, other: Element | _RealScalar) -> ElementArray:
        """Add this array to an element or embedded coefficient scalar."""

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_operator("__radd__", self, other))

        return cast(
            ElementArray,
            _coordinate_add(self, other, subtract=False, reflected=True),
        )

    def __sub__(self, other: Element | ElementArray | _RealScalar) -> ElementArray:
        """Subtract elements or an embedded coefficient scalar.

        Element and ElementArray operands broadcast over leading axes.  A coefficient
        scalar changes coordinate zero only; nonscalar raw ndarrays are
        unsupported.  A zero-dimensional ndarray must have a scalar dtype
        accepted by the parent.
        """

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_operator("__sub__", self, other))

        if type(other) is ElementArray and other._algebra is self._algebra:
            left = self._coefficients
            right = other._coefficients
            if left.shape == right.shape:
                return ElementArray._wrap(self._algebra, np.subtract(left, right))
        return cast(
            ElementArray,
            _coordinate_add(self, other, subtract=True, reflected=False),
        )

    def __rsub__(self, other: Element | _RealScalar) -> ElementArray:
        """Subtract this array from an element or embedded coefficient scalar."""

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_operator("__rsub__", self, other))

        return cast(
            ElementArray,
            _coordinate_add(self, other, subtract=True, reflected=True),
        )

    def __mul__(self, other: Element | ElementArray | _RealScalar) -> ElementArray:
        """Return the pairwise Cayley--Dickson product or real scaling.

        Leading axes broadcast.  Raw coefficient arrays are not accepted by
        operators; wrap them first or use :meth:`Algebra.multiply`.
        """

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_operator("__mul__", self, other))

        if type(other) is ElementArray and other._algebra is self._algebra:
            left = self._coefficients
            right = other._coefficients
            if left.shape == right.shape:
                coefficients = np.empty(left.shape, dtype=np.float64)
                if coefficients.size:
                    _core._apply_binary_into(
                        _core._BINARY_MULTIPLY,
                        coefficients,
                        left,
                        right,
                        left.shape[-1],
                    )
                return ElementArray._wrap(self._algebra, coefficients)
        elif type(other) is Element and other.algebra is self._algebra:
            return _element_product(self, self._coefficients, other)
        return cast(
            ElementArray,
            _multiply(self.algebra, self, other, allow_array_like=False),
        )

    def __rmul__(self, other: Element | _RealScalar) -> ElementArray:
        """Left-multiply pairwise or scale by a coefficient scalar.

        Operand order is preserved because multiplication is generally
        noncommutative.
        """

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_operator("__rmul__", self, other))

        if type(other) is Element and other.algebra is self._algebra:
            return _element_product(self, other, self._coefficients)
        return cast(
            ElementArray,
            _multiply(self.algebra, other, self, allow_array_like=False),
        )

    def __truediv__(self, other: _RealScalar) -> ElementArray:
        """Divide every coefficient by a nonzero coefficient scalar.

        Parameters
        ----------
        other : coefficient scalar
            Coordinate divisor.

        Returns
        -------
        ElementArray
            Quotients in newly allocated storage.

        Raises
        ------
        TypeError
            If ``other`` is not a coefficient scalar.
        ZeroDivisionError
            If the coefficient divisor is zero.

        Notes
        -----
        Division by a complete CD record is not defined. A matching finite
        field scalar or field-scalar array may be a coefficient divisor. Use
        :meth:`reciprocal` explicitly for a CD reciprocal when intended.
        """

        if self._algebra._ctx is not None:
            return cast(
                "ElementArray", _domain_array_operator("__truediv__", self, other)
            )

        if not _is_real_scalar(other):
            return NotImplemented
        scalar = float(other)
        if scalar == 0.0:
            raise ZeroDivisionError("division by zero scalar")
        return ElementArray._from_coefficients(
            self.algebra, self._coefficients / scalar
        )

    def __pow__(self, exponent: SupportsIndex, modulus: None = None) -> ElementArray:
        """Raise each record to the same integer power.

        Parameters
        ----------
        exponent : int
            Python integer-like exponent.  Negative powers first apply the
            canonical reciprocal independently to each record.
        modulus : object, optional
            Must be omitted; modular exponentiation is undefined.

        Returns
        -------
        ElementArray
            Powers in newly allocated storage.  Exponent zero returns the
            multiplicative identity for every record.

        Raises
        ------
        TypeError
            If ``exponent`` is not integer-like or ``modulus`` is supplied.
        ZeroDivisionError
            If a negative power encounters zero quadratic norm, including
            nonzero isotropic records over complex or finite coefficients.

        Notes
        -----
        Exponentiation by squaring uses the written multiplication order;
        ``exponent == 2`` selects the specialized linear-time square kernel.
        Use ``x ** k``: Python's ``x ^ k`` is bitwise XOR and is unsupported.
        """

        if self._algebra._ctx is not None:
            return cast(
                "ElementArray",
                _domain_array_dispatch("__pow__", self, exponent, modulus),
            )

        if modulus is not None:
            raise TypeError(MODULAR_POWER_MESSAGE)
        try:
            power = operator.index(exponent)
        except TypeError as error:
            raise TypeError(INTEGER_EXPONENT_MESSAGE) from error
        if power == 0:
            shape = self.batch_shape + (self.dimension,)
            coefficients = np.zeros(shape, dtype=np.float64)
            coefficients[..., 0] = 1.0
            return ElementArray._from_coefficients(self.algebra, coefficients)
        if power == 1:
            return self.copy()
        if power == 2:
            return self.square()
        base = self.reciprocal() if power < 0 else self.copy()
        power = abs(power)
        result: ElementArray | None = None
        while power:
            if power & 1:
                result = base.copy() if result is None else result * base
            power >>= 1
            if power:
                base = base.square()
        assert result is not None
        return result

    def __abs__(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return real Euclidean norms; complex/finite parents raise TypeError."""

        return self.norm()

    # Describe array-valued equality for supported fastCD and scalar operands
    @overload
    def __eq__(  # type: ignore[overload-overlap]
        self, other: Element | ElementArray | _RealScalar
    ) -> np.ndarray[Any, np.dtype[np.bool_]]: ...

    # Describe Boolean equality for arbitrary unsupported operands
    @overload
    def __eq__(self, other: object) -> bool: ...

    def __eq__(self, other: Any) -> Any:
        """Return exact whole-element equality over broadcast batch axes.

        The result is a Boolean ndarray of the broadcast batch shape.  A coefficient
        scalar compares as an element with zero imaginary coordinates.
        """

        if self._algebra._ctx is not None:
            return _domain_array_dispatch("__eq__", self, other)

        return _coordinate_equal(self, other, invert=False)

    # Describe array-valued inequality for supported fastCD and scalar operands
    @overload
    def __ne__(  # type: ignore[overload-overlap]
        self, other: Element | ElementArray | _RealScalar
    ) -> np.ndarray[Any, np.dtype[np.bool_]]: ...

    # Describe Boolean inequality for arbitrary unsupported operands
    @overload
    def __ne__(self, other: object) -> bool: ...

    def __ne__(self, other: Any) -> Any:
        """Return exact whole-element inequality over broadcast batch axes."""

        if self._algebra._ctx is not None:
            return _domain_array_dispatch("__ne__", self, other)

        return _coordinate_equal(self, other, invert=True)

    def conjugate(self) -> ElementArray:
        """Return standard conjugates for all batch records.

        Returns
        -------
        ElementArray
            Newly allocated records with coordinate zero preserved and every
            imaginary coordinate negated.

        Notes
        -----
        The CD involution fixes coefficient scalars, including complex values.
        It does not apply coefficient complex conjugation or finite-field Frobenius.
        """

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_dispatch("conjugate", self))

        return _unary(self.algebra, self, _core._UNARY_CONJUGATE)

    def imag_part(self) -> ElementArray:
        """Return imaginary parts for all batch records.

        Returns
        -------
        ElementArray
            Newly allocated records with coordinate zero replaced by zero and
            all remaining coordinates preserved.
        """

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_dispatch("imag_part", self))

        return _unary(self.algebra, self, _core._UNARY_IMAG_PART)

    def normalized(self) -> ElementArray:
        """Normalize every batch element independently.

        Returns
        -------
        ElementArray
            Newly allocated records divided by their Euclidean norms.

        Raises
        ------
        TypeError
            If coefficients are complex or finite; normalization requires R.
        ZeroDivisionError
            If any visited record has all coefficients exactly zero.  No
            near-zero tolerance is applied.

        Notes
        -----
        Scaled norm accumulation reduces avoidable finite overflow and
        underflow.  NaN and infinity follow the native IEEE-754 path.
        """

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_dispatch("normalized", self))

        return _unary(self.algebra, self, _core._UNARY_NORMALIZE)

    def reciprocal(self) -> ElementArray:
        """Return the canonical reciprocal of each batch record.

        Returns
        -------
        ElementArray
            Newly allocated values ``conjugate(x) / norm_squared(x)``.

        Raises
        ------
        ZeroDivisionError
            If any record has zero quadratic norm, including nonzero isotropic
            complex/finite records. No near-zero tolerance is applied.

        Notes
        -----
        In higher nonassociative algebras this canonical two-sided reciprocal
        is not a solver for a general left- or right-multiplication equation.
        """

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_dispatch("reciprocal", self))

        return _unary(self.algebra, self, _core._UNARY_RECIPROCAL)

    def square(self) -> ElementArray:
        """Square every record with the specialized linear-time kernel.

        Returns
        -------
        ElementArray
            Newly allocated elementwise squares.

        Notes
        -----
        For ``x = alpha + u``, the kernel evaluates
        ``x**2 = (alpha**2 - inner(u, u)) + 2*alpha*u``.  This schedule may
        round differently from general pairwise multiplication.
        """

        if self._algebra._ctx is not None:
            return cast("ElementArray", _domain_array_dispatch("square", self))

        return _unary(self.algebra, self, _core._UNARY_SQUARE)

    def inner(self, other: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return pairwise coefficient-bilinear products ``sum_i x_i*y_i``.

        Parameters
        ----------
        other : Element or ElementArray
            Same-parent operand.  Leading batch axes are broadcast.

        Returns
        -------
        numpy.ndarray or ElementArray
            Real/complex ndarrays over the broadcast batch shape, or a
            finite-field ElementArray over the matching coefficient parent.

        Raises
        ------
        TypeError
            If ``other`` is not an Element or ElementArray.
        ValueError
            If its parent or leading batch shape is incompatible.

        Notes
        -----
        The real path uses sequential double arithmetic, without compensation.
        Complex coefficients are not conjugated; finite accumulation is exact
        in the coefficient field. Wrap raw inputs with ``algebra.asarray``.
        """

        if self._algebra._ctx is not None:
            return cast(
                "np.ndarray[Any, np.dtype[np.float64]]",
                _domain_array_operator("inner", self, other),
            )

        return _scalar_binary(self.algebra, self, other, _core._SCALAR_INNER)

    def norm_squared(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return the quadratic form of every record over the batch shape.

        Returns
        -------
        numpy.ndarray or ElementArray
            One ``sum_i x_i**2`` in the coefficient field per record:
            float64 over R, complex128 over C, or a finite-field ElementArray.

        Notes
        -----
        This is the same form as ``quadratic_norm()``. Over R it uses scaled
        accumulation to reduce avoidable overflow and underflow, with
        IEEE-754 NaN/infinity behavior. Complex and finite
        forms can vanish on nonzero records and are not Hermitian norms.
        """

        if self._algebra._ctx is not None:
            return cast(
                "np.ndarray[Any, np.dtype[np.float64]]",
                _domain_array_dispatch("norm_squared", self),
            )

        return _scalar_unary(self.algebra, self, _core._SCALAR_NORM_SQUARED)

    def quadratic_norm(self) -> Any:
        """Return the quadratic form ``sum_i x_i**2`` in the coefficient field."""

        return self.norm_squared()

    def norm(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return Euclidean norms over the batch shape.

        Returns
        -------
        numpy.ndarray
            One native ``float64`` norm per batch record; this is also the
            value returned by ``abs(self)``.

        Raises
        ------
        TypeError
            For complex or finite parents; use ``quadratic_norm()`` there.
        """

        if self._algebra._ctx is not None:
            return cast(
                "np.ndarray[Any, np.dtype[np.float64]]",
                _domain_array_dispatch("norm", self),
            )

        return _scalar_unary(self.algebra, self, _core._SCALAR_NORM)

    def imag_cross(self, other: Any) -> ElementArray:
        """Return pairwise alternating products of the imaginary parts.

        Parameters
        ----------
        other : Element or ElementArray
            Same-parent operand.  Leading batch axes are broadcast.

        Returns
        -------
        ElementArray
            New records equal to ``(X*Y - Y*X)/2``, where ``X`` and ``Y``
            are the imaginary parts.  Coordinate zero is zero.

        Raises
        ------
        TypeError
            If ``other`` is not an Element or ElementArray.
        ValueError
            If its parent or leading batch shape is incompatible.

        Notes
        -----
        Scalar coordinates are ignored.  At dimension four this matches the
        ordinary three-dimensional vector cross product under fastCD's basis
        orientation; higher-dimensional vector identities need not hold.
        """

        if self._algebra._ctx is not None:
            return cast(
                "ElementArray", _domain_array_operator("imag_cross", self, other)
            )

        return _binary(self.algebra, self, other, _core._BINARY_IMAG_CROSS)

    def commutator(self, other: Any) -> ElementArray:
        """Return pairwise algebra commutators.

        Parameters
        ----------
        other : Element or ElementArray
            Same-parent operand.  Leading batch axes are broadcast.

        Returns
        -------
        ElementArray
            New records ``self*other - other*self``.

        Raises
        ------
        TypeError
            If ``other`` is not an Element or ElementArray.
        ValueError
            If its parent or leading batch shape is incompatible.

        Notes
        -----
        The optimized matrix-free alternating-product kernel does not form
        and subtract two independent complete products.
        """

        if self._algebra._ctx is not None:
            return cast(
                "ElementArray", _domain_array_operator("commutator", self, other)
            )

        return _binary(self.algebra, self, other, _core._BINARY_COMMUTATOR)

    def anticommutator(self, other: Any) -> ElementArray:
        """Return pairwise algebra anticommutators.

        Parameters
        ----------
        other : Element or ElementArray
            Same-parent operand.  Leading batch axes are broadcast.

        Returns
        -------
        ElementArray
            New records ``self*other + other*self``.

        Raises
        ------
        TypeError
            If ``other`` is not an Element or ElementArray.
        ValueError
            If its parent or leading batch shape is incompatible.

        Notes
        -----
        The native kernel uses a linear-time scalar--imaginary identity rather
        than forming two complete products.
        """

        if self._algebra._ctx is not None:
            return cast(
                "ElementArray", _domain_array_operator("anticommutator", self, other)
            )

        return _binary(self.algebra, self, other, _core._BINARY_ANTICOMMUTATOR)

    def associator(self, y: Any, z: Any) -> ElementArray:
        """Return pairwise Cayley--Dickson associators.

        Parameters
        ----------
        y, z : Element or ElementArray
            Same-parent operands.  All leading batch axes are broadcast.

        Returns
        -------
        ElementArray
            New records ``(self*y)*z - self*(y*z)``.

        Raises
        ------
        TypeError
            If either operand is not an Element or ElementArray.
        ValueError
            If a parent or leading batch shape is incompatible.

        Notes
        -----
        The displayed parenthesization is part of the numerical contract once
        multiplication is nonassociative.  Wrap raw arrays before calling.
        """

        if self._algebra._ctx is not None:
            return cast(
                "ElementArray", _domain_array_dispatch("associator", self, y, z)
            )

        return _associator(self.algebra, self, y, z)

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Dispatch the intentional NumPy ufunc subset algebraically.

        Supported ``__call__`` ufuncs are ``add``, ``subtract``, ``true_divide``,
        ``power``, ``negative``, ``positive``, ``multiply``, ``square``,
        ``conjugate``, ``absolute``, ``equal``, and ``not_equal``.
        Multiplication remains Cayley--Dickson multiplication,
        never coordinatewise multiplication.  Only ``multiply`` accepts
        ``out=``; it uses :meth:`Algebra.multiply`'s exact-shape,
        C-contiguous, nonoverlapping output contract.

        Unsupported ufuncs, nonscalar raw ndarray operands, methods such as
        ``reduce``, and extra keyword options return ``NotImplemented`` and
        normally cause NumPy to raise ``TypeError``.  Compatible zero-dimensional
        ndarrays are coefficient scalars. CD conjugation fixes coefficients;
        ``absolute`` raises TypeError for complex/finite parents, as does abs().
        """

        if self._algebra._ctx is not None:
            return _domain_array_dispatch(
                "__array_ufunc__", self, ufunc, method, *inputs, **kwargs
            )

        return _operators._dispatch_array_ufunc(self, ufunc, method, inputs, kwargs)


def _element_product(array: ElementArray, left: Any, right: Any) -> ElementArray:
    """Multiply every record of a real array by one Element of its algebra."""

    shape = array._coefficients.shape
    coefficients = np.empty(shape, dtype=np.float64)
    if coefficients.size:
        _core._apply_binary_into(
            _core._BINARY_MULTIPLY, coefficients, left, right, shape[-1]
        )
    return ElementArray._wrap(array._algebra, coefficients)


def _element_array_from_pickle(algebra: Algebra, coefficients: Any) -> ElementArray:
    """Reconstruct an owning element array during unpickling."""

    return algebra.array(coefficients)


def _strict_operand(algebra: Algebra, value: Any) -> np.ndarray[Any, Any] | None:
    """Return coefficients for a same-parent fastCD operand when supported."""

    if isinstance(value, ElementArray):
        if value.algebra is not algebra:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
        return value._coefficients
    if isinstance(value, Element):
        algebra._require_element(value)
        return np.asarray(value)
    return None


def _general_operand(algebra: Algebra, value: Any) -> np.ndarray[Any, Any]:
    """Coerce a same-parent fastCD value or general array-like operand."""

    strict = _strict_operand(algebra, value)
    if strict is not None:
        return strict
    return _coerce_coefficients(algebra, value, copy=None)


def _broadcast_coefficients(
    algebra: Algebra, *operands: np.ndarray[Any, Any]
) -> tuple[tuple[int, ...], list[np.ndarray[Any, Any]]]:
    """Broadcast batch axes while preserving the complete coefficient axis.

    The native kernels and NumPy repeat an operand that holds one record, so
    operands are expanded here only when two different batch shapes must be
    combined.
    """

    batch_shape: tuple[int, ...] = ()
    for value in operands:
        candidate = value.shape[:-1]
        if candidate:
            if batch_shape and candidate != batch_shape:
                break
            batch_shape = candidate
    else:
        return batch_shape, list(operands)
    batch_shape = np.broadcast_shapes(*(value.shape[:-1] for value in operands))
    full_shape = batch_shape + (algebra.dimension,)
    return batch_shape, [np.broadcast_to(value, full_shape) for value in operands]


def _element_output(
    algebra: Algebra,
    full_shape: tuple[int, ...],
    out: Any | None,
    inputs: tuple[np.ndarray[Any, Any], ...],
) -> tuple[np.ndarray[Any, Any], ElementArray]:
    """Allocate or validate storage for an element-valued result.

    Inputs are read-only and may share storage with each other; the native
    layer rejects only record pairs that partially overlap.
    """

    if out is None:
        coefficients = np.empty(full_shape, dtype=np.float64)
        return coefficients, ElementArray._wrap(algebra, coefficients)
    if isinstance(out, ElementArray):
        if out.algebra is not algebra:
            raise ValueError("out belongs to a different algebra")
        coefficients = out._coefficients
        result = out
    elif isinstance(out, np.ndarray):
        coefficients = out
        result = ElementArray._from_coefficients(algebra, coefficients)
    else:
        raise TypeError("out must be an ElementArray or numpy.ndarray")
    if coefficients.shape != full_shape:
        raise ValueError(f"out must have shape {full_shape}")
    if coefficients.dtype != np.dtype(np.float64) or not coefficients.dtype.isnative:
        raise TypeError("out must have native float64 dtype")
    if not coefficients.flags.writeable or not coefficients.flags.aligned:
        raise ValueError("out must be writable and aligned")
    if not coefficients.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")
    if any(np.shares_memory(coefficients, source) for source in inputs):
        raise ValueError("out must not overlap an input")
    return coefficients, result


def _unary(algebra: Algebra, value: Any, operation: int) -> ElementArray:
    """Apply one native unary operation to an array-like coefficient batch."""

    source = _general_operand(algebra, value)
    coefficients = np.empty(source.shape, dtype=np.float64)
    if coefficients.size:
        _core._apply_unary_into(operation, coefficients, source, algebra.dimension)
    return ElementArray._wrap(algebra, coefficients)


def _binary(
    algebra: Algebra,
    left: Any,
    right: Any,
    operation: int,
    *,
    out: Any | None = None,
    allow_array_like: bool = False,
    algorithm: int | None = None,
) -> ElementArray:
    """Apply one native binary operation over broadcast coefficient batches.

    ``algorithm`` is a validated multiplication algorithm code from
    ``_core._algorithm``; it selects the explicit-algorithm multiplication
    kernel and is meaningful only for ``_core._BINARY_MULTIPLY``.
    """

    operand = _general_operand if allow_array_like else _strict_operand
    left_values = operand(algebra, left)
    right_values = operand(algebra, right)
    if left_values is None or right_values is None:
        raise TypeError("array arithmetic requires Element or ElementArray operands")
    batch_shape, (left_broadcast, right_broadcast) = _broadcast_coefficients(
        algebra, left_values, right_values
    )
    coefficients, result = _element_output(
        algebra,
        batch_shape + (algebra.dimension,),
        out,
        (left_broadcast, right_broadcast),
    )
    if not coefficients.size:
        return result
    if algorithm is None:
        _core._apply_binary_into(
            operation,
            coefficients,
            left_broadcast,
            right_broadcast,
            algebra.dimension,
        )
    else:
        _core._apply_multiply_into(
            algorithm,
            coefficients,
            left_broadcast,
            right_broadcast,
            algebra.dimension,
        )
    return result


def _multiply(
    algebra: Algebra,
    left: Any,
    right: Any,
    *,
    out: Any | None = None,
    allow_array_like: bool = False,
    algorithm: int | None = None,
) -> Any:
    """Dispatch real scaling or pairwise Cayley--Dickson multiplication.

    ``algorithm`` is a validated code from ``_core._algorithm``; ``None``
    keeps automatic selection.  Real scaling ignores it.
    """

    if _is_real_scalar(left):
        right_values = (
            _general_operand(algebra, right)
            if allow_array_like
            else _strict_operand(algebra, right)
        )
        if right_values is None:
            return NotImplemented
        if out is not None:
            coefficients, result = _element_output(
                algebra, right_values.shape, out, (right_values,)
            )
            np.multiply(right_values, float(left), out=coefficients)
            return result
        return ElementArray._from_coefficients(algebra, right_values * float(left))
    if _is_real_scalar(right):
        left_values = (
            _general_operand(algebra, left)
            if allow_array_like
            else _strict_operand(algebra, left)
        )
        if left_values is None:
            return NotImplemented
        if out is not None:
            coefficients, result = _element_output(
                algebra, left_values.shape, out, (left_values,)
            )
            np.multiply(left_values, float(right), out=coefficients)
            return result
        return ElementArray._from_coefficients(algebra, left_values * float(right))
    if not allow_array_like and not (
        isinstance(left, (Element, ElementArray))
        and isinstance(right, (Element, ElementArray))
    ):
        return NotImplemented
    return _binary(
        algebra,
        left,
        right,
        _core._BINARY_MULTIPLY,
        out=out,
        allow_array_like=allow_array_like,
        algorithm=algorithm,
    )


def _coordinate_add(
    array: ElementArray, other: Any, *, subtract: bool, reflected: bool
) -> Any:
    """Add or subtract an element array, element, or embedded real scalar."""

    if _is_real_scalar(other):
        coefficients = array._coefficients.copy()
        scalar = float(other)
        if reflected and subtract:
            coefficients *= -1.0
            coefficients[..., 0] += scalar
        else:
            coefficients[..., 0] += -scalar if subtract else scalar
        return ElementArray._from_coefficients(array.algebra, coefficients)
    other_values = _strict_operand(array.algebra, other)
    if other_values is None:
        return NotImplemented
    _batch_shape, (left, right) = _broadcast_coefficients(
        array.algebra,
        other_values if reflected else array._coefficients,
        array._coefficients if reflected else other_values,
    )
    coefficients = np.subtract(left, right) if subtract else np.add(left, right)
    return ElementArray._from_coefficients(array.algebra, coefficients)


def _coordinate_equal(array: ElementArray, other: Any, *, invert: bool) -> Any:
    """Compare complete broadcast elements for equality or inequality."""

    if _is_real_scalar(other):
        scalar = _core._comparison_scalar(other)
        if scalar is NotImplemented:
            return NotImplemented
        real = array._coefficients[..., 0]
        # NumPy's numeric promotion can round large integers or narrow floats,
        # so object comparison uses Python's exact numeric comparisons
        values = real if isinstance(scalar, float) else real.astype(object)
        comparison = values == scalar
        if array.dimension > 1:
            comparison &= np.all(array._coefficients[..., 1:] == 0.0, axis=-1)
    elif (
        isinstance(other, (Element, ElementArray))
        and other.algebra is not array.algebra
    ):
        # As NumPy does for incomparable operands, every pair is unequal
        other_shape = other.batch_shape if isinstance(other, ElementArray) else ()
        shape = np.broadcast_shapes(array.batch_shape, other_shape)
        comparison = np.zeros(shape, dtype=bool)
    else:
        other_values = _strict_operand(array.algebra, other)
        if other_values is None:
            return NotImplemented
        _batch_shape, (left, right) = _broadcast_coefficients(
            array.algebra, array._coefficients, other_values
        )
        comparison = (left == right).all(axis=-1)
    result = np.logical_not(comparison) if invert else comparison
    return np.asarray(result)


def _scalar_unary(algebra: Algebra, value: Any, operation: int) -> np.ndarray[Any, Any]:
    """Apply a native scalar-valued unary operation over a coefficient batch."""

    source = _general_operand(algebra, value)
    result = np.empty(source.shape[:-1], dtype=np.float64)
    if result.size:
        _core._apply_scalar_into(operation, result, source, None, algebra.dimension)
    return result


def _scalar_binary(
    algebra: Algebra, left: Any, right: Any, operation: int
) -> np.ndarray[Any, Any]:
    """Apply a native scalar-valued binary operation over broadcast batches."""

    left_values = _strict_operand(algebra, left)
    right_values = _strict_operand(algebra, right)
    if left_values is None or right_values is None:
        raise TypeError("array arithmetic requires Element or ElementArray operands")
    batch_shape, (left_broadcast, right_broadcast) = _broadcast_coefficients(
        algebra, left_values, right_values
    )
    result = np.empty(batch_shape, dtype=np.float64)
    if result.size:
        _core._apply_scalar_into(
            operation, result, left_broadcast, right_broadcast, algebra.dimension
        )
    return result


def _associator(algebra: Algebra, x: Any, y: Any, z: Any) -> ElementArray:
    """Apply the native associator over three broadcast fastCD operands."""

    operands = [_strict_operand(algebra, value) for value in (x, y, z)]
    if any(value is None for value in operands):
        raise TypeError("array arithmetic requires Element or ElementArray operands")
    operand_arrays = cast(list[np.ndarray[Any, Any]], operands)
    batch_shape, broadcast = _broadcast_coefficients(algebra, *operand_arrays)
    coefficients = np.empty(batch_shape + (algebra.dimension,), dtype=np.float64)
    if coefficients.size:
        _core._apply_associator_into(
            coefficients,
            broadcast[0],
            broadcast[1],
            broadcast[2],
            algebra.dimension,
        )
    return ElementArray._wrap(algebra, coefficients)


def _basis_action(
    algebra: Algebra, index: int, x: Any, *, left: bool, out: Any | None
) -> ElementArray:
    """Apply left or right basis multiplication to an array-like batch."""

    source = _general_operand(algebra, x)
    coefficients, result = _element_output(algebra, source.shape, out, (source,))
    if coefficients.size:
        _core._apply_basis_into(left, index, coefficients, source, algebra.dimension)
    return result


def _operator_commutator(
    algebra: Algebra,
    a: Any,
    b: Any,
    x: Any,
    *,
    left: bool,
    out: Any | None,
) -> ElementArray:
    """Apply a multiplication-operator commutator to broadcast batches."""

    operands = [_general_operand(algebra, value) for value in (a, b, x)]
    batch_shape, broadcast = _broadcast_coefficients(algebra, *operands)
    coefficients, result = _element_output(
        algebra, batch_shape + (algebra.dimension,), out, tuple(broadcast)
    )
    if coefficients.size:
        _core._apply_operator_into(
            left,
            coefficients,
            broadcast[0],
            broadcast[1],
            broadcast[2],
            algebra.dimension,
        )
    return result


# The ufunc dispatcher imports ElementArray while it loads, so it is bound
# here only after the class exists
from . import _operators  # noqa: E402
