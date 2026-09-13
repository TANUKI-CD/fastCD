"""Parents and constructors for standard real Cayley--Dickson algebras.

The public entry points in this module are re-exported from :mod:`pyfastcd`;
applications normally use ``import pyfastcd as fcd``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, final

import numpy as np

from . import _core
from ._core import Element
from ._typing import _RealScalar
from ._validation import (
    COMPLEX_SCALAR_MESSAGE,
    DIFFERENT_ALGEBRA_MESSAGE,
    _basis_index,
    _dimension_and_level,
)

if TYPE_CHECKING:
    from ._array import ElementArray


@dataclass(frozen=True)
class BasisProduct:
    """The signed result of multiplying two standard basis elements.

    If ``p = A.basis_product(i, j)``, then
    ``A.basis(i) * A.basis(j) == p.sign * A.basis(p.index)``.

    Attributes
    ----------
    index : int
        Index ``k`` of the resulting standard basis element.
    sign : int
        For values returned by :meth:`Algebra.basis_product`, either ``1`` or
        ``-1`` such that ``e_i*e_j = sign*e_k``.  Direct construction does
        not validate the fields.

    Notes
    -----
    Instances are immutable and hashable, compare by value, and unpack as
    ``index, sign``.
    """

    index: int
    sign: int

    def __iter__(self) -> Iterator[int]:
        """Unpack as ``(index, sign)``."""

        yield self.index
        yield self.sign


@final
class Algebra:
    """A standard real Cayley--Dickson algebra with ``float64`` coefficients.

    Parameters
    ----------
    dimension : int, optional
        Positive power-of-two real dimension.  Boolean values are rejected.
    level : int, optional
        Nonnegative doubling level ``n``, giving dimension ``2**n``.
        Boolean values are rejected.

    Raises
    ------
    TypeError
        If neither or both keywords are supplied, or if the selected value
        is not integer-like.
    ValueError
        If ``dimension`` is not a positive power of two or ``level`` is
        negative.
    OverflowError
        If one element's coefficient storage would exceed Python's addressable
        container size.

    Notes
    -----
    Exactly one of ``dimension`` and ``level`` must be supplied.  Algebra
    objects are immutable, cannot be subclassed, and are cached by dimension,
    so repeated construction returns the same parent object.  All public
    arithmetic uses the standard real tower and native ``float64``
    coefficients.

    Examples
    --------
    >>> import pyfastcd as fcd
    >>> H = fcd.Algebra(dimension=4)
    >>> H([1, 2, 3, 4])
    Element([1.0, 2.0, 3.0, 4.0], algebra=Algebra(dimension=4, level=2, dtype=float64))
    """

    __slots__ = ("_dimension", "_level", "_initialized")
    _dimension: int
    _level: int
    _initialized: bool
    _cache: dict[int, Algebra] = {}

    def __new__(
        cls, *, dimension: Any | None = None, level: Any | None = None
    ) -> Algebra:
        """Create or retrieve the cached parent selected by one keyword.

        Validation, immutable-state initialization, and dimension caching are
        performed here.  See :class:`Algebra` for the full parameter and
        exception contract.
        """

        if cls is not Algebra:
            raise TypeError("Algebra cannot be subclassed")
        normalized_dimension, normalized_level = _dimension_and_level(
            dimension=dimension, level=level
        )
        cached = cls._cache.get(normalized_dimension)
        if cached is not None:
            return cached
        self = super().__new__(cls)
        object.__setattr__(self, "_dimension", normalized_dimension)
        object.__setattr__(self, "_level", normalized_level)
        object.__setattr__(self, "_initialized", True)
        return cls._cache.setdefault(normalized_dimension, self)

    def __init__(
        self, *, dimension: Any | None = None, level: Any | None = None
    ) -> None:
        """Complete construction without changing cached parent state.

        ``__new__`` performs validation and initialization.  This no-op is
        called again when construction returns a cached Algebra.
        """

        # Validation and initialization are performed in __new__.  The no-op
        # __init__ is invoked again when a cached parent is returned.
        del dimension, level

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject creation of Algebra subclasses."""

        del cls, kwargs
        raise TypeError("Algebra cannot be subclassed")

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject mutation of an Algebra.

        Raises
        ------
        AttributeError
            Always, because parent objects are immutable.
        """

        raise AttributeError("Algebra objects are immutable")

    def __delattr__(self, name: str) -> None:
        """Reject deletion of Algebra state.

        Raises
        ------
        AttributeError
            Always, because cached parent objects are immutable.
        """

        del name
        raise AttributeError("Algebra objects are immutable")

    @property
    def dimension(self) -> int:
        """int: Number of real coefficients in each element."""

        return self._dimension

    @property
    def level(self) -> int:
        """int: Doubling level ``n`` satisfying ``dimension == 2**n``."""

        return self._level

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """numpy.dtype: Coefficient dtype, always native ``float64``."""

        return np.dtype(np.float64)

    @property
    def default_algorithm(self) -> str:
        """str: Algorithm that ``"auto"`` selects for a single product.

        One of ``"direct"``, ``"cariow_cariowa"``, or ``"quasilinear"``; the
        selection depends only on the dimension.  Products of two
        :class:`Element` objects, and batches containing exactly one record,
        use this algorithm, so passing the name explicitly to
        :meth:`multiply` reproduces ``x * y`` exactly for elements.  Batches
        of two or more records use :attr:`default_batch_algorithm`.
        """

        return _core._default_algorithm(self.dimension)

    @property
    def default_batch_algorithm(self) -> str:
        """str: Algorithm that ``"auto"`` selects for batches of two or more.

        One of ``"direct"``, ``"cariow_cariowa"``, or ``"quasilinear"``.
        Batches evaluate two independent products per SIMD instruction where
        the build supports it, which can change the fastest algorithm below
        dimension 32; both selections come from the library benchmark.
        Passing this name explicitly to :meth:`multiply` reproduces ``X * Y``
        exactly for arrays with at least two records.
        """

        return _core._default_batch_algorithm(self.dimension)

    def __repr__(self) -> str:
        """Return a constructor-oriented description of this parent."""

        return (
            f"Algebra(dimension={self.dimension}, level={self.level}, "
            "dtype=float64)"
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Return reconstruction data that resolves through the parent cache."""

        return (_algebra_from_dimension, (self.dimension,))

    def __call__(self, value: Any = 0.0) -> Element:
        """Construct one owned element of this algebra.

        Parameters
        ----------
        value : real scalar, Element, or array-like, optional
            A scalar is embedded in coordinate zero.  A coefficient input
            must be one-dimensional and contain exactly ``dimension`` real
            values.  A same-dimension :class:`Element` is also accepted.

        Returns
        -------
        Element
            A newly allocated immutable element belonging to this algebra.

        Raises
        ------
        ValueError
            If a coefficient input has the wrong shape or length, or an
            element has a different dimension.
        TypeError
            If the input contains complex, object, or nonnumeric values.
        OverflowError
            If a coefficient cannot be represented as a Python float.

        Notes
        -----
        Construction always copies coefficient data.  Later mutation of a
        NumPy input cannot change the returned element.  A scalar ``r`` is
        interpreted as ``r * e_0``, not repeated across all coordinates.
        """

        if isinstance(value, memoryview):
            try:
                value = np.asarray(value)
            except NotImplementedError as error:
                raise TypeError("unsupported coefficient buffer format") from error
        if isinstance(value, (complex, np.complexfloating)):
            raise TypeError(COMPLEX_SCALAR_MESSAGE)
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf":
                raise TypeError(
                    "coefficient arrays must contain real numeric values"
                )
        return _core._element(self, self.dimension, value)

    def scalar(self, value: _RealScalar) -> Element:
        """Return ``value`` embedded in the scalar coordinate.

        Parameters
        ----------
        value : scalar
            Noncomplex scalar accepted by :class:`float`, placed at coordinate
            zero.  A zero-dimensional ndarray must have a real numeric dtype;
            object, string, and structured ndarrays are rejected.

        Returns
        -------
        Element
            A newly allocated scalar element ``value * e_0``.

        Raises
        ------
        TypeError
            If ``value`` is complex or nonscalar.
        ValueError, OverflowError
            If :class:`float` cannot convert or represent ``value``.
        """

        if isinstance(
            value, (complex, np.complexfloating, str, bytes, bytearray, memoryview)
        ):
            raise TypeError(COMPLEX_SCALAR_MESSAGE)
        if isinstance(value, np.ndarray):
            if value.ndim != 0 or value.dtype.kind not in "biuf":
                raise TypeError("scalar must be a real scalar value")
        return _core._scalar(self, self.dimension, float(value))

    def zero(self) -> Element:
        """Return the additive identity as a new element.

        Returns
        -------
        Element
            A newly allocated element whose coefficients are all zero.
        """

        return _core._scalar(self, self.dimension, 0.0)

    def one(self) -> Element:
        """Return the multiplicative identity as a new element.

        Returns
        -------
        Element
            A newly allocated copy of ``e_0``.
        """

        return _core._scalar(self, self.dimension, 1.0)

    def basis(self, index: Any) -> Element:
        """Return one standard recursive basis element ``e_index``.

        Parameters
        ----------
        index : int
            Basis index in ``range(dimension)``.

        Returns
        -------
        Element
            A newly allocated element with coefficient one at ``index`` and
            zero elsewhere.

        Raises
        ------
        TypeError
            If ``index`` is not integer-like or is a Boolean.
        IndexError
            If ``index`` is outside the standard basis.
        """

        normalized = _basis_index(index, self.dimension)
        return _core._basis(self, self.dimension, normalized)

    def basis_product(self, left_index: Any, right_index: Any) -> BasisProduct:
        """Return the signed standard-basis product.

        The result ``p`` satisfies
        ``e_left_index * e_right_index == p.sign * e_(p.index)``.

        Parameters
        ----------
        left_index, right_index : int
            Basis indices in ``range(dimension)``.

        Returns
        -------
        BasisProduct
            The destination basis index and its sign.  No element or dense
            multiplication table is constructed.

        Raises
        ------
        TypeError
            If either index is not integer-like or is a Boolean.
        IndexError
            If either index is outside the standard basis.

        Notes
        -----
        Under the recursive basis convention, the destination index is
        ``left_index ^ right_index``; the sign is computed matrix-free.
        """

        left = _basis_index(left_index, self.dimension)
        right = _basis_index(right_index, self.dimension)
        index, sign = _core._basis_product(self.dimension, left, right)
        return BasisProduct(index=index, sign=sign)

    def array(self, values: Any) -> ElementArray:
        """Create an owning, C-contiguous array of algebra elements.

        Parameters
        ----------
        values : Element, ElementArray, or array-like
            Real coefficients with shape ``(..., dimension)``.

        Returns
        -------
        ElementArray
            A mutable wrapper around newly allocated, writable, aligned,
            C-contiguous native ``float64`` storage.

        Raises
        ------
        ValueError
            If the final axis has the wrong length or an existing fastCD
            object belongs to another algebra.
        TypeError
            If values are complex, object-valued, or nonnumeric.

        Notes
        -----
        Coefficients are always copied, even when ``values`` already has a
        compatible layout.  The final axis stores one algebra element; all
        preceding axes are batch axes.
        """

        from ._array import _array_from_values

        return _array_from_values(self, values, copy=True)

    def asarray(self, values: Any, *, copy: bool | None = None) -> ElementArray:
        """Interpret an array with final axis ``dimension`` as elements.

        Parameters
        ----------
        values : Element, ElementArray, array-like, or DLPack producer
            Real coefficients with shape ``(..., dimension)``.  DLPack input
            must use a supported host-accessible device.
        copy : bool or None, optional
            ``True`` always copies. ``None`` copies only when dtype, alignment,
            or final-axis layout requires it. ``False`` raises if zero-copy
            wrapping is impossible.

        Returns
        -------
        ElementArray
            A wrapper whose storage may be shared with ``values``.  Source
            leading-axis strides are preserved on the zero-copy path.  Direct
            NumPy and buffer sources also preserve writeability; legacy
            DLPack imports can expose a read-only view of writable storage.

        Raises
        ------
        TypeError
            If ``copy`` is not ``True``, ``False``, or ``None``, or values
            have an unsupported dtype.
        ValueError
            If the final axis is wrong, a fastCD parent differs, a DLPack
            input is not on a supported host-accessible device, or
            ``copy=False`` cannot be honored.

        Notes
        -----
        Zero-copy storage must be aligned native ``float64`` with a unit
        positive stride on the final axis.  For the one-dimensional real
        algebra only, a zero final-axis stride is also valid because that
        axis contains a single coefficient.  C-contiguous arrays with an
        empty leading batch axis are also valid when NumPy reports zero
        strides, because no coefficient is visited.  A view created from an
        immutable :class:`Element` remains read-only.
        """

        from ._array import _array_from_values

        return _array_from_values(self, values, copy=copy)

    def multiply(
        self,
        x: Any,
        y: Any,
        *,
        out: Any | None = None,
        algorithm: str = "auto",
    ) -> Element | ElementArray:
        """Return the Cayley--Dickson product of elements or element arrays.

        Parameters
        ----------
        x, y : real scalar, Element, ElementArray, or array-like
            Factors.  A real scalar scales every coordinate.  Nonscalar
            coefficient arrays must have final axis ``dimension``.
        out : ElementArray or numpy.ndarray, optional
            Exact-shape output storage.  It must be writable, aligned,
            C-contiguous native ``float64`` and must not overlap an input.
        algorithm : str, optional
            Multiplication algorithm name: ``"auto"`` (the default),
            ``"direct"``, ``"cariow_cariowa"``, or ``"quasilinear"``.
            ``"auto"`` selects :attr:`default_algorithm` for a single
            product and :attr:`default_batch_algorithm` for batches of two
            or more records.  The name is validated before any arithmetic,
            including for real-scalar operands and empty batches.

        Returns
        -------
        Element or ElementArray
            With no ``out``, scalar and :class:`Element` operands produce an
            :class:`Element`; any batched or array-like operand produces an
            :class:`ElementArray`.  When ``out`` is an ElementArray, that same
            wrapper is returned; an ndarray is returned inside a new wrapper.

        Raises
        ------
        TypeError
            If an operand or output object is unsupported, or if
            ``algorithm`` is not a string.
        ValueError
            If ``algorithm`` is not a known name or cannot multiply elements
            of this dimension (Cariow--Cariowa kernels exist only through
            dimension 1024), or if parents, shapes, broadcasting, output layout,
            or output/input separation are incompatible.

        Notes
        -----
        Leading batch axes follow NumPy broadcasting.  The final coefficient
        axis is fixed and is never broadcast.  ``"direct"`` evaluates the
        defining doubling recursion with quadratic arithmetic,
        ``"cariow_cariowa"`` uses fixed straight-line kernels through
        dimension 64 and table-driven kernels from 128 through 1024, where it
        is available but not recommended, and ``"quasilinear"`` uses
        ``O(N log N)`` arithmetic at every dimension.  The algorithms can
        round differently.  Ordinary
        operators, ``numpy.multiply``, and powers always use ``"auto"``.
        Real scalar operands are scaled coordinatewise without invoking a
        multiplication kernel, so the algorithm does not change their result.
        """

        from ._array import ElementArray, _is_real_scalar, _multiply

        code = _core._algorithm(algorithm, self.dimension)
        if isinstance(x, Element) and isinstance(y, Element) and out is None:
            self._require_element(x)
            self._require_element(y)
            return _core._multiply_element(code, x, y)
        if isinstance(x, Element) and _is_real_scalar(y) and out is None:
            self._require_element(x)
            return cast(Element, x * y)
        if _is_real_scalar(x) and isinstance(y, Element) and out is None:
            self._require_element(y)
            return cast(Element, x * y)
        if _is_real_scalar(x) and _is_real_scalar(y):
            product = self.scalar(float(x) * float(y))
            if out is None:
                return product
            return cast(
                ElementArray,
                _multiply(
                    self, product, 1.0, out=out, allow_array_like=True
                ),
            )

        return cast(
            ElementArray,
            _multiply(
                self, x, y, out=out, allow_array_like=True, algorithm=code
            ),
        )

    def left_basis_multiply(
        self, index: Any, x: Any, *, out: Any | None = None
    ) -> Element | ElementArray:
        """Return ``e_index * x`` by a signed coordinate permutation.

        Parameters
        ----------
        index : int
            Basis index in ``range(dimension)``.
        x : Element, ElementArray, or array-like
            Element coefficients with final axis ``dimension``.
        out : ElementArray or numpy.ndarray, optional
            Nonoverlapping output satisfying the contract of
            :meth:`multiply`.

        Returns
        -------
        Element or ElementArray
            One :class:`Element` input and no ``out`` produce an Element;
            otherwise an ElementArray is returned.

        Raises
        ------
        TypeError
            If the index, operand, or output object is unsupported.
        IndexError
            If ``index`` is outside the standard basis.
        ValueError
            If the parent, shape, layout, or overlap contract is violated.

        Notes
        -----
        No multiplication matrix is built.  The native kernel performs only
        a signed permutation of coordinates.
        """

        normalized = _basis_index(index, self.dimension)
        if isinstance(x, Element) and out is None:
            self._require_element(x)
            return _core._basis_lmul(normalized, x)
        from ._array import _basis_action

        return _basis_action(self, normalized, x, left=True, out=out)

    def right_basis_multiply(
        self, x: Any, index: Any, *, out: Any | None = None
    ) -> Element | ElementArray:
        """Return ``x * e_index`` by a signed coordinate permutation.

        Parameters
        ----------
        x : Element, ElementArray, or array-like
            Element coefficients with final axis ``dimension``.
        index : int
            Basis index in ``range(dimension)``.
        out : ElementArray or numpy.ndarray, optional
            Nonoverlapping output satisfying the contract of
            :meth:`multiply`.

        Returns
        -------
        Element or ElementArray
            One :class:`Element` input and no ``out`` produce an Element;
            otherwise an ElementArray is returned.

        Raises
        ------
        TypeError
            If the index, operand, or output object is unsupported.
        IndexError
            If ``index`` is outside the standard basis.
        ValueError
            If the parent, shape, layout, or overlap contract is violated.

        Notes
        -----
        No multiplication matrix is built.  Left and right signs can differ
        because the algebra is generally noncommutative.
        """

        normalized = _basis_index(index, self.dimension)
        if isinstance(x, Element) and out is None:
            self._require_element(x)
            return _core._basis_rmul(x, normalized)
        from ._array import _basis_action

        return _basis_action(self, normalized, x, left=False, out=out)

    def left_operator_commutator(
        self, a: Any, b: Any, x: Any, *, out: Any | None = None
    ) -> Element | ElementArray:
        """Apply the left multiplication-operator commutator matrix-free.

        Parameters
        ----------
        a, b, x : Element, ElementArray, or array-like
            Inputs to ``[L_a, L_b](x) = a*(b*x) - b*(a*x)``.  Array inputs
            must have final axis ``dimension``.
        out : ElementArray or numpy.ndarray, optional
            Nonoverlapping output satisfying the contract of
            :meth:`multiply`.

        Returns
        -------
        Element or ElementArray
            Three Elements and no ``out`` produce an Element.  Otherwise,
            leading batch axes are broadcast and an ElementArray is returned.

        Raises
        ------
        TypeError
            If an operand or output object is unsupported.
        ValueError
            If parents, shapes, broadcasting, layout, or overlap are
            incompatible.

        Notes
        -----
        The expression is evaluated in the displayed order by native kernels;
        no dense left-multiplication operator is materialized.
        """

        if all(isinstance(value, Element) for value in (a, b, x)) and out is None:
            for value in (a, b, x):
                self._require_element(value)
            return _core._lop_commutator(a, b, x)
        from ._array import _operator_commutator

        return _operator_commutator(self, a, b, x, left=True, out=out)

    def right_operator_commutator(
        self, a: Any, b: Any, x: Any, *, out: Any | None = None
    ) -> Element | ElementArray:
        """Apply the right multiplication-operator commutator matrix-free.

        Parameters
        ----------
        a, b, x : Element, ElementArray, or array-like
            Inputs to ``[R_a, R_b](x) = (x*b)*a - (x*a)*b``.  Array inputs
            must have final axis ``dimension``.
        out : ElementArray or numpy.ndarray, optional
            Nonoverlapping output satisfying the contract of
            :meth:`multiply`.

        Returns
        -------
        Element or ElementArray
            Three Elements and no ``out`` produce an Element.  Otherwise,
            leading batch axes are broadcast and an ElementArray is returned.

        Raises
        ------
        TypeError
            If an operand or output object is unsupported.
        ValueError
            If parents, shapes, broadcasting, layout, or overlap are
            incompatible.

        Notes
        -----
        Parentheses are part of the contract in nonassociative algebras.  No
        dense right-multiplication operator is materialized.
        """

        if all(isinstance(value, Element) for value in (a, b, x)) and out is None:
            for value in (a, b, x):
                self._require_element(value)
            return _core._rop_commutator(a, b, x)
        from ._array import _operator_commutator

        return _operator_commutator(self, a, b, x, left=False, out=out)

    def _require_element(self, value: Element) -> None:
        """Require an element to belong to this cached algebra parent."""

        if value.algebra is not self:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)


def _algebra_from_dimension(dimension: int) -> Algebra:
    """Recover a cached algebra parent from its real dimension."""

    return Algebra(dimension=dimension)


def cd_element(coefficients: Any) -> Element:
    """Construct a Cayley--Dickson element and infer its dimension.

    Parameters
    ----------
    coefficients : Element or one-dimensional array-like
        A nonempty power-of-two number of real coefficients.  An existing
        Element is copied.

    Returns
    -------
    Element
        An owned element of the cached algebra with the inferred dimension.

    Raises
    ------
    ValueError
        If the input is not one-dimensional or its length is not a positive
        power of two.
    TypeError
        If coefficients are complex, object-valued, or nonnumeric.

    Notes
    -----
    This convenience function uses NumPy to inspect the input length.  If the
    dimension is already known, calling an :class:`Algebra` makes the parent
    explicit and avoids inference.  Coefficients are always copied.
    """

    if isinstance(coefficients, Element):
        return Algebra(dimension=coefficients.dimension)(coefficients)
    values = np.asarray(coefficients)
    if values.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")
    algebra = Algebra(dimension=values.shape[0])
    return algebra(values)
