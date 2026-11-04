"""Parents and constructors for standard Cayley--Dickson algebras.

The public entry points in this module are re-exported from :mod:`pyfastcd`;
applications normally use ``import pyfastcd as fcd``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from types import GenericAlias
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
    """A standard Cayley--Dickson algebra over an explicitly selected field.

    Parameters
    ----------
    dimension : int, optional
        Positive power-of-two dimension over the coefficient field.
        Boolean values are rejected.
    level : int, optional
        Nonnegative doubling level ``n``, giving dimension ``2**n``.
        Boolean values are rejected.

    char : int, optional
        Zero (default) for characteristic zero, or an odd prime at most 2**31-1.
    deg : int, optional
        Coefficient extension degree; 1 for R, 2 for C when char is zero.
        Finite degrees lie in 1..12 and are independent of the CD level.
    field : FiniteField, optional
        Reuse a finite coefficient field with default char/deg selectors.

    Raises
    ------
    TypeError
        If neither or both dimension keywords are supplied, or if the selected value
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
    objects are immutable, cannot be subclassed, and are cached by dimension
    and coefficient presentation. Default calls use the original real tower
    and native ``float64`` implementation. Explicit complex parents use
    ``complex128``; finite parents use canonical ``uint32`` polynomial words.
    See ``coefficient_shape`` for complete record axes.

    Examples
    --------
    >>> import pyfastcd as fcd
    >>> H = fcd.Algebra(dimension=4)
    >>> H([1, 2, 3, 4])
    Element([1.0, 2.0, 3.0, 4.0], algebra=Algebra(dimension=4, level=2, dtype=float64))
    """

    __slots__ = (
        "_dimension",
        "_level",
        "_initialized",
        "_char",
        "_deg",
        "_field",
        "_ctx",
    )
    _dimension: int
    _level: int
    _initialized: bool
    _cache: dict[Any, Algebra] = {}
    _char: int
    _deg: int
    _field: Any
    _ctx: Any

    def __class_getitem__(cls, parameters: Any) -> Any:
        """Support optional domain-specific static type annotations."""

        return GenericAlias(cls, parameters)

    def __new__(
        cls,
        *,
        dimension: Any | None = None,
        level: Any | None = None,
        char: Any = 0,
        deg: Any = 1,
        field: Any = None,
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
        from ._field import FiniteField, _integer

        p, d = _integer(char, "char"), _integer(deg, "deg")
        if field is not None:
            if not isinstance(field, FiniteField):
                raise TypeError("field must be a FiniteField")
            if p != 0 or d != 1:
                raise TypeError("field cannot be combined with char or deg")
            p, d = field.char, field.deg
        elif p:
            field = FiniteField(char=p, deg=d)
        elif d not in (1, 2):
            raise ValueError("char=0 supports only deg=1 (R) or deg=2 (C)")
        # Keep the original integer cache key for real parents. New domains
        # include the actual finite presentation, not merely its field order.
        key = (
            normalized_dimension
            if p == 0 and d == 1
            else (normalized_dimension, p, d, field)
        )
        cached = cls._cache.get(key)
        if cached is not None:
            return cached
        self = super().__new__(cls)
        object.__setattr__(self, "_dimension", normalized_dimension)
        object.__setattr__(self, "_level", normalized_level)
        object.__setattr__(self, "_initialized", True)
        object.__setattr__(self, "_char", p)
        object.__setattr__(self, "_deg", d)
        object.__setattr__(self, "_field", field)
        # Only explicitly selected domains construct the separate native
        # context; the default real path retains its original implementation.
        if p or d != 1:
            from . import _domains_core

            ctx = _domains_core.context(
                field._native if field is not None else None,
                normalized_dimension,
                False,
            )
        else:
            ctx = None
        object.__setattr__(self, "_ctx", ctx)
        return cls._cache.setdefault(key, self)

    def __init__(
        self,
        *,
        dimension: Any | None = None,
        level: Any | None = None,
        char: Any = 0,
        deg: Any = 1,
        field: Any = None,
    ) -> None:
        """Complete construction without changing cached parent state.

        ``__new__`` performs validation and initialization.  This no-op is
        called again when construction returns a cached Algebra.
        """

        # Validation and initialization are performed in __new__.  The no-op
        # __init__ is invoked again when a cached parent is returned.
        del dimension, level, char, deg, field

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
        """int: Number of coefficients over the selected field."""

        return self._dimension

    @property
    def level(self) -> int:
        """int: Doubling level ``n`` satisfying ``dimension == 2**n``."""

        return self._level

    @property
    def char(self) -> int:
        """Coefficient characteristic; zero denotes R or C."""

        return self._char

    @property
    def deg(self) -> int:
        """Coefficient extension degree, independent of the CD level."""

        return self._deg

    @property
    def field(self) -> Any:
        """Finite coefficient field, or None for R and C."""

        return self._field

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        """Trailing storage axes of one complete algebra element."""

        return (
            (self.dimension, self.deg)
            if self.char and self.deg > 1
            else (self.dimension,)
        )

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """numpy.dtype: Native coefficient storage dtype for the selected field."""

        if self.char:
            return cast("np.dtype[np.float64]", np.dtype(np.uint32))
        if self.deg == 2:
            return cast("np.dtype[np.float64]", np.dtype(np.complex128))
        return np.dtype(np.float64)

    @property
    def default_algorithm(self) -> str:
        """str: Algorithm that ``"auto"`` selects for a single product.

        One of ``"direct"``, ``"cariow_cariowa"``, or ``"quasilinear"``; the
        real selection depends only on dimension; complex/finite parents
        always select quasilinear.  Products of two
        :class:`Element` objects, and batches containing exactly one record,
        use this algorithm, so passing the name explicitly to
        :meth:`multiply` reproduces ``x * y`` exactly for elements.  Batches
        of two or more records use :attr:`default_batch_algorithm`.
        """

        if self._ctx is not None:
            return "quasilinear"
        return _core._default_algorithm(self.dimension)

    @property
    def default_batch_algorithm(self) -> str:
        """str: Algorithm that ``"auto"`` selects for batches of two or more.

        One of ``"direct"``, ``"cariow_cariowa"``, or ``"quasilinear"``.
        Real batches can evaluate two products per SIMD instruction, and
        their defaults come from the real benchmark. Complex/finite parents
        currently select quasilinear for every batch size.
        Passing this name explicitly to :meth:`multiply` reproduces ``X * Y``
        exactly for arrays with at least two records.
        """

        if self._ctx is not None:
            return "quasilinear"
        return _core._default_batch_algorithm(self.dimension)

    def __repr__(self) -> str:
        """Return a constructor-oriented description of this parent."""

        if self.char:
            return f"Algebra(dimension={self.dimension}, field={self.field!r})"
        if self.deg == 2:
            return f"Algebra(dimension={self.dimension}, char=0, deg=2)"
        return f"Algebra(dimension={self.dimension}, level={self.level}, dtype=float64)"

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Return reconstruction data that resolves through the parent cache."""

        if self._ctx is not None:
            return (
                _algebra_from_domain,
                (self.dimension, self.char, self.deg, self.field),
            )
        return (_algebra_from_dimension, (self.dimension,))

    def __call__(self, value: Any = 0) -> Element:
        """Construct one owned element of this algebra.

        Parameters
        ----------
        value : coefficient scalar, Element, or array-like, optional
            A scalar embeds in coordinate zero. A complete coefficient record
            must have exactly ``coefficient_shape``. Wrapped elements must have
            the same parent; matching finite coefficient scalars also embed.

        Returns
        -------
        Element
            A newly allocated immutable element belonging to this algebra.

        Raises
        ------
        ValueError
            If the record shape or parent is incompatible.
        TypeError
            If the selected field does not support the coefficients. Real
            parents reject complex/object inputs; finite parents require
            integers or matching field scalars.
        OverflowError
            If a real/complex floating-point conversion exceeds binary64 range.

        Notes
        -----
        Construction always copies. Finite integer coordinates reduce modulo
        ``char`` before narrowing, including large Python integers. For a finite
        extension, a record has shape ``(dimension, deg)``; a sequence of matching
        field scalars also supplies its complete coefficients.
        """

        if self._ctx is not None:
            from ._domains import element

            return cast(Element, element(self, value))
        if isinstance(value, Element) and value.algebra._ctx is not None:
            raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)

        if isinstance(value, memoryview):
            try:
                value = np.asarray(value)
            except NotImplementedError as error:
                raise TypeError("unsupported coefficient buffer format") from error
        if isinstance(value, (complex, np.complexfloating)):
            raise TypeError(COMPLEX_SCALAR_MESSAGE)
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf":
                raise TypeError("coefficient arrays must contain real numeric values")
        return _core._element(self, self.dimension, value)

    def scalar(self, value: _RealScalar) -> Element:
        """Return ``value`` embedded in the scalar coordinate.

        Parameters
        ----------
        value : coefficient scalar
            Real numeric scalar for R, complex numeric scalar for C, or integer
            or matching FiniteField scalar for a finite parent. Use ``F([...])``
            for a non-prime-subfield extension value; raw coordinate vectors
            and complete CD records are not scalar inputs.

        Returns
        -------
        Element
            A newly allocated scalar element ``value * e_0``.

        Raises
        ------
        TypeError
            If the selected field does not support the scalar or it is nonscalar.
        ValueError, OverflowError
            If numeric conversion fails or a floating-point value is too large.
        """

        if self._ctx is not None:
            from ._domains import scalar

            if not scalar(value, self):
                raise TypeError("scalar must belong to the coefficient field")
            return self(value)

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

        if self._ctx is not None:
            return self(0)

        return _core._scalar(self, self.dimension, 0.0)

    def one(self) -> Element:
        """Return the multiplicative identity as a new element.

        Returns
        -------
        Element
            A newly allocated copy of ``e_0``.
        """

        if self._ctx is not None:
            return self(1)

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

        if self._ctx is not None:
            i = _basis_index(index, self.dimension)
            coefficients = np.zeros(self.coefficient_shape, dtype=self.dtype)
            coefficients.flat[i * (self.deg if self.char else 1)] = 1
            return self(coefficients)

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
            Complete records with shape ``batch_shape + coefficient_shape``.
            Wrapped records must have this parent. Finite inputs may contain
            matching field scalars or integer polynomial coordinates.

        Returns
        -------
        ElementArray
            Newly allocated writable, aligned, C-contiguous storage in ``dtype``:
            float64 over R, complex128 over C, or uint32 over finite fields.

        Raises
        ------
        ValueError
            If a record shape or wrapped parent is incompatible.
        TypeError
            If the selected field does not support the coefficients.

        Notes
        -----
        Coefficients are always copied. Only leading batch axes broadcast;
        finite extension records retain their trailing ``(dimension, deg)``
        axes. Integer residues reduce before narrowing.
        """

        if self._ctx is not None:
            from ._domains import array

            return cast("ElementArray", array(self, values, copy=True))

        from ._array import _array_from_values

        return _array_from_values(self, values, copy=True)

    def asarray(self, values: Any, *, copy: bool | None = None) -> ElementArray:
        """Interpret complete coefficient records as algebra elements.

        Parameters
        ----------
        values : Element, ElementArray, array-like, or DLPack producer
            Storage with shape ``batch_shape + coefficient_shape``. DLPack input
            must use a supported host-accessible device.
        copy : bool or None, optional
            ``True`` always copies. ``None`` shares compatible storage or copies
            as required. ``False`` requires zero-copy sharing or raises.

        Returns
        -------
        ElementArray
            A wrapper preserving leading strides and source writeability when
            storage is shared. Legacy DLPack imports may expose read-only views.

        Raises
        ------
        TypeError
            If the copy flag or coefficients are unsupported.
        ValueError
            If shapes, parents or devices are incompatible, or ``copy=False``
            cannot satisfy native dtype, alignment, layout and canonical residues.

        Notes
        -----
        The real path requires native float64 and a final stride of eight bytes;
        dimension one also permits zero stride, and empty C-contiguous real
        arrays need no final-stride check. Complex/finite parents require native
        contiguous record axes, ignoring singleton axes and empty arrays.
        Finite zero-copy storage must already be canonical. Shared mutable
        residues are checked again before arithmetic; synchronize concurrent
        writers throughout each call. Views of immutable Elements stay read-only.
        """

        if self._ctx is not None:
            from ._domains import array

            return cast("ElementArray", array(self, values, copy=copy))

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
        x, y : coefficient scalar, Element, ElementArray, or array-like
            Factors. A coefficient scalar scales every coordinate. Nonscalar
            arrays must end in the parent's complete ``coefficient_shape``.
        out : ElementArray or numpy.ndarray, optional
            Exact-shape output storage.  It must be writable, aligned,
            C-contiguous with this parent's native ``dtype`` and must not
            overlap an input.
        algorithm : str, optional
            Multiplication algorithm name: ``"auto"`` (the default),
            ``"direct"``, ``"cariow_cariowa"``, or ``"quasilinear"``.
            ``"auto"`` selects :attr:`default_algorithm` for a single
            product and :attr:`default_batch_algorithm` for batches of two
            or more records.  The name is validated before any arithmetic,
            including for scalar operands and empty batches. Complex and
            finite parents support only auto, direct, and quasilinear.

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
        Leading batch axes follow NumPy broadcasting. Complete trailing
        coefficient axes are fixed and never broadcast. ``"direct"`` evaluates the
        defining doubling recursion with quadratic arithmetic,
        ``"cariow_cariowa"`` uses fixed straight-line kernels through
        dimension 64 and table-driven kernels from 128 through 1024, where it
        is available but not recommended for real parents, and ``"quasilinear"`` uses
        ``O(N log N)`` arithmetic at every dimension.  The algorithms can
        round differently.  Ordinary
        operators, ``numpy.multiply``, and powers always use ``"auto"``.
        Coefficient scalar operands are scaled coordinatewise without invoking a
        multiplication kernel, so the algorithm does not change their result.
        """

        if self._ctx is not None:
            from ._domains import multiply

            return cast(
                "Element | ElementArray",
                multiply(self, x, y, out=out, algorithm=algorithm),
            )

        from ._array import _is_real_scalar, _multiply

        code = _core._algorithm(algorithm, self.dimension)
        if isinstance(x, Element) and isinstance(y, Element) and out is None:
            self._require_element(x)
            self._require_element(y)
            return _core._multiply_element(code, x, y)
        if isinstance(x, Element) and _is_real_scalar(y) and out is None:
            self._require_element(x)
            return cast("Element | ElementArray", x * y)
        if _is_real_scalar(x) and isinstance(y, Element) and out is None:
            self._require_element(y)
            return cast("Element | ElementArray", x * y)
        if _is_real_scalar(x) and _is_real_scalar(y):
            product = self.scalar(float(x) * float(y))
            if out is None:
                return product
            return cast(
                "Element | ElementArray",
                _multiply(self, product, 1.0, out=out, allow_array_like=True),
            )

        return cast(
            "Element | ElementArray",
            _multiply(self, x, y, out=out, allow_array_like=True, algorithm=code),
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
            Complete records with trailing ``coefficient_shape``. Scalars are
            not embedded implicitly; use ``self.scalar(r)`` for a scalar record.
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

        if self._ctx is not None:
            from ._domains import apply

            return cast(
                "Element | ElementArray",
                apply(
                    self,
                    "basis_left",
                    x,
                    out=out,
                    index=_basis_index(index, self.dimension),
                    coerce=True,
                ),
            )

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
            Complete records with trailing ``coefficient_shape``. Scalars are
            not embedded implicitly; use ``self.scalar(r)`` for a scalar record.
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

        if self._ctx is not None:
            from ._domains import apply

            return cast(
                "Element | ElementArray",
                apply(
                    self,
                    "basis_right",
                    x,
                    out=out,
                    index=_basis_index(index, self.dimension),
                    coerce=True,
                ),
            )

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
            must end in ``coefficient_shape``. Scalar inputs are not embedded;
            construct a scalar record explicitly when intended.
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

        if self._ctx is not None:
            from ._domains import apply

            return cast(
                "Element | ElementArray",
                apply(self, "left_operator_commutator", a, b, x, out=out, coerce=True),
            )

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
            must end in ``coefficient_shape``. Scalar inputs are not embedded;
            construct a scalar record explicitly when intended.
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

        if self._ctx is not None:
            from ._domains import apply

            return cast(
                "Element | ElementArray",
                apply(self, "right_operator_commutator", a, b, x, out=out, coerce=True),
            )

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


def cd_element(
    coefficients: Any, *, char: Any = 0, deg: Any = 1, field: Any = None
) -> Element:
    """Construct a Cayley--Dickson element and infer its dimension.

    Parameters
    ----------
    coefficients : Element or array-like
        One complete record with a nonempty power-of-two number of CD
        coefficients. Default real and explicit complex/prime-field inputs
        have shape ``(N,)``; finite extensions have shape ``(N, deg)``.
        An existing Element is copied, retaining its parent when no new
        coefficient selectors are supplied.
    char, deg, field : optional
        Coefficient selectors with the same contract as :class:`Algebra`.
        Omitting them preserves the original real constructor calls.

    Returns
    -------
    Element
        An owned element of the cached algebra with the inferred dimension.

    Raises
    ------
    ValueError
        If the input is not one complete coefficient record or its CD
        dimension is not a positive power of two.
    TypeError
        If coefficients are unsupported by the selected field. Default real
        calls reject complex/object inputs; finite calls require integers.

    Notes
    -----
    This convenience function uses NumPy to inspect the input length.  If the
    dimension is already known, calling an :class:`Algebra` makes the parent
    explicit and avoids inference.  Coefficients are always copied.
    """

    if char != 0 or deg != 1 or field is not None:
        probe = Algebra(dimension=1, char=char, deg=deg, field=field)
        if isinstance(coefficients, Element):
            if (
                coefficients.algebra.char != probe.char
                or coefficients.algebra.deg != probe.deg
                or coefficients.algebra.field is not probe.field
            ):
                raise ValueError(DIFFERENT_ALGEBRA_MESSAGE)
            return cast(Element, coefficients.copy())
        from ._domains import _finite_values

        values = np.asarray(
            _finite_values(probe, coefficients) if probe.char else coefficients
        )
        axes = 2 if probe.char and probe.deg > 1 else 1
        if values.ndim != axes:
            raise ValueError("expected one complete coefficient record")
        return Algebra(dimension=values.shape[0], char=char, deg=deg, field=field)(
            coefficients
        )
    if isinstance(coefficients, Element) and coefficients.algebra._ctx is not None:
        return cast(Element, coefficients.copy())

    if isinstance(coefficients, Element):
        return Algebra(dimension=coefficients.dimension)(coefficients)
    values = np.asarray(coefficients)
    if values.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")
    algebra = Algebra(dimension=values.shape[0])
    return algebra(values)


def _algebra_from_domain(dimension: int, char: int, deg: int, field: Any) -> Algebra:
    """Restore a new-domain pickle using its actual finite field presentation."""

    return (
        Algebra(dimension=dimension, field=field)
        if field is not None
        else Algebra(dimension=dimension, char=char, deg=deg)
    )
