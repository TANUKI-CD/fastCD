"""Immutable polynomial-basis coefficient fields for exact CD arithmetic."""

from __future__ import annotations

import operator
from typing import Any, final

import numpy as np

from . import _domains_core


def _integer(value: Any, name: str) -> int:
    """Validate a selector with the integer protocol while rejecting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


@final
class FiniteField:
    """An immutable coefficient parent for the finite field F_(char**deg).

    Parameters
    ----------
    char : int
        Odd prime in ``3..2**31-1``. Boolean values are rejected.
    deg : int, optional
        Extension degree in ``1..12``, defaulting to the prime field.
        This is independent of a containing algebra's CD level.

    Attributes
    ----------
    char, deg : int
        The characteristic and extension degree.
    order : int
        Exact Python integer ``char**deg``, without a machine-word limit.
    modulus : tuple[int, ...]
        Actual monic irreducible polynomial, constant coefficient first.
    dtype : numpy.dtype
        Native ``uint32`` storage for canonical prime-field coordinates.
    coefficient_shape : tuple[int, ...]
        ``()`` at degree one, otherwise ``(deg,)`` for one field scalar.

    Raises
    ------
    TypeError
        If either selector is not integer-like or is a boolean.
    ValueError
        If the characteristic is not a supported odd prime or the degree
        is outside the supported range.

    Notes
    -----
    The constructor chooses a deterministic polynomial basis. Use
    :meth:`from_modulus` for an explicit presentation. Parents are cached by
    their actual defining polynomial: isomorphic fields with different
    presentations do not coerce implicitly. Pickles retain that polynomial.

    Scalars and arrays use the existing ``Element`` and ``ElementArray``
    classes. Integers embed in the prime subfield and reduce before storage
    conversion; floating-point coefficients are rejected. For extensions,
    pass all ``deg`` polynomial coordinates explicitly when constructing a
    non-prime-subfield value. ``quadratic_norm()`` on a scalar c is c*c;
    it is not the relative field norm from this field to its prime subfield.

    Examples
    --------
    >>> import pyfastcd as fcd
    >>> F = fcd.FiniteField(char=13, deg=5)
    >>> c = F([1, 2, 0, 0, 0])
    >>> c * c.reciprocal() == F.one()
    True
    """

    __slots__ = ("char", "deg", "order", "modulus", "_native", "_ctx")
    char: int
    deg: int
    order: int
    modulus: tuple[int, ...]
    _native: Any
    _ctx: Any
    _cache: dict[tuple[int, tuple[int, ...]], FiniteField] = {}
    _defaults: dict[tuple[int, int], FiniteField] = {}

    def __new__(cls, *, char: Any, deg: Any = 1) -> FiniteField:
        """Validate selectors and retrieve the cached polynomial presentation."""

        if cls is not FiniteField:
            raise TypeError("FiniteField cannot be subclassed")
        p, d = _integer(char, "char"), _integer(deg, "deg")
        if not 3 <= p <= 2**31 - 1 or not 1 <= d <= 12:
            raise ValueError(
                "char must be an odd prime in 3..2**31-1 and deg must be in 1..12"
            )
        if (p, d) not in cls._defaults:
            cls._defaults[p, d] = _make_field(p, d, None)
        return cls._defaults[p, d]

    def __init__(self, *, char: Any, deg: Any = 1) -> None:
        """Initialization is completed once by the cached constructor."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject subclasses so cached field identity remains unambiguous."""

        raise TypeError("FiniteField cannot be subclassed")

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject mutation of a field shared by scalars and containing algebras."""

        raise AttributeError("FiniteField objects are immutable")

    def __delattr__(self, name: str) -> None:
        """Reject deletion of immutable cached field state."""

        raise AttributeError("FiniteField objects are immutable")

    @classmethod
    def from_modulus(cls, *, char: Any, modulus: Any) -> FiniteField:
        """Validate an explicit polynomial presentation, constant term first.

        ``modulus`` is an ordered integer sequence of length ``deg+1``;
        entries reduce modulo ``char`` and the leading coefficient must be
        one. The characteristic and derived degree have the same bounds as
        the default constructor. Reducibility raises ``ValueError``.

        The returned cached parent records the actual polynomial, even when
        it also equals the automatically selected presentation. No implicit
        isomorphism is installed between different extension presentations.
        Degree-one monic moduli all select the canonical prime parent with
        modulus (0, 1), since its scalar coordinates do not depend on the root.
        """

        p = _integer(char, "char")
        if not 3 <= p <= 2**31 - 1:
            raise ValueError("char must be an odd prime in 3..2**31-1")
        if isinstance(modulus, (set, dict, str, bytes)):
            raise TypeError("modulus must be an ordered integer sequence")
        mod = tuple(_integer(v, "modulus coefficient") % p for v in modulus)
        if not 2 <= len(mod) <= 13:
            raise ValueError("modulus degree must lie in 1..12")
        if len(mod) == 2:
            if mod[1] != 1:
                raise ValueError("modulus must be monic")
            # Linear moduli do not change degree-one coordinates or arithmetic.
            # Give all such presentations the single canonical prime parent.
            return cls(char=p)
        return _make_field(p, len(mod) - 1, mod)

    @property
    def dtype(self) -> np.dtype[np.uint32]:
        """Native uint32 storage for prime-field polynomial coordinates."""

        return np.dtype(np.uint32)

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        """Storage axes of one field scalar."""

        return () if self.deg == 1 else (self.deg,)

    @property
    def dimension(self) -> int:
        """One coefficient over this field itself."""

        return 1

    @property
    def level(self) -> int:
        """A coefficient-field scalar is at CD level zero."""

        return 0

    @property
    def field(self) -> FiniteField:
        """The coefficient field of a field scalar is this parent."""

        return self

    def __repr__(self) -> str:
        """Show the selectors or explicit polynomial needed to recreate this parent."""

        # Default status belongs to the polynomial, not construction history.
        # Resolve the cached default lazily; pickles still restore the actual
        # explicit polynomial without depending on default selection.
        if self is FiniteField(char=self.char, deg=self.deg):
            return f"FiniteField(char={self.char}, deg={self.deg})"
        return f"FiniteField.from_modulus(char={self.char}, modulus={self.modulus!r})"

    def __reduce__(self) -> Any:
        """Serialize the actual polynomial rather than rerunning default selection."""

        return (_restore_field, (self.char, self.modulus))

    def __call__(self, value: Any = 0) -> Any:
        """Copy a matching scalar or construct polynomial coordinates."""

        from ._domains import element

        return element(self, value)

    def scalar(self, value: Any) -> Any:
        """Construct one coefficient-field scalar."""

        return self(value)

    def zero(self) -> Any:
        """Return the additive identity."""

        return self(0)

    def one(self) -> Any:
        """Return the multiplicative identity."""

        return self(1)

    def array(self, values: Any) -> Any:
        """Copy values into writable canonical coefficient storage."""

        from ._domains import array

        return array(self, values, copy=True)

    def asarray(self, values: Any, *, copy: bool | None = None) -> Any:
        """Construct a field array under the requested copy policy.

        ``copy=None`` shares compatible native canonical storage, ``True``
        always copies, and ``False`` requires sharing or raises ``ValueError``.
        Extension polynomial axes stay contiguous; only leading batch axes
        may be broadcast or strided. Shared mutable storage is checked again
        for canonical residues before native arithmetic.
        """

        from ._domains import array

        return array(self, values, copy=copy)

    def multiply(self, x: Any, y: Any, *, out: Any = None) -> Any:
        """Multiply field scalars or broadcast field arrays.

        Return an ``Element`` for scalar operands, otherwise an
        ``ElementArray``. ``out`` follows the algebra multiplication contract:
        exact broadcast shape, native writable contiguous storage, and no
        overlap with inputs. A supplied wrapper is returned unchanged.
        """

        from ._domains import multiply

        return multiply(self, x, y, out=out)


def _make_field(p: int, d: int, mod: tuple[int, ...] | None) -> FiniteField:
    """Publish one immutable parent per validated polynomial presentation.

    Native setup chooses or checks the polynomial before cache lookup. The
    field-scalar context copies that state, so every product can use its own
    scratch without changing the shared parent.
    """

    if mod is not None and (p, mod) in FiniteField._cache:
        return FiniteField._cache[p, mod]
    native, actual = _domains_core.field(p, d, mod)
    key = (p, actual)
    cached = FiniteField._cache.get(key)
    if cached is not None:
        return cached
    obj = object.__new__(FiniteField)
    for name, value in {
        "char": p,
        "deg": d,
        "order": p**d,
        "modulus": actual,
        "_native": native,
        "_ctx": _domains_core.context(native, 1, True),
    }.items():
        object.__setattr__(obj, name, value)
    FiniteField._cache[key] = obj
    return obj


def _restore_field(p: int, mod: tuple[int, ...]) -> FiniteField:
    """Reconstruct a pickle through validation of its explicit presentation."""

    return FiniteField.from_modulus(char=p, modulus=mod)
