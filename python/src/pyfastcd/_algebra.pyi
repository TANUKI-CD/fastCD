from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Generic, Literal, final, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeVar

from ._array import ElementArray
from ._core import Element
from ._field import FiniteField

_D = TypeVar("_D", bound=np.generic, default=np.float64)
_S = TypeVar("_S", default=float)
_R = TypeVar("_R", default=NDArray[np.float64])
_P = TypeVar("_P", default=Algebra)
_H = TypeVar("_H", bound=tuple[int, ...], default=tuple[int])
_FieldScalar = Element[np.uint32, Any, FiniteField, tuple[int, ...]]
_FieldArray = ElementArray[np.uint32, _FieldScalar, Any, FiniteField, tuple[int, ...]]

@dataclass(frozen=True)
class BasisProduct:
    index: int
    sign: int

    def __iter__(self) -> Iterator[int]: ...

@final
class Algebra(Generic[_D, _S, _R, _H]):
    @overload
    def __new__(
        cls,
        *,
        dimension: Any = None,
        level: Any = None,
        char: Literal[0] = 0,
        deg: Literal[1] = 1,
        field: None = None,
    ) -> Algebra[np.float64, float, NDArray[np.float64]]: ...
    @overload
    def __new__(
        cls,
        *,
        dimension: Any = None,
        level: Any = None,
        char: Literal[0] = 0,
        deg: Literal[2],
        field: None = None,
    ) -> Algebra[np.complex128, complex, NDArray[np.complex128]]: ...
    @overload
    def __new__(
        cls,
        *,
        dimension: Any = None,
        level: Any = None,
        char: Literal[0] = 0,
        deg: Literal[1] = 1,
        field: FiniteField,
    ) -> Algebra[np.uint32, _FieldScalar, _FieldArray, tuple[int, ...]]: ...
    @overload
    def __new__(
        cls,
        *,
        dimension: Any = None,
        level: Any = None,
        char: Any = 0,
        deg: Any = 1,
        field: Any = None,
    ) -> Algebra[Any, Any, Any, Any]: ...
    @overload
    def __init__(
        self: Algebra[np.float64, float, NDArray[np.float64]],
        *,
        dimension: Any = None,
        level: Any = None,
        char: Literal[0] = 0,
        deg: Literal[1] = 1,
        field: None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: Algebra[np.complex128, complex, NDArray[np.complex128]],
        *,
        dimension: Any = None,
        level: Any = None,
        char: Literal[0] = 0,
        deg: Literal[2],
        field: None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: Algebra[np.uint32, _FieldScalar, _FieldArray, tuple[int, ...]],
        *,
        dimension: Any = None,
        level: Any = None,
        char: Literal[0] = 0,
        deg: Literal[1] = 1,
        field: FiniteField,
    ) -> None: ...
    @overload
    def __init__(
        self: Algebra[Any, Any, Any, Any],
        *,
        dimension: Any = None,
        level: Any = None,
        char: Any = 0,
        deg: Any = 1,
        field: Any = None,
    ) -> None: ...
    @property
    def dimension(self) -> int: ...
    @property
    def level(self) -> int: ...
    @property
    def char(self) -> int: ...
    @property
    def deg(self) -> int: ...
    @property
    def field(self) -> FiniteField | None: ...
    @property
    def coefficient_shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype[_D]: ...
    @property
    def default_algorithm(self) -> str: ...
    @property
    def default_batch_algorithm(self) -> str: ...
    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]: ...
    def __call__(
        self, value: Any = 0
    ) -> Element[_D, _S, Algebra[_D, _S, _R, _H], _H]: ...
    def scalar(self, value: Any) -> Element[_D, _S, Algebra[_D, _S, _R, _H], _H]: ...
    def zero(self) -> Element[_D, _S, Algebra[_D, _S, _R, _H], _H]: ...
    def one(self) -> Element[_D, _S, Algebra[_D, _S, _R, _H], _H]: ...
    def basis(self, index: Any) -> Element[_D, _S, Algebra[_D, _S, _R, _H], _H]: ...
    def basis_product(self, left_index: Any, right_index: Any) -> BasisProduct: ...
    def array(
        self, values: Any
    ) -> ElementArray[_D, _S, _R, Algebra[_D, _S, _R, _H], _H]: ...
    def asarray(
        self, values: Any, *, copy: bool | None = None
    ) -> ElementArray[_D, _S, _R, Algebra[_D, _S, _R, _H], _H]: ...
    def multiply(
        self, x: Any, y: Any, *, out: Any | None = None, algorithm: str = "auto"
    ) -> (
        Element[_D, _S, Algebra[_D, _S, _R, _H], _H]
        | ElementArray[_D, _S, _R, Algebra[_D, _S, _R, _H], _H]
    ): ...
    def left_basis_multiply(
        self, index: Any, x: Any, *, out: Any | None = None
    ) -> (
        Element[_D, _S, Algebra[_D, _S, _R, _H], _H]
        | ElementArray[_D, _S, _R, Algebra[_D, _S, _R, _H], _H]
    ): ...
    def right_basis_multiply(
        self, x: Any, index: Any, *, out: Any | None = None
    ) -> (
        Element[_D, _S, Algebra[_D, _S, _R, _H], _H]
        | ElementArray[_D, _S, _R, Algebra[_D, _S, _R, _H], _H]
    ): ...
    def left_operator_commutator(
        self, a: Any, b: Any, x: Any, *, out: Any | None = None
    ) -> (
        Element[_D, _S, Algebra[_D, _S, _R, _H], _H]
        | ElementArray[_D, _S, _R, Algebra[_D, _S, _R, _H], _H]
    ): ...
    def right_operator_commutator(
        self, a: Any, b: Any, x: Any, *, out: Any | None = None
    ) -> (
        Element[_D, _S, Algebra[_D, _S, _R, _H], _H]
        | ElementArray[_D, _S, _R, Algebra[_D, _S, _R, _H], _H]
    ): ...
    def _require_element(
        self, value: Element[_D, _S, Algebra[_D, _S, _R, _H], _H]
    ) -> None: ...

@overload
def cd_element(coefficients: Element[_D, _S, _P, _H]) -> Element[_D, _S, _P, _H]: ...
@overload
def cd_element(
    coefficients: Any, *, char: Literal[0] = 0, deg: Literal[1] = 1, field: None = None
) -> Element: ...
@overload
def cd_element(
    coefficients: Any, *, char: Literal[0] = 0, deg: Literal[2], field: None = None
) -> Element[
    np.complex128, complex, Algebra[np.complex128, complex, NDArray[np.complex128]]
]: ...
@overload
def cd_element(
    coefficients: Any, *, field: FiniteField, char: Literal[0] = 0, deg: Literal[1] = 1
) -> Element[
    np.uint32,
    _FieldScalar,
    Algebra[np.uint32, _FieldScalar, _FieldArray, tuple[int, ...]],
    tuple[int, ...],
]: ...
@overload
def cd_element(
    coefficients: Any, *, char: Any = 0, deg: Any = 1, field: Any = None
) -> Element[Any, Any, Any, Any]: ...
def _algebra_from_dimension(dimension: int) -> Algebra: ...
def _algebra_from_domain(
    dimension: int, char: int, deg: int, field: Any
) -> Algebra: ...
