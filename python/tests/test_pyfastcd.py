"""Deterministic tests for the public :mod:`pyfastcd` interface."""

from __future__ import annotations

import copy
import ctypes
import gc
import importlib.metadata
import inspect
import math
import pickle
import re
import struct
import sys
import unittest
import weakref
from typing import Any

import numpy as np
import pyfastcd as fcd
from pyfastcd import _array as array_module
from pyfastcd import _core

try:  # Support both unittest discovery and package-style test collection.
    from . import reference_cd as reference
except ImportError:  # pragma: no cover - depends on the collection mechanism
    import reference_cd as reference


RTOL = 8.0e-12
ATOL = 8.0e-12
ALGORITHM_NAMES = ("auto", "direct", "cariow_cariowa", "quasilinear")
CONCRETE_ALGORITHM_NAMES = ("direct", "cariow_cariowa", "quasilinear")
CARIOW_CARIOWA_MAX_DIMENSION = 1024
ALGORITHM_DIMENSIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _supported_algorithms(dimension: int) -> tuple[str, ...]:
    """Return the public algorithm names available at *dimension*."""

    return tuple(
        name
        for name in ALGORITHM_NAMES
        if name != "cariow_cariowa" or dimension <= CARIOW_CARIOWA_MAX_DIMENSION
    )


class _DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [
        ("version", _DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", _DLTensor),
    ]


_CAPSULE_GET_POINTER = ctypes.pythonapi.PyCapsule_GetPointer
_CAPSULE_GET_POINTER.argtypes = [ctypes.py_object, ctypes.c_char_p]
_CAPSULE_GET_POINTER.restype = ctypes.c_void_p


class _HostDeviceProducer:
    """Test producer that relabels a valid capsule as host-accessible."""

    def __init__(self, coefficients: np.ndarray, device_type: int) -> None:
        """Store coefficient memory and the device type to report."""

        self.coefficients = coefficients
        self.device_type = device_type

    def __dlpack_device__(self) -> tuple[int, int]:
        """Report the configured DLPack device and device zero."""

        return (self.device_type, 0)

    def __dlpack__(
        self,
        *,
        stream: object = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object:
        """Export coefficients after relabeling the capsule's device type."""

        del max_version, dl_device
        capsule = self.coefficients.__dlpack__(
            stream=stream, max_version=(1, 0), copy=copy
        )
        pointer = _CAPSULE_GET_POINTER(capsule, b"dltensor_versioned")
        managed = _DLManagedTensorVersioned.from_address(pointer)
        managed.dl_tensor.device.device_type = self.device_type
        managed.dl_tensor.device.device_id = 0
        return capsule


class _DualArrayProducer:
    """Expose both scalar and NumPy array protocols for dispatch tests."""

    def __init__(self, coefficients: object, scalar: float = 97.0) -> None:
        """Store independent array and scalar protocol values."""

        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.scalar = scalar

    def __float__(self) -> float:
        """Return the value exposed through the scalar protocol."""

        return self.scalar

    def __array__(
        self, dtype: np.dtype[object] | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """Return coefficients through NumPy's requested conversion policy."""

        result = np.asarray(self.coefficients, dtype=dtype)
        return result.copy() if copy else result


class _DualDLPackProducer(_HostDeviceProducer):
    """Expose both scalar and DLPack protocols for dispatch tests."""

    def __init__(self, coefficients: np.ndarray, scalar: float = 89.0) -> None:
        """Store DLPack coefficients and an independent scalar value."""

        super().__init__(coefficients, 1)
        self.scalar = scalar

    def __dlpack__(self, *, stream: object = None, **kwargs: object) -> object:
        """Forward the consumer's request without forcing a capsule version."""
        return self.coefficients.__dlpack__(stream=stream, **kwargs)

    @property
    def ndim(self) -> int:
        """Return the coefficient array's number of dimensions."""

        return self.coefficients.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the coefficient array's shape."""

        return self.coefficients.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the coefficient array's dtype."""

        return self.coefficients.dtype

    def __float__(self) -> float:
        """Return the value exposed through the scalar protocol."""

        return self.scalar


class FastCDTestCase(unittest.TestCase):
    """Common numerical assertions with useful failure diagnostics."""

    def assertCoefficientsEqual(self, actual: object, expected: object) -> None:
        """Assert exact equality after exposing both coefficient arrays."""

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def assertCoefficientsClose(
        self,
        actual: object,
        expected: object,
        *,
        rtol: float = RTOL,
        atol: float = ATOL,
    ) -> None:
        """Assert coefficient closeness with project test tolerances."""

        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol
        )


class PrivateCoreSafetyTests(FastCDTestCase):
    def test_coefficient_conversion_uses_a_stable_sequence_snapshot(self) -> None:
        """Keep conversion stable when coefficient callbacks mutate the source."""

        algebra = fcd.Algebra(dimension=4)

        class Coefficient:
            def __init__(
                self, value: float, source: list[object], action: str
            ) -> None:
                """Store a value and one source mutation to perform."""

                self.value = value
                self.source = source
                self.action = action

            def __float__(self) -> float:
                """Mutate the source once while returning this coefficient."""

                if self.value == 1.0:
                    if self.action == "clear":
                        self.source.clear()
                    elif self.action == "replace":
                        self.source[:] = [99.0] * 4
                    else:
                        self.source.extend([99.0] * 20)
                return self.value

        for mutation in ("clear", "replace", "grow"):
            with self.subTest(mutation=mutation):
                values: list[object] = []
                values.extend(
                    Coefficient(float(i), values, mutation) for i in range(1, 5)
                )
                references = [weakref.ref(value) for value in values]
                element = algebra(values)
                self.assertCoefficientsEqual(element, [1.0, 2.0, 3.0, 4.0])
                values.clear()
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references))

    def test_failed_coefficient_conversion_releases_snapshot_references(self) -> None:
        """Release temporary snapshot references after conversion failure."""

        values = []

        class Coefficient:
            def __float__(self) -> float:
                """Clear the source and raise the intended conversion failure."""

                values.clear()
                raise ValueError("coefficient conversion failed")

        values.extend(Coefficient() for _ in range(4))
        references = [weakref.ref(value) for value in values]
        with self.assertRaisesRegex(ValueError, "coefficient conversion failed"):
            fcd.Algebra(dimension=4)(values)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_element_participates_in_cyclic_garbage_collection(self) -> None:
        """Collect a reference cycle passing through an Element parent."""

        class Owner:
            pass

        owner = Owner()
        element = _core._element(owner, 2, [1.0, 2.0])
        self.assertTrue(gc.is_tracked(element))
        owner.element = element
        owner_reference = weakref.ref(owner)

        del element
        del owner
        gc.collect()

        self.assertIsNone(owner_reference())

    def test_private_buffer_kernels_reject_shifted_output_overlap(self) -> None:
        """Reject shifted output overlap across every private buffer kernel."""

        right = np.array([[2.0, -1.0], [0.5, 3.0]])
        third = np.array([[1.5, 0.25], [-2.0, 4.0]])

        def cases(storage: np.ndarray) -> tuple[object, ...]:
            """Build private-kernel calls sharing shifted input and output."""

            output = storage[1:]
            source = storage[:-1]
            return (
                lambda: _core._apply_unary_into(
                    _core._UNARY_CONJUGATE, output, source, 2
                ),
                lambda: _core._apply_binary_into(
                    _core._BINARY_ANTICOMMUTATOR, output, source, right, 2
                ),
                lambda: _core._apply_multiply_into(
                    _core._algorithm("direct", 2), output, source, right, 2
                ),
                lambda: _core._apply_associator_into(
                    output, source, right, third, 2
                ),
                lambda: _core._apply_basis_into(True, 1, output, source, 2),
                lambda: _core._apply_operator_into(
                    True, output, source, right, third, 2
                ),
            )

        for case_index in range(6):
            storage = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            original = storage.copy()
            with self.subTest(case=case_index), self.assertRaisesRegex(
                ValueError, "output must not overlap"
            ):
                cases(storage)[case_index]()
            np.testing.assert_array_equal(storage, original)

        storage = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        original = storage.copy()
        with self.assertRaisesRegex(ValueError, "output must not overlap"):
            _core._apply_scalar_into(
                _core._SCALAR_NORM, storage[1:, 0], storage[:-1], None, 2
            )
        np.testing.assert_array_equal(storage, original)

    def test_public_multiply_allows_disjoint_holes_in_shared_storage(self) -> None:
        """Allow buffers whose address envelopes overlap but touched bytes do not."""

        algebra = fcd.Algebra(dimension=2)
        storage = np.arange(10.0)
        left = np.lib.stride_tricks.as_strided(
            storage, shape=(2, 2), strides=(64, 8)
        )
        output = storage[2:6].reshape(2, 2)
        right = np.ones((2, 2))
        expected = reference.multiply_batch(left.copy(), right)

        self.assertTrue(np.may_share_memory(left, output))
        self.assertFalse(np.shares_memory(left, output))
        result = algebra.multiply(
            algebra.asarray(left, copy=False),
            algebra.asarray(right, copy=False),
            out=output,
        )

        self.assertTrue(np.shares_memory(result.coefficients, output))
        self.assertCoefficientsClose(output, expected)

    def test_private_overlap_check_handles_negative_and_zero_strides(self) -> None:
        """Distinguish disjoint and overlapping records with unusual strides."""

        for batch_stride, source_offset in ((-64, 8), (0, 4)):
            with self.subTest(batch_stride=batch_stride):
                storage = np.arange(12.0)
                source = np.lib.stride_tricks.as_strided(
                    storage[source_offset:],
                    shape=(2, 2),
                    strides=(batch_stride, 8),
                )
                output = np.lib.stride_tricks.as_strided(
                    storage[2:], shape=(2, 2), strides=(32, 8)
                )
                expected = source.copy()
                expected[:, 1:] *= -1.0

                self.assertTrue(np.may_share_memory(source, output))
                self.assertFalse(np.shares_memory(source, output))
                _core._apply_unary_into(
                    _core._UNARY_CONJUGATE, output, source, 2
                )
                self.assertCoefficientsEqual(output, expected)

        overlap_cases = ((-64, 8, slice(1, 5)), (0, 4, slice(3, 7)))
        for batch_stride, source_offset, output_slice in overlap_cases:
            with self.subTest(overlap_stride=batch_stride):
                storage = np.arange(12.0)
                original = storage.copy()
                source = np.lib.stride_tricks.as_strided(
                    storage[source_offset:],
                    shape=(2, 2),
                    strides=(batch_stride, 8),
                )
                output = storage[output_slice].reshape(2, 2)
                with self.assertRaisesRegex(ValueError, "output must not overlap"):
                    _core._apply_unary_into(
                        _core._UNARY_CONJUGATE, output, source, 2
                    )
                np.testing.assert_array_equal(storage, original)

    def test_private_buffer_metadata_overflow_is_rejected_before_access(
        self,
    ) -> None:
        """Reject overflowing buffer spans before dereferencing their storage."""

        aligned_maximum = sys.maxsize - sys.maxsize % 8
        cases = ((4, aligned_maximum), (2, -sys.maxsize - 1))

        for records, stride in cases:
            with self.subTest(stride=stride):
                storage = np.zeros(2)
                source = np.lib.stride_tricks.as_strided(
                    storage, shape=(records, 2), strides=(stride, 8)
                )
                output = np.full((records, 2), 17.0)
                original = output.copy()
                with self.assertRaisesRegex(OverflowError, "buffer"):
                    _core._apply_unary_into(
                        _core._UNARY_CONJUGATE, output, source, 2
                    )
                np.testing.assert_array_equal(output, original)

    def test_private_buffer_kernels_reject_indirect_storage(self) -> None:
        """Reject PEP 3118 buffers that require indirect suboffset access."""

        try:
            from _testbuffer import ndarray
        except ImportError:  # pragma: no cover - CPython test module is optional
            self.skipTest("CPython _testbuffer module is unavailable")

        source = ndarray([1.0, 2.0, 3.0, 4.0], shape=[2, 2], format="d")
        source.add_suboffsets()
        output = np.empty((2, 2))
        with self.assertRaisesRegex(ValueError, "direct strided storage"):
            _core._apply_unary_into(
                _core._UNARY_CONJUGATE, output, source, 2
            )


class PublicSurfaceTests(FastCDTestCase):
    def test_conventional_import_surface_and_version(self) -> None:
        """Verify the exported package surface and semantic version."""

        self.assertEqual(
            fcd.__all__,
            [
                "__version__",
                "Algebra",
                "BasisProduct",
                "Element",
                "ElementArray",
                "cd_element",
            ],
        )
        self.assertRegex(fcd.__version__, r"^\d+\.\d+\.\d+")
        self.assertEqual(
            fcd.__version__, importlib.metadata.version("pyfastcd")
        )
        self.assertEqual(fcd.Algebra.__module__, "pyfastcd._algebra")
        self.assertEqual(fcd.Element.__module__, "pyfastcd")

    def test_every_public_callable_has_documentation(self) -> None:
        """Require runtime documentation on every public callable."""

        public_callables = [
            fcd.Algebra,
            fcd.BasisProduct,
            fcd.Element,
            fcd.ElementArray,
            fcd.cd_element,
        ]
        public_callables.extend(
            value
            for name, value in inspect.getmembers(fcd.Algebra)
            if not name.startswith("_") and callable(value)
        )
        public_callables.extend(
            value
            for name, value in inspect.getmembers(fcd.Element)
            if not name.startswith("_") and callable(value)
        )
        public_callables.extend(
            value
            for name, value in inspect.getmembers(fcd.ElementArray)
            if not name.startswith("_") and callable(value)
        )
        for value in public_callables:
            with self.subTest(callable=value):
                self.assertTrue(inspect.getdoc(value))

    def test_direct_construction_and_subclassing_are_intentionally_closed(self) -> None:
        """Keep construction parent-controlled and public classes final."""

        with self.assertRaises(TypeError):
            fcd.Element([1.0])
        with self.assertRaises(TypeError):
            fcd.ElementArray()
        with self.assertRaises(TypeError):
            type("ElementSubclass", (fcd.Element,), {})
        with self.assertRaises(TypeError):
            type("AlgebraSubclass", (fcd.Algebra,), {})

    def test_multiply_signature_and_removed_diagnostic_methods(self) -> None:
        """Expose keyword-only algorithm selection without diagnostic methods."""

        parameters = inspect.signature(fcd.Algebra.multiply).parameters
        self.assertEqual(list(parameters), ["self", "x", "y", "out", "algorithm"])
        self.assertEqual(parameters["out"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(
            parameters["algorithm"].kind, inspect.Parameter.KEYWORD_ONLY
        )
        self.assertEqual(parameters["algorithm"].default, "auto")
        self.assertIsInstance(fcd.Algebra.default_algorithm, property)
        self.assertTrue(inspect.getdoc(fcd.Algebra.default_algorithm))
        self.assertIsInstance(fcd.Algebra.default_batch_algorithm, property)
        self.assertTrue(inspect.getdoc(fcd.Algebra.default_batch_algorithm))
        for name in (
            "karatsuba_multiply",
            "cc_multiply",
            "reference_multiply",
            "direct_multiply",
            "legacy_hybrid_multiply",
            "_benchmark_multiply",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(fcd.Algebra, name))
                self.assertFalse(hasattr(array_module, name))
        for name in (
            "_benchmark_multiply_element",
            "_apply_benchmark_multiply_into",
            "_BENCHMARK_CC",
            "_BENCHMARK_REFERENCE",
            "_BENCHMARK_DIRECT",
            "_BENCHMARK_LEGACY_HYBRID",
            "_BENCHMARK_KARATSUBA",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(_core, name))


class AlgebraTests(FastCDTestCase):
    def test_dimension_level_cache_and_immutability(self) -> None:
        """Verify algebra caching, level conversion, and immutable metadata."""

        for level in range(8):
            dimension = 1 << level
            by_dimension = fcd.Algebra(dimension=dimension)
            by_level = fcd.Algebra(level=level)
            with self.subTest(level=level):
                self.assertIs(by_dimension, by_level)
                self.assertEqual(by_dimension.dimension, dimension)
                self.assertEqual(by_dimension.level, level)
                self.assertEqual(by_dimension.dtype, np.dtype(np.float64))
                self.assertIn(f"dimension={dimension}", repr(by_dimension))
        algebra = fcd.Algebra(dimension=4)
        with self.assertRaises(AttributeError):
            algebra.dimension = 8
        with self.assertRaises(AttributeError):
            del algebra._dimension
        self.assertEqual(algebra.dimension, 4)

    def test_invalid_algebra_specifications(self) -> None:
        """Reject missing, conflicting, invalid, and oversized specifications."""

        invalid_calls = [
            lambda: fcd.Algebra(),
            lambda: fcd.Algebra(dimension=4, level=2),
            lambda: fcd.Algebra(dimension=0),
            lambda: fcd.Algebra(dimension=-2),
            lambda: fcd.Algebra(dimension=3),
            lambda: fcd.Algebra(level=-1),
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()
        with self.assertRaises(TypeError):
            fcd.Algebra(dimension=True)
        with self.assertRaises(TypeError):
            fcd.Algebra(level=False)
        with self.assertRaises(OverflowError):
            fcd.Algebra(level=10_000)
        with self.assertRaises(OverflowError):
            fcd.Algebra(
                level=(sys.maxsize // struct.calcsize("d")).bit_length()
            )

    def test_scalar_identity_basis_and_inferred_constructor(self) -> None:
        """Verify standard constructors and dimension inference."""

        algebra = fcd.Algebra(dimension=8)
        self.assertCoefficientsEqual(algebra.zero(), np.zeros(8))
        self.assertCoefficientsEqual(algebra.one(), [1, 0, 0, 0, 0, 0, 0, 0])
        self.assertCoefficientsEqual(algebra.scalar(-2.5), [-2.5, 0, 0, 0, 0, 0, 0, 0])
        for index in range(8):
            expected = np.zeros(8)
            expected[index] = 1.0
            with self.subTest(index=index):
                self.assertCoefficientsEqual(algebra.basis(index), expected)

        element = fcd.cd_element((1, 2, 3, 4))
        self.assertIs(element.algebra, fcd.Algebra(dimension=4))
        self.assertCoefficientsEqual(element, [1, 2, 3, 4])
        copied = fcd.cd_element(element)
        self.assertEqual(copied, element)
        self.assertIsNot(copied, element)

    def test_constructor_validation(self) -> None:
        """Reject malformed coefficients, unsupported scalars, and indices."""

        algebra = fcd.Algebra(dimension=4)

        class AllocationFailure:
            def __float__(self) -> float:
                """Raise a synthetic allocation-like conversion failure."""

                raise MemoryError("synthetic allocation failure")

        with self.assertRaises(ValueError):
            algebra([1, 2, 3])
        with self.assertRaises(ValueError):
            algebra(np.ones((1, 4)))
        with self.assertRaises(TypeError):
            algebra([1, 2j, 3, 4])
        with self.assertRaises(TypeError):
            algebra(np.ones(4, dtype=np.complex128))
        with self.assertRaises(TypeError):
            algebra(np.array([1, 2, 3, 4], dtype=object))
        with self.assertRaises(TypeError):
            algebra(np.zeros(4, dtype="V8"))
        with self.assertRaises(TypeError):
            algebra(
                np.arange(
                    "2020-01-01", "2020-01-05", dtype="datetime64[D]"
                )
            )
        with self.assertRaises(TypeError):
            algebra("abcd")
        for unordered in (
            {1.0, 2.0, 3.0, 4.0},
            frozenset({1.0, 2.0, 3.0, 4.0}),
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}.keys(),
        ):
            with self.subTest(unordered=type(unordered).__name__):
                with self.assertRaises(TypeError):
                    algebra(unordered)
        expected = algebra([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(algebra(range(1, 5)), expected)
        self.assertEqual(algebra(float(i) for i in range(1, 5)), expected)
        self.assertEqual(algebra(iter([1, 2, 3, 4])), expected)
        with self.assertRaises(TypeError):
            algebra.scalar(1j)
        with self.assertRaises(TypeError):
            algebra.scalar("1")
        with self.assertRaises(TypeError):
            algebra.scalar(np.array(1, dtype=object))
        with self.assertRaises(MemoryError):
            algebra([1, 2, AllocationFailure(), 4])
        with self.assertRaises(ValueError):
            fcd.cd_element([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            fcd.cd_element([])
        for index in (-1, 4):
            with self.subTest(index=index), self.assertRaises(IndexError):
                algebra.basis(index)

    def test_algebra_multiply_preserves_scalar_result_types(self) -> None:
        """Preserve Element and ElementArray result types for scalar products."""

        algebra = fcd.Algebra(dimension=4)
        element = algebra([1, 2, 3, 4])
        self.assertIsInstance(algebra.multiply(element, 2), fcd.Element)
        self.assertIsInstance(algebra.multiply(2, element), fcd.Element)
        self.assertCoefficientsEqual(algebra.multiply(2, 3), [6, 0, 0, 0])
        out = np.empty(4, dtype=np.float64)
        result = algebra.multiply(2, 3, out=out)
        self.assertIsInstance(result, fcd.ElementArray)
        self.assertCoefficientsEqual(out, [6, 0, 0, 0])

    def test_algebra_and_element_pickle_round_trip(self) -> None:
        """Restore cached algebras and independent elements through pickle."""

        algebra = fcd.Algebra(dimension=16)
        element = algebra(np.arange(16, dtype=np.float64))
        restored_algebra = pickle.loads(pickle.dumps(algebra))
        restored_element = pickle.loads(pickle.dumps(element))
        self.assertIs(restored_algebra, algebra)
        self.assertEqual(restored_element, element)
        self.assertIs(restored_element.algebra, algebra)
        self.assertIsNot(restored_element, element)

    def test_python_copy_protocol_for_immutable_elements(self) -> None:
        """Return independent values through shallow and deep copy protocols."""

        algebra = fcd.Algebra(dimension=4)
        element = algebra([1, 2, 3, 4])
        for copier in (copy.copy, copy.deepcopy):
            restored = copier(element)
            with self.subTest(copier=copier.__name__):
                self.assertEqual(restored, element)
                self.assertIsNot(restored, element)
                self.assertIs(restored.algebra, algebra)


class ElementProtocolTests(FastCDTestCase):
    def setUp(self) -> None:
        """Create one quaternion element for protocol tests."""

        self.algebra = fcd.Algebra(dimension=4)
        self.element = self.algebra([1.0, -2.0, 3.0, -4.0])

    def test_properties_sequence_repr_truth_and_hash(self) -> None:
        """Verify element descriptors and core immutable sequence behavior."""

        element = self.element
        self.assertIsInstance(element, fcd.Element)
        self.assertIs(element.algebra, self.algebra)
        self.assertEqual(element.dimension, 4)
        self.assertEqual(element.level, 2)
        self.assertEqual(element.real, 1.0)
        self.assertEqual(element.shape, (4,))
        self.assertEqual(element.dtype, np.dtype(np.float64))
        self.assertEqual(len(element), 4)
        self.assertEqual(list(element), [1.0, -2.0, 3.0, -4.0])
        self.assertTrue(hasattr(fcd.Element, "__iter__"))
        self.assertEqual(list(iter(element)), [1.0, -2.0, 3.0, -4.0])
        self.assertEqual(element[-1], -4.0)
        sequence_get_item = ctypes.pythonapi.PySequence_GetItem
        sequence_get_item.argtypes = [ctypes.py_object, ctypes.c_ssize_t]
        sequence_get_item.restype = ctypes.py_object
        self.assertEqual(sequence_get_item(element, 0), 1.0)
        self.assertEqual(sequence_get_item(element, -1), -4.0)
        for index in (4, -5, -8):
            with self.subTest(sequence_index=index), self.assertRaises(IndexError):
                sequence_get_item(element, index)
        self.assertEqual(element[1:4:2], (-2.0, -4.0))
        self.assertIn("Element([1.0, -2.0, 3.0, -4.0]", repr(element))
        self.assertTrue(element)
        self.assertFalse(self.algebra.zero())
        self.assertEqual(hash(element), hash(self.algebra(element)))
        self.assertEqual(hash(self.algebra.scalar(3)), hash(3))
        self.assertEqual(hash(self.algebra.scalar(3)), hash(3.0))
        self.assertEqual(hash(self.algebra.scalar(True)), hash(True))
        self.assertEqual(hash(self.algebra([0.0, -0.0, 0.0, 0.0])), hash(0))
        self.assertEqual(
            hash(self.algebra([1.0, -0.0, 2.0, 3.0])),
            hash(self.algebra([1.0, 0.0, 2.0, 3.0])),
        )
        nan_element = self.algebra([math.nan, 1.0, 0.0, 0.0])
        self.assertEqual(hash(nan_element), hash(nan_element))
        self.assertEqual({element: "a", self.algebra(element): "b"}, {element: "b"})
        self.assertEqual(len({element, self.algebra(element), -element}), 2)
        self.assertIn(element, {self.algebra(element)})
        self.assertIsNone(fcd.ElementArray.__hash__)
        with self.assertRaises(TypeError):
            hash(self.algebra.array(np.ones((2, 4))))
        with self.assertRaises(IndexError):
            _ = element[4]
        with self.assertRaises(TypeError):
            _ = element["coordinate"]
        with self.assertRaises(TypeError):
            element[0] = 7.0

    def test_input_and_result_copy_contract(self) -> None:
        """Copy constructor inputs and every immutable element-valued result."""

        source = np.array([1.0, 2.0, 3.0, 4.0])
        element = self.algebra(source)
        source[0] = 99.0
        self.assertEqual(element[0], 1.0)

        ctypes_source = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)
        self.assertCoefficientsEqual(
            self.algebra(memoryview(ctypes_source)), [1.0, 2.0, 3.0, 4.0]
        )
        numeric_buffers = [
            memoryview((ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)),
            memoryview((ctypes.c_int64 * 4)(1, 2, 3, 4)),
            memoryview(np.array([1.0, 2.0, 3.0, 4.0], dtype=">f8")),
        ]
        for buffer in numeric_buffers:
            with self.subTest(format=buffer.format):
                self.assertCoefficientsEqual(
                    self.algebra(buffer), [1.0, 2.0, 3.0, 4.0]
                )

        copies = [
            self.algebra(element),
            element.copy(),
            +element,
            fcd.Algebra(dimension=1)(3.0).conjugate(),
        ]
        originals = [element, element, element, fcd.Algebra(dimension=1)(3.0)]
        for copied, original in zip(copies, originals, strict=True):
            with self.subTest(copied=copied):
                self.assertEqual(copied, original)
                self.assertIsNot(copied, original)

    def test_strided_coefficient_constructors_copy_logical_values(self) -> None:
        """Copy positive and negative coefficient strides through both constructors."""

        for dtype in (np.float64, np.float32, np.int64, ">f8"):
            for step in (2, -2):
                for use_memoryview in (False, True):
                    for constructor in (self.algebra, fcd.cd_element):
                        with self.subTest(
                            dtype=dtype,
                            step=step,
                            memoryview=use_memoryview,
                            constructor=constructor,
                        ):
                            storage = np.arange(8, dtype=dtype)
                            values = storage[::step]
                            expected = values.astype(np.float64, copy=True)
                            source = memoryview(values) if use_memoryview else values
                            element = constructor(source)
                            self.assertIs(element.algebra, self.algebra)
                            self.assertCoefficientsEqual(element, expected)
                            storage[:] = -99
                            self.assertCoefficientsEqual(element, expected)

    def test_strided_coefficient_constructors_reject_invalid_inputs(self) -> None:
        """Retain dtype and shape validation for strided coefficient inputs."""

        invalid_dtypes = (np.complex128, object, "U8")
        for dtype in invalid_dtypes:
            values = np.array(list(range(8)), dtype=dtype)[::2]
            for constructor in (self.algebra, fcd.cd_element):
                with self.subTest(dtype=dtype, constructor=constructor):
                    with self.assertRaises(TypeError):
                        constructor(values)
        for values in (np.arange(6.0)[::2], np.arange(16.0).reshape(4, 4)[::2]):
            for constructor in (self.algebra, fcd.cd_element):
                with self.subTest(shape=values.shape, constructor=constructor):
                    with self.assertRaises(ValueError):
                        constructor(values)

    def test_read_only_buffer_and_numpy_copy_controls(self) -> None:
        """Expose read-only storage and honor explicit NumPy copy controls."""

        view = memoryview(self.element)
        self.assertTrue(view.readonly)
        self.assertEqual(view.format, "d")
        self.assertEqual(view.shape, (4,))
        with self.assertRaises(TypeError):
            view[0] = 10.0

        coefficients = np.asarray(self.element)
        self.assertEqual(coefficients.dtype, np.float64)
        self.assertFalse(coefficients.flags.writeable)
        self.assertTrue(np.shares_memory(coefficients, np.asarray(self.element)))
        with self.assertRaises(ValueError):
            coefficients[0] = 10.0

        no_copy = self.element.to_numpy(copy=False)
        copied = self.element.to_numpy()
        self.assertFalse(no_copy.flags.writeable)
        self.assertTrue(copied.flags.writeable)
        self.assertFalse(np.shares_memory(no_copy, copied))
        copied[0] = 10.0
        self.assertEqual(self.element[0], 1.0)
        self.assertTrue(np.shares_memory(self.element.coefficients, no_copy))
        with self.assertRaises(TypeError):
            self.element.to_numpy(False)
        with self.assertRaises(TypeError):
            self.element.to_numpy(copy=1)

    def test_coordinate_equality_and_scalar_embedding(self) -> None:
        """Compare exact coordinates with consistent real-scalar embedding."""

        self.assertEqual(self.algebra.scalar(3), 3)
        self.assertEqual(3, self.algebra.scalar(3))
        self.assertNotEqual(self.algebra([3, 1, 0, 0]), 3)
        self.assertNotEqual(self.element, self.algebra(self.element) + 1)
        self.assertNotEqual(self.element, fcd.Algebra(dimension=8)(0))
        self.assertFalse(self.element == object())
        for unsupported in (1j, "1", b"1", [1.0, -2.0, 3.0, -4.0], {"real": 1.0}):
            with self.subTest(unsupported=repr(unsupported)):
                self.assertIs(self.element.__eq__(unsupported), NotImplemented)
                self.assertFalse(self.element == unsupported)
                self.assertTrue(self.element != unsupported)
        self.assertIn(self.element, [1j, "1", self.element])
        self.assertEqual(self.element == fcd.Algebra(dimension=8).zero(), False)

        class RaisingFloat:
            def __float__(self) -> float:
                raise ValueError("synthetic conversion failure")

        with self.assertRaises(ValueError):
            _ = self.element == RaisingFloat()
        nan_element = self.algebra([math.nan, 0, 0, 0])
        self.assertNotEqual(nan_element, nan_element)

    def test_foreign_algebra_comparisons_are_unequal_like_numpy(self) -> None:
        """Compare across algebras as all-unequal instead of raising."""

        other = fcd.Algebra(dimension=8)
        batch = self.algebra.array(np.arange(12, dtype=np.float64).reshape(3, 4))
        self.assertFalse(self.element == other.zero())
        self.assertTrue(self.element != other.zero())
        np.testing.assert_array_equal(batch == other.zero(), [False] * 3)
        np.testing.assert_array_equal(batch != other.zero(), [True] * 3)
        foreign = other.array(np.ones((1, 8)))
        np.testing.assert_array_equal(batch == foreign, [False] * 3)
        column = self.algebra.asarray(np.arange(12, dtype=np.float64).reshape(3, 1, 4))
        np.testing.assert_array_equal(
            column != other.array(np.ones((2, 8))), np.ones((3, 2), dtype=bool)
        )
        with self.assertRaises(ValueError):
            _ = batch == other.array(np.ones((2, 8)))

    def test_error_messages_are_shared_across_layers(self) -> None:
        """Report identical conditions with identical messages in C and Python."""

        from pyfastcd import _validation

        other = fcd.Algebra(dimension=8)
        batch = self.algebra.array(np.ones((2, 4)))
        cases: list[tuple[type[Exception], str, list[Any]]] = [
            (
                ValueError,
                _validation.DIFFERENT_ALGEBRA_MESSAGE,
                [
                    lambda: self.element * other.zero(),
                    lambda: self.algebra(other.zero()),
                    lambda: batch * other.zero(),
                    lambda: batch.inner(other.array(np.ones((2, 8)))),
                ],
            ),
            (
                TypeError,
                _validation.COMPLEX_SCALAR_MESSAGE,
                [lambda: self.algebra(1j), lambda: self.algebra.scalar(1j)],
            ),
            (
                TypeError,
                _validation.MODULAR_POWER_MESSAGE,
                [lambda: pow(self.element, 2, 3), lambda: pow(batch, 2, 3)],
            ),
            (
                TypeError,
                _validation.INTEGER_EXPONENT_MESSAGE,
                [lambda: self.element**1.5, lambda: batch**1.5],
            ),
        ]
        for exception, message, operations in cases:
            for operation in operations:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(exception, re.escape(message)):
                        operation()


class ElementArithmeticTests(FastCDTestCase):
    def setUp(self) -> None:
        """Create fixed octonion operands for scalar arithmetic tests."""

        self.algebra = fcd.Algebra(dimension=8)
        self.x_values = np.array([0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0])
        self.y_values = np.array([-1.25, 2.0, 0.5, -3.0, 1.0, -0.5, 0.75, 2.5])
        self.x = self.algebra(self.x_values)
        self.y = self.algebra(self.y_values)

    def test_basic_arithmetic_and_reflected_scalars(self) -> None:
        """Verify coordinate arithmetic and reflected real-scalar operations."""

        self.assertCoefficientsEqual(self.x + self.y, self.x_values + self.y_values)
        self.assertCoefficientsEqual(self.x - self.y, self.x_values - self.y_values)
        self.assertCoefficientsEqual(-self.x, -self.x_values)
        for scalar in (-2.0, 0.5):
            embedded = np.zeros(8)
            embedded[0] = scalar
            with self.subTest(scalar=scalar):
                self.assertCoefficientsEqual(self.x + scalar, self.x_values + embedded)
                self.assertCoefficientsEqual(scalar + self.x, embedded + self.x_values)
                self.assertCoefficientsEqual(self.x - scalar, self.x_values - embedded)
                self.assertCoefficientsEqual(scalar - self.x, embedded - self.x_values)
                self.assertCoefficientsEqual(self.x * scalar, self.x_values * scalar)
                self.assertCoefficientsEqual(scalar * self.x, scalar * self.x_values)
                self.assertCoefficientsEqual(self.x / scalar, self.x_values / scalar)
        with self.assertRaises(ZeroDivisionError):
            _ = self.x / 0

    def test_product_square_and_integer_power(self) -> None:
        """Compare multiplication, square, and integer powers with references."""

        expected = reference.multiply(self.x_values, self.y_values)
        self.assertCoefficientsClose(self.x * self.y, expected)
        self.assertCoefficientsClose(self.algebra.multiply(self.x, self.y), expected)
        self.assertCoefficientsClose(
            self.x.square(), reference.multiply(self.x_values, self.x_values)
        )
        self.assertCoefficientsClose(self.x**2, self.x.square())
        self.assertCoefficientsEqual(self.x**0, self.algebra.one())
        self.assertEqual(self.x**1, self.x)
        self.assertIsNot(self.x**1, self.x)
        for exponent in (3, 4, 5):
            with self.subTest(exponent=exponent):
                self.assertCoefficientsClose(
                    self.x**exponent,
                    reference.left_associated_power(self.x_values, exponent),
                    rtol=3.0e-11,
                    atol=3.0e-11,
                )
        inverse = self.x**-1
        self.assertCoefficientsClose(inverse, self.x.reciprocal())
        with self.assertRaises(TypeError):
            _ = self.x**1.5
        with self.assertRaises(TypeError):
            pow(self.x, 2, 3)
        self.assertCoefficientsClose(self.x ** np.int64(2), self.x.square())

    def test_conjugate_imag_inner_norm_normalize_and_reciprocal(self) -> None:
        """Verify standard unary, inner-product, norm, and reciprocal operations."""

        expected_conjugate = reference.conjugate(self.x_values)
        expected_imag = reference.imag_part(self.x_values)
        self.assertCoefficientsEqual(self.x.conjugate(), expected_conjugate)
        self.assertCoefficientsEqual(self.x.imag, expected_imag)
        self.assertCoefficientsEqual(self.x.imag_part(), expected_imag)
        nonfinite_scalar = self.algebra([math.nan, 1.0, -2.0, 3.0, 0, 0, 0, 0])
        projected = nonfinite_scalar.imag_part()
        self.assertEqual(projected.real, 0.0)
        self.assertFalse(np.signbit(projected.real))
        self.assertTrue(np.all(np.isfinite(projected.coefficients)))
        self.assertEqual(
            self.x.inner(self.y), float(np.dot(self.x_values, self.y_values))
        )
        expected_norm_squared = float(np.dot(self.x_values, self.x_values))
        self.assertEqual(self.x.norm_squared(), expected_norm_squared)
        self.assertEqual(self.x.norm(), math.sqrt(expected_norm_squared))
        self.assertEqual(abs(self.x), self.x.norm())
        self.assertCoefficientsClose(
            self.x.normalized(), self.x_values / np.linalg.norm(self.x_values)
        )
        reciprocal = expected_conjugate / expected_norm_squared
        self.assertCoefficientsClose(self.x.reciprocal(), reciprocal)
        self.assertCoefficientsClose(
            self.x * self.x.reciprocal(), self.algebra.one()
        )
        self.assertCoefficientsClose(
            self.x.reciprocal() * self.x, self.algebra.one()
        )
        with self.assertRaises(ZeroDivisionError):
            self.algebra.zero().normalized()
        with self.assertRaises(ZeroDivisionError):
            self.algebra.zero().reciprocal()

    def test_cross_commutator_anticommutator_and_associator(self) -> None:
        """Compare noncommutative and nonassociative products with references."""

        z_values = np.array([1.0, 0.25, -1.0, 2.0, -0.5, 3.0, 1.5, -2.0])
        z = self.algebra(z_values)
        self.assertCoefficientsClose(
            self.x.imag_cross(self.y),
            reference.imag_cross(self.x_values, self.y_values),
        )
        self.assertCoefficientsClose(
            self.x.commutator(self.y),
            reference.commutator(self.x_values, self.y_values),
        )
        self.assertCoefficientsClose(
            self.x.anticommutator(self.y),
            reference.anticommutator(self.x_values, self.y_values),
        )
        self.assertCoefficientsClose(
            self.x.associator(self.y, z),
            reference.associator(self.x_values, self.y_values, z_values),
        )
        self.assertEqual(self.x.imag_cross(self.y).real, 0.0)
        self.assertCoefficientsClose(
            self.x.commutator(self.y), 2.0 * self.x.imag_cross(self.y)
        )
        nonfinite_x = self.x_values.copy()
        nonfinite_y = self.y_values.copy()
        nonfinite_x[0] = math.nan
        nonfinite_y[0] = math.inf
        zero_scalar_x = self.x_values.copy()
        zero_scalar_y = self.y_values.copy()
        zero_scalar_x[0] = 0.0
        zero_scalar_y[0] = 0.0
        cross = self.algebra(nonfinite_x).imag_cross(self.algebra(nonfinite_y))
        commutator = self.algebra(nonfinite_x).commutator(
            self.algebra(nonfinite_y)
        )
        self.assertCoefficientsClose(
            cross,
            self.algebra(zero_scalar_x).imag_cross(self.algebra(zero_scalar_y)),
        )
        self.assertCoefficientsClose(
            commutator,
            self.algebra(zero_scalar_x).commutator(self.algebra(zero_scalar_y)),
        )
        self.assertFalse(np.signbit(cross.real))
        self.assertFalse(np.signbit(commutator.real))

    def test_mixed_dimensions_and_operand_types_fail_cleanly(self) -> None:
        """Reject mismatched dimensions and unsupported operand types."""

        other = fcd.Algebra(dimension=4)([1, 2, 3, 4])
        for operation in (
            lambda: self.x + other,
            lambda: self.x - other,
            lambda: self.x * other,
            lambda: self.x.inner(other),
            lambda: self.x.associator(self.y, other),
        ):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                operation()
        with self.assertRaises(TypeError):
            self.x.inner(2)
        with self.assertRaises(TypeError):
            self.x.commutator(2)

    def test_numpy_scalar_ownership_and_complex_rejection(self) -> None:
        """Accept real NumPy scalars while rejecting complex and nonnumeric ones."""

        class CustomRealScalar:
            dtype = np.dtype(object)

            def __float__(self) -> float:
                """Expose the fixed real value used by scalar-protocol tests."""

                return 2.0

        embedded_two = np.zeros(8)
        embedded_two[0] = 2.0
        self.assertCoefficientsEqual(
            np.float64(2.0) + self.x, embedded_two + self.x_values
        )
        self.assertCoefficientsEqual(
            np.float64(2.0) - self.x, embedded_two - self.x_values
        )
        self.assertCoefficientsEqual(
            np.int64(2) * self.x, 2.0 * self.x_values
        )
        custom = CustomRealScalar()
        self.assertCoefficientsEqual(self.algebra(custom), embedded_two)
        self.assertCoefficientsEqual(self.algebra([custom] * 8), np.full(8, 2.0))
        self.assertCoefficientsEqual(self.x + custom, self.x_values + embedded_two)
        batch = self.algebra.array(self.x_values)
        self.assertCoefficientsEqual(
            (batch + custom).coefficients, self.x_values + embedded_two
        )
        for value in (
            np.array(2, dtype=np.int16),
            np.array(2, dtype=np.uint64),
            np.array(2, dtype=np.float32),
            np.array(True, dtype=np.bool_),
        ):
            with self.subTest(real_zero_dimensional_array=value):
                scalar = float(value)
                embedded = np.zeros(8)
                embedded[0] = scalar
                self.assertCoefficientsEqual(
                    self.algebra(value), embedded
                )
                self.assertCoefficientsEqual(
                    self.algebra.scalar(value), embedded
                )
                self.assertCoefficientsEqual(
                    self.x + value, self.x_values + embedded
                )
                self.assertCoefficientsEqual(
                    value - self.x, embedded - self.x_values
                )
                self.assertCoefficientsEqual(
                    self.x * value, self.x_values * scalar
                )
                self.assertCoefficientsEqual(
                    self.x / value, self.x_values / scalar
                )
        nonnumeric_scalars = (
            np.array("2"),
            np.array(b"2"),
            np.array(2, dtype=object),
            np.array("2026-09-03", dtype="datetime64[D]"),
            np.array(2, dtype="timedelta64[D]"),
            np.zeros((), dtype="V8"),
        )
        for value in nonnumeric_scalars:
            operations = (
                lambda value=value: self.algebra(value),
                lambda value=value: self.algebra.scalar(value),
                lambda value=value: self.algebra([value] * 8),
                lambda value=value: self.x + value,
                lambda value=value: self.x - value,
                lambda value=value: self.x * value,
                lambda value=value: self.x / value,
                lambda value=value: self.x == value,
            )
            for operation in operations:
                with (
                    self.subTest(nonnumeric_zero_dimensional_array=value),
                    self.assertRaises(TypeError),
                ):
                    operation()
        for value in ("2", b"2"):
            with self.subTest(nonnumeric_sequence_item=value), self.assertRaises(
                TypeError
            ):
                self.algebra([value] * 8)
        for value in (np.complex64(1 + 2j), np.complex128(1 + 2j)):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self.algebra(value)
                with self.assertRaises(TypeError):
                    self.algebra.scalar(value)
                with self.assertRaises(TypeError):
                    _ = self.x + value
                with self.assertRaises(TypeError):
                    _ = self.x - value
                with self.assertRaises(TypeError):
                    _ = self.x * value
                with self.assertRaises(TypeError):
                    _ = self.x / value
                with self.assertRaises(TypeError):
                    _ = self.x == value

    def test_numpy_array_operands_fail_without_recursion(self) -> None:
        """Reject raw coefficient vectors without recursive NumPy dispatch."""

        coefficients = self.x_values.copy()
        operations = (
            lambda: coefficients + self.x,
            lambda: coefficients - self.x,
            lambda: coefficients * self.x,
            lambda: coefficients == self.x,
        )
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaises(TypeError):
                operation()


class MultiplicationOracleTests(FastCDTestCase):
    def test_at_least_one_thousand_products_against_defining_recursion(self) -> None:
        """Compare over one thousand products with defining recursion."""

        # 205 trials in each of five dimensions gives 1,025 independent
        # products while keeping the quadratic Python oracle inexpensive.
        trials_per_dimension = 205
        total_products = 0
        for dimension in (1, 2, 4, 8, 16):
            rng = np.random.default_rng(0xF45CD + dimension)
            left = rng.uniform(-1.0, 1.0, size=(trials_per_dimension, dimension))
            right = rng.uniform(-1.0, 1.0, size=(trials_per_dimension, dimension))
            expected = reference.multiply_batch(left, right)
            actual = fcd.Algebra(dimension=dimension).multiply(left, right)
            with self.subTest(dimension=dimension):
                self.assertCoefficientsClose(actual.coefficients, expected)
            total_products += trials_per_dimension
        self.assertGreaterEqual(total_products, 1_000)

    def test_quasilinear_default_across_multiple_recursion_levels(self) -> None:
        """Exercise the production quasilinear path across nested levels."""

        # Exercise nested complex recursion and Python workspace allocation
        # through the real production interface.
        for dimension in (32, 64, 128):
            rng = np.random.default_rng(64_000 + dimension)
            left = rng.normal(0.0, 0.25, size=(3, dimension))
            right = rng.normal(0.0, 0.25, size=(3, dimension))
            expected = reference.multiply_batch(left, right)
            actual = fcd.Algebra(dimension=dimension).multiply(left, right)
            with self.subTest(dimension=dimension):
                self.assertCoefficientsClose(
                    actual.coefficients, expected, rtol=3.0e-11, atol=3.0e-11
                )


class MultiplicationAlgorithmTests(FastCDTestCase):
    def assertProductWithinBound(
        self, actual: object, expected: object, left: object, right: object
    ) -> None:
        """Assert ``||actual - expected|| <= 128*eps*N*||left||*||right||``.

        Norms are Euclidean and evaluated record by record after broadcasting
        the operand coefficients to the expected shape.
        """

        actual_values = np.asarray(actual, dtype=np.float64)
        expected_values = np.asarray(expected, dtype=np.float64)
        self.assertEqual(actual_values.shape, expected_values.shape)
        dimension = expected_values.shape[-1]
        left_values = np.broadcast_to(
            np.asarray(left, dtype=np.float64), expected_values.shape
        )
        right_values = np.broadcast_to(
            np.asarray(right, dtype=np.float64), expected_values.shape
        )
        bound = (
            128.0
            * np.finfo(np.float64).eps
            * dimension
            * np.linalg.norm(left_values, axis=-1)
            * np.linalg.norm(right_values, axis=-1)
        )
        error = np.linalg.norm(actual_values - expected_values, axis=-1)
        self.assertTrue(
            np.all(error <= bound),
            msg=f"largest error {np.max(error)!r} exceeds bound {np.min(bound)!r}",
        )

    def test_every_algorithm_matches_the_defining_recursion(self) -> None:
        """Compare each algorithm with the oracle on scalars and broadcasts."""

        for dimension in ALGORITHM_DIMENSIONS:
            algebra = fcd.Algebra(dimension=dimension)
            rng = np.random.default_rng(0xA160 + dimension)
            left_values = rng.normal(size=(2, 1, dimension))
            right_values = rng.normal(size=(1, 3, dimension))
            broadcast_left, broadcast_right = np.broadcast_arrays(
                left_values, right_values
            )
            expected = reference.multiply_batch(broadcast_left, broadcast_right)
            left_element = algebra(left_values[0, 0])
            right_element = algebra(right_values[0, 0])
            left_array = algebra.asarray(left_values, copy=False)
            right_array = algebra.asarray(right_values, copy=False)

            for name in _supported_algorithms(dimension):
                with self.subTest(dimension=dimension, algorithm=name):
                    scalar = algebra.multiply(
                        left_element, right_element, algorithm=name
                    )
                    self.assertIsInstance(scalar, fcd.Element)
                    self.assertIs(scalar.algebra, algebra)
                    self.assertProductWithinBound(
                        scalar, expected[0, 0], left_values[0, 0], right_values[0, 0]
                    )

                    strided = algebra.multiply(left_array, right_array, algorithm=name)
                    self.assertIsInstance(strided, fcd.ElementArray)
                    self.assertIs(strided.algebra, algebra)
                    self.assertEqual(strided.shape, expected.shape)
                    self.assertProductWithinBound(
                        strided.coefficients, expected, broadcast_left, broadcast_right
                    )

                    reversed_batch = algebra.multiply(
                        left_array[::-1], right_array, algorithm=name
                    )
                    self.assertProductWithinBound(
                        reversed_batch.coefficients,
                        expected[::-1],
                        broadcast_left[::-1],
                        broadcast_right[::-1],
                    )

                    contiguous = algebra.multiply(
                        np.ascontiguousarray(broadcast_left),
                        np.ascontiguousarray(broadcast_right),
                        algorithm=name,
                    )
                    self.assertProductWithinBound(
                        contiguous.coefficients,
                        expected,
                        broadcast_left,
                        broadcast_right,
                    )

                    broadcast_element = algebra.multiply(
                        left_element, right_array, algorithm=name
                    )
                    self.assertEqual(broadcast_element.shape, (1, 3, dimension))
                    self.assertProductWithinBound(
                        broadcast_element.coefficients,
                        expected[:1],
                        left_values[0, 0],
                        right_values,
                    )

    def test_cariow_cariowa_rejects_dimensions_above_its_maximum(self) -> None:
        """Reject Cariow--Cariowa above dimension 1024, even for empty batches."""

        for dimension in (2048, 4096):
            algebra = fcd.Algebra(dimension=dimension)
            x = algebra.one()
            y = algebra.basis(1)
            batch = algebra.array(np.zeros((2, dimension)))
            empty = algebra.array(np.empty((0, dimension)))
            out = np.full((2, dimension), 17.0)
            cases = (
                ((x, y), {}),
                ((batch, batch), {}),
                ((batch, y), {"out": out}),
                ((empty, empty), {}),
                ((empty, y), {"out": np.empty((0, dimension))}),
                ((np.empty((0, dimension)), np.empty((0, dimension))), {}),
                ((x, 2.0), {}),
                ((2, 3), {}),
            )
            for index, (operands, options) in enumerate(cases):
                with (
                    self.subTest(dimension=dimension, case=index),
                    self.assertRaisesRegex(ValueError, "does not support dimension"),
                ):
                    algebra.multiply(
                        *operands, algorithm="cariow_cariowa", **options
                    )
            np.testing.assert_array_equal(out, 17.0)

        for dimension in (1, CARIOW_CARIOWA_MAX_DIMENSION):
            algebra = fcd.Algebra(dimension=dimension)
            with self.subTest(dimension=dimension):
                self.assertEqual(
                    algebra.multiply(
                        algebra.one(), algebra.one(), algorithm="cariow_cariowa"
                    ),
                    algebra.one(),
                )
                empty = algebra.array(np.empty((0, dimension)))
                self.assertEqual(
                    algebra.multiply(empty, empty, algorithm="cariow_cariowa").shape,
                    (0, dimension),
                )

    def test_invalid_algorithm_names_and_types_are_rejected(self) -> None:
        """Raise ValueError for unknown names and TypeError for non-strings."""

        algebra = fcd.Algebra(dimension=4)
        x = algebra([1.0, 2.0, 3.0, 4.0])
        y = algebra([0.5, -1.0, 2.0, -3.0])
        batch = algebra.array(np.ones((2, 4)))
        empty = algebra.array(np.empty((0, 4)))
        operand_pairs = ((x, y), (batch, y), (empty, empty), (x, 2.0), (2, 3))
        unknown_names = (
            "bogus",
            "AUTO",
            "Direct",
            "cariow-cariowa",
            "quasi",
            "",
            "auto\x00",
            " auto",
        )
        for name in unknown_names:
            for left, right in operand_pairs:
                with (
                    self.subTest(name=name, left=type(left), right=type(right)),
                    self.assertRaisesRegex(
                        ValueError, "unknown multiplication algorithm"
                    ),
                ):
                    algebra.multiply(left, right, algorithm=name)
        for value in (0, 1, 3, 1.5, None, True, b"auto", ("auto",), ["auto"]):
            for left, right in operand_pairs:
                with (
                    self.subTest(value=value, left=type(left), right=type(right)),
                    self.assertRaisesRegex(TypeError, "algorithm must be a str"),
                ):
                    algebra.multiply(left, right, algorithm=value)
        with self.assertRaises(TypeError):
            algebra.multiply(x, y, None, "auto")
        self.assertCoefficientsEqual(
            algebra.multiply(x, y, algorithm=np.str_("auto")), x * y
        )

    def test_default_algorithm_names_the_automatic_selection(self) -> None:
        """Expose both automatic selections and reproduce operators bitwise."""

        for dimension in ALGORITHM_DIMENSIONS:
            algebra = fcd.Algebra(dimension=dimension)
            name = algebra.default_algorithm
            batch_name = algebra.default_batch_algorithm
            with self.subTest(dimension=dimension, default=name, batch=batch_name):
                self.assertIsInstance(name, str)
                self.assertIsInstance(batch_name, str)
                self.assertIn(name, CONCRETE_ALGORITHM_NAMES)
                self.assertIn(batch_name, CONCRETE_ALGORITHM_NAMES)
                if dimension == 4 or dimension >= 64:
                    self.assertEqual(name, "quasilinear")
                    self.assertEqual(batch_name, "quasilinear")
                self.assertIn(name, _supported_algorithms(dimension))
                self.assertIn(batch_name, _supported_algorithms(dimension))
                self.assertEqual(algebra.default_algorithm, name)
                self.assertEqual(algebra.default_batch_algorithm, batch_name)

                rng = np.random.default_rng(0xDEFA + dimension)
                x = algebra(rng.normal(size=dimension))
                y = algebra(rng.normal(size=dimension))
                left_values = rng.normal(size=(3, 1, dimension))
                right_values = rng.normal(size=(1, 2, dimension))
                left = algebra.array(left_values)
                right = algebra.array(right_values)

                product = x * y
                self.assertCoefficientsEqual(
                    algebra.multiply(x, y, algorithm=name), product
                )
                self.assertCoefficientsEqual(
                    algebra.multiply(x, y, algorithm="auto"), product
                )
                self.assertCoefficientsEqual(algebra.multiply(x, y), product)
                self.assertCoefficientsEqual(np.multiply(x, y), product)
                self.assertCoefficientsEqual(x**3, x * x.square())

                batch = left * right
                for explicit in (batch_name, "auto"):
                    self.assertCoefficientsEqual(
                        algebra.multiply(
                            left, right, algorithm=explicit
                        ).coefficients,
                        batch.coefficients,
                    )
                    self.assertCoefficientsEqual(
                        algebra.multiply(
                            np.ascontiguousarray(
                                np.broadcast_to(left_values, batch.shape)
                            ),
                            np.ascontiguousarray(
                                np.broadcast_to(right_values, batch.shape)
                            ),
                            algorithm=explicit,
                        ).coefficients,
                        batch.coefficients,
                    )
                self.assertCoefficientsEqual(
                    np.multiply(left, right).coefficients, batch.coefficients
                )
                self.assertCoefficientsEqual(
                    (left**3).coefficients, (left * left.square()).coefficients
                )

        with self.assertRaises(AttributeError):
            fcd.Algebra(dimension=4).default_algorithm = "direct"
        with self.assertRaises(AttributeError):
            fcd.Algebra(dimension=4).default_batch_algorithm = "direct"

    def test_single_record_batches_use_the_single_product_selection(self) -> None:
        """Route one-record batches through the single-product default."""

        for dimension in ALGORITHM_DIMENSIONS:
            algebra = fcd.Algebra(dimension=dimension)
            rng = np.random.default_rng(0xBA7C + dimension)
            x = algebra.array(rng.normal(size=(1, dimension)))
            y = algebra.array(rng.normal(size=(1, dimension)))
            with self.subTest(dimension=dimension):
                self.assertCoefficientsEqual(
                    (x * y).coefficients,
                    algebra.multiply(
                        x, y, algorithm=algebra.default_algorithm
                    ).coefficients,
                )
                two_x = algebra.array(rng.normal(size=(2, dimension)))
                two_y = algebra.array(rng.normal(size=(2, dimension)))
                self.assertCoefficientsEqual(
                    (two_x * two_y).coefficients,
                    algebra.multiply(
                        two_x, two_y, algorithm=algebra.default_batch_algorithm
                    ).coefficients,
                )

    def test_out_is_reused_with_every_algorithm(self) -> None:
        """Reuse wrapped and raw outputs for every algorithm and batch path."""

        algebra = fcd.Algebra(dimension=8)
        rng = np.random.default_rng(0x0A7)
        left_values = rng.normal(size=(2, 1, 8))
        right_values = rng.normal(size=(1, 3, 8))
        broadcast_left, broadcast_right = np.broadcast_arrays(
            left_values, right_values
        )
        expected = reference.multiply_batch(broadcast_left, broadcast_right)
        left_element = algebra(left_values[0, 0])
        right_element = algebra(right_values[0, 0])
        right = algebra.asarray(right_values, copy=False)

        for name in ALGORITHM_NAMES:
            with self.subTest(algorithm=name):
                out = algebra.array(np.empty_like(expected))
                returned = algebra.multiply(
                    left_values, right, out=out, algorithm=name
                )
                self.assertIs(returned, out)
                self.assertProductWithinBound(
                    out.coefficients, expected, broadcast_left, broadcast_right
                )

                raw_out = np.empty_like(expected)
                wrapped = algebra.multiply(
                    left_values, right_values, out=raw_out, algorithm=name
                )
                self.assertIsInstance(wrapped, fcd.ElementArray)
                self.assertTrue(np.shares_memory(wrapped.coefficients, raw_out))
                self.assertProductWithinBound(
                    raw_out, expected, broadcast_left, broadcast_right
                )

                contiguous_out = np.empty_like(expected)
                algebra.multiply(
                    np.ascontiguousarray(broadcast_left),
                    np.ascontiguousarray(broadcast_right),
                    out=contiguous_out,
                    algorithm=name,
                )
                self.assertProductWithinBound(
                    contiguous_out, expected, broadcast_left, broadcast_right
                )

                scalar_out = np.empty(8, dtype=np.float64)
                scalar_result = algebra.multiply(
                    left_element, right_element, out=scalar_out, algorithm=name
                )
                self.assertIsInstance(scalar_result, fcd.ElementArray)
                self.assertTrue(
                    np.shares_memory(scalar_result.coefficients, scalar_out)
                )
                self.assertProductWithinBound(
                    scalar_out, expected[0, 0], left_values[0, 0], right_values[0, 0]
                )

    def test_large_batches_use_every_algorithm_on_both_native_paths(self) -> None:
        """Run batches past the GIL-release threshold on strided and packed paths."""

        algebra = fcd.Algebra(dimension=4)
        rng = np.random.default_rng(0x61A)
        left_values = rng.normal(size=(40, 1, 4))
        right_values = rng.normal(size=(1, 3, 4))
        broadcast_left, broadcast_right = np.broadcast_arrays(
            left_values, right_values
        )
        expected = reference.multiply_batch(broadcast_left, broadcast_right)
        left = algebra.asarray(left_values, copy=False)
        right = algebra.asarray(right_values, copy=False)
        packed_left = algebra.array(broadcast_left)
        packed_right = algebra.array(broadcast_right)

        for name in ALGORITHM_NAMES:
            with self.subTest(algorithm=name):
                strided = algebra.multiply(left, right, algorithm=name)
                self.assertProductWithinBound(
                    strided.coefficients, expected, broadcast_left, broadcast_right
                )
                packed = algebra.multiply(packed_left, packed_right, algorithm=name)
                self.assertProductWithinBound(
                    packed.coefficients, expected, broadcast_left, broadcast_right
                )
                broadcast_right_only = algebra.multiply(
                    packed_left, algebra(right_values[0, 0]), algorithm=name
                )
                self.assertProductWithinBound(
                    broadcast_right_only.coefficients,
                    reference.multiply_batch(
                        broadcast_left,
                        np.broadcast_to(right_values[0, 0], broadcast_left.shape),
                    ),
                    broadcast_left,
                    right_values[0, 0],
                )

    def test_simd_batches_and_odd_tail_match_scalar_products(self) -> None:
        """Preserve scalar rounding with paired, strided, and broadcast inputs."""

        rng = np.random.default_rng(20260906)
        for dimension in (16, 32, 64, 128):
            algebra = fcd.Algebra(dimension=dimension)
            left = rng.normal(size=(5, dimension + 1))[:, :dimension]
            right = rng.normal(size=(5, dimension + 1))[:, :dimension]
            for name in ALGORITHM_NAMES:
                if (name == "cariow_cariowa"
                        and dimension > CARIOW_CARIOWA_MAX_DIMENSION):
                    continue
                for broadcast in (False, True):
                    with self.subTest(
                        dimension=dimension, algorithm=name, broadcast=broadcast
                    ):
                        rhs = right[0] if broadcast else right
                        # Batches of two or more records follow the batch
                        # selection; reproduce it with single products.
                        scalar_name = (
                            algebra.default_batch_algorithm
                            if name == "auto"
                            else name
                        )
                        expected = np.array(
                            [
                                algebra.multiply(
                                    algebra(left[i]),
                                    algebra(right[0] if broadcast else right[i]),
                                    algorithm=scalar_name,
                                ).coefficients
                                for i in range(5)
                            ]
                        )
                        for count in (2, 5):
                            storage = np.full(count * dimension + 2, 19.0)
                            output = storage[1:-1].reshape(count, dimension)
                            algebra.multiply(
                                algebra.asarray(left[:count], copy=False),
                                algebra.asarray(
                                    rhs if broadcast else rhs[:count], copy=False
                                ),
                                out=output,
                                algorithm=name,
                            )
                            np.testing.assert_array_equal(
                                output, expected[:count]
                            )
                            np.testing.assert_array_equal(storage[[0, -1]], 19.0)

    def test_validation_is_shared_by_every_algorithm(self) -> None:
        """Apply parent, overlap, shape, and dtype validation for every algorithm."""

        algebra = fcd.Algebra(dimension=8)
        other = fcd.Algebra(dimension=4).one()
        left = algebra.array(np.arange(16, dtype=np.float64).reshape(2, 8))
        right = algebra.array(np.linspace(-1.0, 1.0, 16).reshape(2, 8))
        flat = np.arange(24, dtype=np.float64)
        overlapping_left = algebra.asarray(flat[:16].reshape(2, 8), copy=False)
        overlapping_right = algebra.asarray(flat[1:17].reshape(2, 8), copy=False)

        for name in ALGORITHM_NAMES:
            with self.subTest(algorithm=name, error="parent"):
                with self.assertRaises(ValueError):
                    algebra.multiply(algebra.one(), other, algorithm=name)
            with self.subTest(algorithm=name, error="output overlap"):
                with self.assertRaises(ValueError):
                    algebra.multiply(left, right, out=left, algorithm=name)
            with self.subTest(algorithm=name, error="input overlap"):
                with self.assertRaises(ValueError):
                    algebra.multiply(
                        overlapping_left, overlapping_right, algorithm=name
                    )
            with self.subTest(algorithm=name, error="shape"):
                with self.assertRaises(ValueError):
                    algebra.multiply(
                        left, right, out=np.empty((1, 8)), algorithm=name
                    )
            with self.subTest(algorithm=name, error="dtype"):
                with self.assertRaises(TypeError):
                    algebra.multiply(
                        left,
                        right,
                        out=np.empty((2, 8), dtype=np.float32),
                        algorithm=name,
                    )
            with self.subTest(algorithm=name, error="operator"):
                with self.assertRaises(TypeError):
                    _ = left * right.coefficients

    def test_inputs_may_share_storage_when_records_never_conflict(self) -> None:
        """Accept overlapping read-only views whose paired records never conflict."""

        algebra = fcd.Algebra(dimension=8)
        storage = np.arange(40, dtype=np.float64).reshape(5, 8)
        views = {
            "adjacent windows": (storage[:-1], storage[1:]),
            "broadcast first record": (storage, storage[:1]),
            "reversed": (storage, storage[::-1]),
            "outer product": (storage[:, None, :], storage),
        }
        for label, (left_view, right_view) in views.items():
            self.assertTrue(np.shares_memory(left_view, right_view))
            left = algebra.asarray(left_view, copy=False)
            right = algebra.asarray(right_view, copy=False)
            left_copy = algebra.array(left_view)
            right_copy = algebra.array(right_view)
            for name in ALGORITHM_NAMES:
                with self.subTest(case=label, algorithm=name):
                    result = algebra.multiply(left, right, algorithm=name)
                    expected = algebra.multiply(left_copy, right_copy, algorithm=name)
                    np.testing.assert_array_equal(
                        result.coefficients, expected.coefficients
                    )
            with self.subTest(case=label, operation="inner"):
                np.testing.assert_array_equal(
                    left.inner(right), left_copy.inner(right_copy)
                )
            with self.subTest(case=label, operation="associator"):
                np.testing.assert_array_equal(
                    left.associator(right, left).coefficients,
                    left_copy.associator(right_copy, left_copy).coefficients,
                )
        with self.subTest(case="dimension one"):
            reals = fcd.Algebra(dimension=1)
            line = np.arange(6, dtype=np.float64).reshape(6, 1)
            product = reals.multiply(
                reals.asarray(line, copy=False), reals.asarray(line[::-1], copy=False)
            )
            np.testing.assert_array_equal(
                product.coefficients, (line * line[::-1])
            )

    def test_dual_protocol_vectors_prefer_array_semantics(self) -> None:
        """Prefer coefficient-array semantics for dual-protocol vectors."""

        algebra = fcd.Algebra(dimension=8)
        left_values = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
        right_values = np.linspace(0.75, -0.5, 8, dtype=np.float64)
        right = algebra(right_values)
        expected = reference.multiply(left_values, right_values)
        producers = (
            _DualArrayProducer(left_values),
            _DualDLPackProducer(left_values),
        )

        for producer in producers:
            for name in ALGORITHM_NAMES:
                with self.subTest(
                    producer=type(producer).__name__, algorithm=name
                ):
                    result = algebra.multiply(producer, right, algorithm=name)
                    self.assertIsInstance(result, fcd.ElementArray)
                    self.assertCoefficientsClose(result.coefficients, expected)

    def test_dual_protocol_zero_dimensional_values_remain_scalars(self) -> None:
        """Keep zero-dimensional dual-protocol inputs on the scalar path."""

        algebra = fcd.Algebra(dimension=8)
        values = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
        element = algebra(values)
        scalar = 2.5
        producers = (
            _DualArrayProducer(np.array(scalar), scalar),
            _DualDLPackProducer(np.array(scalar), scalar),
        )

        for producer in producers:
            for name in ALGORITHM_NAMES:
                with self.subTest(
                    producer=type(producer).__name__, algorithm=name
                ):
                    result = algebra.multiply(element, producer, algorithm=name)
                    self.assertIsInstance(result, fcd.Element)
                    self.assertCoefficientsClose(result, values * scalar)


class BasisAndOperatorTests(FastCDTestCase):
    def test_basis_product_is_complete_through_sedenions(self) -> None:
        """Exhaustively compare basis products through dimension 16."""

        for dimension in (1, 2, 4, 8, 16):
            algebra = fcd.Algebra(dimension=dimension)
            for left in range(dimension):
                for right in range(dimension):
                    expected_index, expected_sign = reference.basis_product(
                        dimension, left, right
                    )
                    actual = algebra.basis_product(left, right)
                    with self.subTest(
                        dimension=dimension, left=left, right=right
                    ):
                        self.assertIsInstance(actual, fcd.BasisProduct)
                        self.assertEqual(actual.index, left ^ right)
                        self.assertEqual(
                            (actual.index, actual.sign),
                            (expected_index, expected_sign),
                        )
        product = fcd.Algebra(dimension=4).basis_product(1, 2)
        index, sign = product
        self.assertEqual((index, sign), (3, 1))
        self.assertEqual(product, fcd.BasisProduct(index=3, sign=1))
        self.assertEqual(hash(product), hash(fcd.BasisProduct(3, 1)))
        self.assertIsInstance(product.index, int)
        with self.assertRaises(AttributeError):
            product.index = 0  # type: ignore[misc]

    def test_scalar_basis_actions_match_full_multiplication(self) -> None:
        """Match specialized scalar basis actions with full products."""

        algebra = fcd.Algebra(dimension=16)
        values = np.linspace(-2.0, 2.0, 16)
        x = algebra(values)
        for index in range(16):
            basis = np.zeros(16)
            basis[index] = 1.0
            with self.subTest(index=index):
                self.assertCoefficientsClose(
                    algebra.left_basis_multiply(index, x),
                    reference.multiply(basis, values),
                )
                self.assertCoefficientsClose(
                    algebra.right_basis_multiply(x, index),
                    reference.multiply(values, basis),
                )

    def test_matrix_free_operator_commutators_match_compositions(self) -> None:
        """Match matrix-free operator commutators with explicit compositions."""

        algebra = fcd.Algebra(dimension=16)
        rng = np.random.default_rng(90210)
        a_values, b_values, x_values = rng.normal(size=(3, 16))
        a, b, x = map(algebra, (a_values, b_values, x_values))
        self.assertCoefficientsClose(
            algebra.left_operator_commutator(a, b, x),
            reference.left_operator_commutator(a_values, b_values, x_values),
            rtol=2.0e-11,
            atol=2.0e-11,
        )
        self.assertCoefficientsClose(
            algebra.right_operator_commutator(a, b, x),
            reference.right_operator_commutator(a_values, b_values, x_values),
            rtol=2.0e-11,
            atol=2.0e-11,
        )


class ElementArrayConstructionTests(FastCDTestCase):
    def setUp(self) -> None:
        """Create the quaternion algebra used by array construction tests."""

        self.algebra = fcd.Algebra(dimension=4)

    def test_shape_properties_indexing_assignment_and_truth(self) -> None:
        """Verify array metadata, indexing, assignment, iteration, and truth."""

        source = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        array = self.algebra.array(source)
        self.assertIsInstance(array, fcd.ElementArray)
        self.assertIs(array.algebra, self.algebra)
        self.assertEqual(array.dimension, 4)
        self.assertEqual(array.level, 2)
        self.assertEqual(array.dtype, np.dtype(np.float64))
        self.assertEqual(array.shape, (2, 3, 4))
        self.assertEqual(array.batch_shape, (2, 3))
        self.assertCoefficientsEqual(array.real, source[..., 0])
        self.assertIsInstance(array[0], fcd.ElementArray)
        self.assertIsInstance(array[0, 1], fcd.Element)
        self.assertEqual(len(list(array)), 2)
        self.assertTrue(all(isinstance(item, fcd.ElementArray) for item in array))
        row_batch = self.algebra.array(source.reshape(6, 4))
        self.assertTrue(all(isinstance(item, fcd.Element) for item in row_batch))
        self.assertCoefficientsEqual(array[0, 1], source[0, 1])
        self.assertIn("ElementArray(shape=(2, 3, 4)", repr(array))
        with self.assertRaises(ValueError):
            bool(array)

        array[0, 0] = self.algebra([9, 8, 7, 6])
        array[0, 1] = [5, 4, 3, 2]
        self.assertCoefficientsEqual(array[0, 0], [9, 8, 7, 6])
        self.assertCoefficientsEqual(array[0, 1], [5, 4, 3, 2])

    def test_array_always_copies_while_asarray_obeys_copy(self) -> None:
        """Distinguish unconditional array copies from asarray copy policy."""

        source = np.arange(12, dtype=np.float64).reshape(3, 4)
        copied = self.algebra.array(source)
        wrapped = self.algebra.asarray(source)
        required_view = self.algebra.asarray(source, copy=False)
        forced_copy = self.algebra.asarray(source, copy=True)
        self.assertFalse(np.shares_memory(copied.coefficients, source))
        self.assertTrue(np.shares_memory(wrapped.coefficients, source))
        self.assertTrue(np.shares_memory(required_view.coefficients, source))
        self.assertFalse(np.shares_memory(forced_copy.coefficients, source))
        source[0, 0] = -50.0
        self.assertNotEqual(copied.coefficients[0, 0], -50.0)
        self.assertEqual(wrapped.coefficients[0, 0], -50.0)

        noncontiguous = source[:, ::-1]
        converted = self.algebra.asarray(noncontiguous)
        self.assertTrue(converted.coefficients.flags.c_contiguous)
        with self.assertRaises(ValueError):
            self.algebra.asarray(noncontiguous, copy=False)
        with self.assertRaises(ValueError):
            self.algebra.asarray(
                np.arange(12, dtype=np.int64).reshape(3, 4), copy=False
            )
        with self.assertRaises(ValueError):
            self.algebra.asarray([1, 2, 3, 4], copy=False)
        with self.assertRaises(TypeError):
            self.algebra.asarray(source, copy=np.bool_(False))

    def test_incompatible_alignment_forces_an_aligned_copy(self) -> None:
        """Copy unaligned storage or reject it when copying is forbidden."""

        storage = np.zeros(33, dtype=np.uint8)
        unaligned = np.ndarray(
            (4,), dtype=np.float64, buffer=storage, offset=1
        )
        self.assertFalse(unaligned.flags.aligned)
        converted = self.algebra.asarray(unaligned)
        self.assertTrue(converted.coefficients.flags.aligned)
        self.assertFalse(np.shares_memory(converted.coefficients, unaligned))
        self.assertCoefficientsEqual(converted.square(), np.zeros(4))
        with self.assertRaises(ValueError):
            self.algebra.asarray(unaligned, copy=False)

    def test_unused_singleton_stride_is_zero_copy_compatible(self) -> None:
        """Allow irrelevant singleton strides in zero-copy coefficient views."""

        source = np.arange(9, dtype=np.float64)
        view = np.lib.stride_tricks.as_strided(
            source, shape=(1, 2, 4), strides=(1, 40, 8)
        )
        self.assertTrue(view.flags.aligned)
        wrapped = self.algebra.asarray(view, copy=False)
        self.assertTrue(np.shares_memory(wrapped.coefficients, source))
        self.assertCoefficientsClose(
            wrapped.square().coefficients,
            reference.multiply_batch(view, view),
        )

    def test_metadata_is_insulated_while_coefficient_writes_are_shared(self) -> None:
        """Insulate exported metadata while sharing coefficient mutations."""

        source = np.arange(12, dtype=np.float64).reshape(3, 4)
        array = self.algebra.asarray(source, copy=False)
        exposed = array.coefficients
        exposed.flags.writeable = False
        self.assertEqual(array.shape, (3, 4))
        self.assertEqual(array.dtype, np.dtype(np.float64))
        self.assertTrue(array.coefficients.flags.writeable)
        array.coefficients[0, 0] = -17.0
        self.assertEqual(source[0, 0], -17.0)

    def test_dimension_one_zero_stride_view(self) -> None:
        """Support broadcast zero-stride storage at dimension one."""

        algebra = fcd.Algebra(dimension=1)
        source = np.broadcast_to(np.array([2.0]), (5, 1))
        wrapped = algebra.asarray(source, copy=False)
        self.assertTrue(np.shares_memory(wrapped.coefficients, source))
        self.assertCoefficientsEqual((wrapped * wrapped).coefficients, 4.0)

    def test_empty_c_contiguous_array_ignores_reported_strides(self) -> None:
        """Preserve harmless reported strides on an empty contiguous array."""

        storage = np.empty(1, dtype=np.float64)
        empty = np.lib.stride_tricks.as_strided(
            storage, shape=(0, 4), strides=(32, -8)
        )
        self.assertTrue(empty.flags.aligned)
        self.assertTrue(empty.flags.c_contiguous)
        wrapped = self.algebra.asarray(empty, copy=False)
        self.assertEqual(wrapped.shape, empty.shape)
        self.assertEqual(wrapped.coefficients.strides, empty.strides)

    def test_zero_batch_real_is_a_zero_dimensional_view(self) -> None:
        """Expose scalar-batch real coordinates as a shared zero-dimensional view."""

        source = np.arange(4, dtype=np.float64)
        wrapped = self.algebra.asarray(source, copy=False)
        real = wrapped.real
        self.assertIsInstance(real, np.ndarray)
        self.assertEqual(real.shape, ())
        self.assertTrue(np.shares_memory(real, source))
        real[...] = -9.0
        self.assertEqual(source[0], -9.0)

    def test_array_constructor_validation(self) -> None:
        """Reject incompatible shapes, dtypes, and algebra parents."""

        with self.assertRaises(ValueError):
            self.algebra.array(np.ones((2, 3)))
        with self.assertRaises(TypeError):
            self.algebra.array(np.ones((2, 4), dtype=np.complex128))
        with self.assertRaises(TypeError):
            self.algebra.array(np.ones((2, 4), dtype=object))
        other = fcd.Algebra(dimension=8).array(np.ones((2, 8)))
        with self.assertRaises(ValueError):
            self.algebra.asarray(other)

    def test_strided_leading_axes_scalar_batch_and_empty_batch(self) -> None:
        """Support strided, scalar-batch, and empty-batch array layouts."""

        algebra = fcd.Algebra(dimension=8)
        storage = np.arange(6 * 2 * 8, dtype=np.float64).reshape(6, 2, 8)
        source = storage[::2]
        wrapped = algebra.asarray(source, copy=False)
        self.assertTrue(np.shares_memory(wrapped.coefficients, storage))
        self.assertEqual(wrapped.coefficients.strides, source.strides)
        expected = reference.multiply_batch(source, source)
        self.assertCoefficientsClose((wrapped * wrapped).coefficients, expected)

        scalar_batch = algebra.array(np.arange(8, dtype=np.float64))
        self.assertEqual(scalar_batch.batch_shape, ())
        self.assertIsInstance(scalar_batch[()], fcd.Element)
        with self.assertRaises(TypeError):
            list(scalar_batch)
        self.assertCoefficientsClose(
            (scalar_batch * scalar_batch).coefficients,
            reference.multiply(np.arange(8), np.arange(8)),
        )

        empty = algebra.array(np.empty((0, 8), dtype=np.float64))
        self.assertEqual((empty * empty).shape, (0, 8))
        self.assertEqual(empty.conjugate().shape, (0, 8))
        self.assertEqual(empty.norm().shape, (0,))

        maximum_level = (sys.maxsize // struct.calcsize("d")).bit_length() - 1
        large_level = min(30, maximum_level)
        large_dimension = 1 << large_level
        large_source = np.empty((0, large_dimension), dtype=np.float64)
        large_empty = fcd.Algebra(level=large_level).asarray(
            large_source, copy=False
        )
        self.assertIs(large_empty.coefficients.base, large_source)
        self.assertEqual((large_empty * large_empty).shape, large_source.shape)
        self.assertEqual(large_empty.conjugate().shape, large_source.shape)
        self.assertEqual(large_empty.norm().shape, (0,))

    def test_numpy_conversion_copy_and_dlpack(self) -> None:
        """Verify NumPy copy controls and the supported DLPack contract."""

        class Producer:
            def __init__(self, coefficients: np.ndarray) -> None:
                """Store coefficients and a log of DLPack requests."""

                self.coefficients = coefficients
                self.requests: list[dict[str, object]] = []

            def __dlpack__(self, **kwargs: object) -> object:
                """Record and forward one DLPack export request."""

                self.requests.append(dict(kwargs))
                return self.coefficients.__dlpack__(**kwargs)

            def __dlpack_device__(self) -> tuple[int, int]:
                """Forward the coefficient array's DLPack device."""

                return self.coefficients.__dlpack_device__()

        source = np.arange(12, dtype=np.float64).reshape(3, 4)
        array = self.algebra.asarray(source)
        no_copy = array.to_numpy(copy=False)
        copied = array.to_numpy()
        implicit = np.asarray(array)
        self.assertTrue(np.shares_memory(no_copy, source))
        self.assertTrue(np.shares_memory(implicit, source))
        self.assertFalse(np.shares_memory(copied, source))
        with self.assertRaises(TypeError):
            array.to_numpy(copy=1)
        with self.assertRaises(TypeError):
            array.__array__(copy=1)
        self.assertTrue(np.shares_memory(np.array(array, copy=False), source))
        self.assertFalse(np.shares_memory(np.array(array, copy=True), source))
        dlpack = np.from_dlpack(array)
        self.assertCoefficientsEqual(dlpack, source)
        self.assertTrue(np.shares_memory(dlpack, source))
        self.assertEqual(array.__dlpack_device__()[0], 1)
        dlpack_parameters = array_module._FROM_DLPACK_PARAMETERS
        if "copy" in dlpack_parameters:
            no_copy_dlpack = np.from_dlpack(array, copy=False)
            copied_dlpack = np.from_dlpack(array, copy=True)
            self.assertTrue(np.shares_memory(no_copy_dlpack, source))
            self.assertFalse(np.shares_memory(copied_dlpack, source))
            self.assertCoefficientsEqual(copied_dlpack, source)

            readonly_source = source.copy()
            readonly_source.flags.writeable = False
            readonly = self.algebra.asarray(readonly_source, copy=False)
            readonly_shared = np.from_dlpack(readonly, copy=False)
            readonly_copy = np.from_dlpack(readonly, copy=True)
            self.assertTrue(np.shares_memory(readonly_shared, readonly_source))
            self.assertFalse(np.shares_memory(readonly_copy, readonly_source))

            parameters = array_module._NDARRAY_DLPACK_PARAMETERS
            try:
                array_module._NDARRAY_DLPACK_PARAMETERS = frozenset(
                    {"self", "stream"}
                )
                legacy_shared = np.from_dlpack(array, copy=False)
                legacy_copy = np.from_dlpack(array, copy=True)
                legacy_readonly = np.from_dlpack(readonly)
                self.assertTrue(np.shares_memory(legacy_shared, source))
                self.assertFalse(np.shares_memory(legacy_copy, source))
                self.assertFalse(
                    np.shares_memory(legacy_readonly, readonly_source)
                )
                with self.assertRaises(BufferError):
                    np.from_dlpack(readonly, copy=False)
            finally:
                array_module._NDARRAY_DLPACK_PARAMETERS = parameters
        if "device" in dlpack_parameters:
            cpu_dlpack = np.from_dlpack(array, device="cpu")
            self.assertTrue(np.shares_memory(cpu_dlpack, source))
        if {"copy", "device"}.issubset(dlpack_parameters):
            for device_type in (3, 11, 13):
                with self.subTest(host_dlpack_device_type=device_type):
                    host_array = self.algebra.asarray(
                        _HostDeviceProducer(source, device_type), copy=False
                    )
                    self.assertEqual(
                        host_array.__dlpack_device__(), (device_type, 0)
                    )
                    host_as_cpu = np.from_dlpack(
                        host_array, device="cpu", copy=False
                    )
                    self.assertEqual(host_as_cpu.__dlpack_device__(), (1, 0))
                    self.assertTrue(np.shares_memory(host_as_cpu, source))

                    parameters = array_module._NDARRAY_DLPACK_PARAMETERS
                    try:
                        array_module._NDARRAY_DLPACK_PARAMETERS = frozenset(
                            {"self", "stream"}
                        )
                        with self.assertRaises(BufferError):
                            host_array.__dlpack__(copy=True)
                        with self.assertRaises(BufferError):
                            host_array.__dlpack__(
                                dl_device=(1, 0), copy=False
                            )
                        capsule = host_array.__dlpack__(
                            dl_device=(1, 0), copy=True
                        )
                        self.assertIsNotNone(capsule)
                    finally:
                        array_module._NDARRAY_DLPACK_PARAMETERS = parameters
        with self.assertRaises(BufferError):
            array.__dlpack__(max_version=(1, 0), dl_device=(2, 0))
        with self.assertRaises(ValueError):
            array.__dlpack__(stream=0)

        producer = Producer(source)
        if "copy" in dlpack_parameters:
            imported = self.algebra.asarray(producer, copy=False)
            self.assertTrue(np.shares_memory(imported.coefficients, source))
            self.assertIs(producer.requests[-1]["copy"], False)
            owned = self.algebra.array(producer)
            self.assertFalse(np.shares_memory(owned.coefficients, source))
            self.assertIsNone(producer.requests[-1]["copy"])

        from_dlpack_parameters = array_module._FROM_DLPACK_PARAMETERS
        try:
            array_module._FROM_DLPACK_PARAMETERS = frozenset({"x"})
            with self.assertRaises(ValueError):
                self.algebra.asarray(producer, copy=False)
        finally:
            array_module._FROM_DLPACK_PARAMETERS = from_dlpack_parameters

    def test_advanced_index_assignment_updates_the_backing_array(self) -> None:
        """Apply advanced-index assignments directly to backing storage."""

        array = self.algebra.array(np.arange(12, dtype=np.float64).reshape(3, 4))
        array[[0, 2]] = 9
        self.assertCoefficientsEqual(array.coefficients[[0, 2]], [[9, 0, 0, 0]] * 2)
        mask = np.array([True, False, True])
        array[mask] = self.algebra([4, 3, 2, 1])
        self.assertCoefficientsEqual(array.coefficients[mask], [[4, 3, 2, 1]] * 2)

    def test_element_array_pickle_round_trip(self) -> None:
        """Restore an independent ElementArray with its cached parent."""

        original = self.algebra.array(
            np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        )
        restored = pickle.loads(pickle.dumps(original))
        self.assertIsInstance(restored, fcd.ElementArray)
        self.assertIs(restored.algebra, self.algebra)
        self.assertCoefficientsEqual(
            restored.coefficients, original.coefficients
        )
        self.assertFalse(
            np.shares_memory(restored.coefficients, original.coefficients)
        )

    def test_element_array_copy_and_deepcopy_are_independent(self) -> None:
        """Return independent storage from both array copy protocols."""

        original = self.algebra.array(
            np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        )
        for copier in (copy.copy, copy.deepcopy):
            restored = copier(original)
            with self.subTest(copier=copier.__name__):
                self.assertIsInstance(restored, fcd.ElementArray)
                self.assertIs(restored.algebra, self.algebra)
                self.assertCoefficientsEqual(
                    restored.coefficients, original.coefficients
                )
                self.assertFalse(
                    np.shares_memory(
                        restored.coefficients, original.coefficients
                    )
                )
                restored.coefficients[0, 0, 0] = -999.0
                self.assertNotEqual(original.coefficients[0, 0, 0], -999.0)


class ElementArrayArithmeticTests(FastCDTestCase):
    def setUp(self) -> None:
        """Create broadcastable octonion arrays for arithmetic tests."""

        self.algebra = fcd.Algebra(dimension=8)
        rng = np.random.default_rng(20260903)
        self.x_values = rng.normal(size=(2, 1, 8))
        self.y_values = rng.normal(size=(1, 3, 8))
        self.x = self.algebra.array(self.x_values)
        self.y = self.algebra.array(self.y_values)

    def test_broadcast_product_and_coordinate_arithmetic(self) -> None:
        """Verify broadcast products and coordinatewise array arithmetic."""

        result = self.x * self.y
        expected = np.empty((2, 3, 8))
        for i in range(2):
            for j in range(3):
                expected[i, j] = reference.multiply(
                    self.x_values[i, 0], self.y_values[0, j]
                )
        self.assertEqual(result.batch_shape, (2, 3))
        self.assertCoefficientsClose(result.coefficients, expected)
        self.assertCoefficientsClose(
            self.algebra.multiply(self.x_values, self.y_values), expected
        )

        sum_result = self.x + self.y
        self.assertCoefficientsEqual(
            sum_result.coefficients,
            np.broadcast_to(self.x_values, (2, 3, 8))
            + np.broadcast_to(self.y_values, (2, 3, 8)),
        )
        scalar_sum = self.x + 3.0
        expected_scalar_sum = self.x_values.copy()
        expected_scalar_sum[..., 0] += 3.0
        self.assertCoefficientsEqual(scalar_sum.coefficients, expected_scalar_sum)
        self.assertCoefficientsEqual((3.0 + self.x).coefficients, expected_scalar_sum)
        expected_reflected_subtract = -self.x_values
        expected_reflected_subtract[..., 0] += 3.0
        self.assertCoefficientsEqual(
            (3.0 - self.x).coefficients, expected_reflected_subtract
        )
        self.assertCoefficientsEqual((-self.x).coefficients, -self.x_values)
        self.assertCoefficientsEqual((self.x * 2).coefficients, 2 * self.x_values)
        self.assertCoefficientsEqual((2 * self.x).coefficients, 2 * self.x_values)
        self.assertCoefficientsEqual((self.x / 2).coefficients, self.x_values / 2)
        with self.assertRaises(ZeroDivisionError):
            _ = self.x / 0
        with self.assertRaises(TypeError):
            _ = self.x * self.y_values

    def test_batch_unary_scalar_and_composite_operations(self) -> None:
        """Compare batched unary, scalar, and composite operations."""

        full_x = np.broadcast_to(self.x_values, (2, 3, 8))
        full_y = np.broadcast_to(self.y_values, (2, 3, 8))
        expected_conjugate = np.stack(
            [
                reference.conjugate(value)
                for value in self.x_values.reshape(-1, 8)
            ]
        ).reshape(self.x_values.shape)
        self.assertCoefficientsEqual(
            self.x.conjugate().coefficients, expected_conjugate
        )
        self.assertCoefficientsEqual(
            self.x.imag_part().coefficients[..., 0], 0.0
        )
        self.assertCoefficientsClose(
            self.x.square().coefficients,
            reference.multiply_batch(self.x_values, self.x_values),
        )
        self.assertCoefficientsClose(
            self.x.norm_squared(), np.sum(self.x_values**2, axis=-1)
        )
        self.assertCoefficientsClose(
            self.x.norm(), np.linalg.norm(self.x_values, axis=-1)
        )
        self.assertCoefficientsClose(abs(self.x), self.x.norm())
        self.assertCoefficientsClose(
            self.x.inner(self.y), np.sum(full_x * full_y, axis=-1)
        )
        self.assertCoefficientsClose(
            self.x.normalized().norm(), np.ones((2, 1))
        )
        reciprocal_expected = np.apply_along_axis(
            reference.conjugate, -1, self.x_values
        ) / np.sum(self.x_values**2, axis=-1, keepdims=True)
        self.assertCoefficientsClose(
            self.x.reciprocal().coefficients, reciprocal_expected
        )

        for actual, expected_function in (
            (self.x.imag_cross(self.y), reference.imag_cross),
            (self.x.commutator(self.y), reference.commutator),
            (self.x.anticommutator(self.y), reference.anticommutator),
        ):
            expected = np.empty((2, 3, 8))
            for i in range(2):
                for j in range(3):
                    expected[i, j] = expected_function(
                        full_x[i, j], full_y[i, j]
                    )
            self.assertCoefficientsClose(actual.coefficients, expected)

        z_values = np.linspace(-1, 1, 8)
        associator = self.x.associator(self.y, self.algebra(z_values))
        expected_associator = np.empty((2, 3, 8))
        for i in range(2):
            for j in range(3):
                expected_associator[i, j] = reference.associator(
                    full_x[i, j], full_y[i, j], z_values
                )
        self.assertCoefficientsClose(associator.coefficients, expected_associator)

    def test_batch_power_and_equality(self) -> None:
        """Verify batch powers and element-valued equality semantics."""

        self.assertCoefficientsClose((self.x**0).coefficients[..., 0], 1.0)
        self.assertCoefficientsClose((self.x**0).coefficients[..., 1:], 0.0)
        self.assertCoefficientsClose(
            (self.x**2).coefficients, self.x.square().coefficients
        )
        self.assertCoefficientsClose(
            (self.x**-1).coefficients, self.x.reciprocal().coefficients
        )
        self.assertTrue(np.all(self.x == self.x.copy()))
        self.assertTrue(np.all(self.x != self.x + 1))
        scalar_array = self.algebra.array([[3.0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.all(scalar_array == 3))
        self.assertFalse(np.any(scalar_array != 3))
        scalar_batch = self.algebra.array([3.0, 0, 0, 0, 0, 0, 0, 0])
        self.assertIsInstance(scalar_batch == 3, np.ndarray)
        self.assertEqual((scalar_batch == 3).shape, ())
        with self.assertRaises(TypeError):
            _ = self.x**0.5
        with self.assertRaises(TypeError):
            pow(self.x, 2, 3)


class OutAndUfuncTests(FastCDTestCase):
    def setUp(self) -> None:
        """Create fixed quaternion arrays and their reference products."""

        self.algebra = fcd.Algebra(dimension=4)
        self.left_values = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, -1, 2, -3]])
        self.right_values = np.array([[2.0, -1.0, 0.5, 3.0], [4, 2, -1, 0.25]])
        self.left = self.algebra.array(self.left_values)
        self.right = self.algebra.array(self.right_values)
        self.expected = reference.multiply_batch(self.left_values, self.right_values)

    def test_out_element_array_and_ndarray(self) -> None:
        """Reuse compatible ElementArray and raw ndarray outputs."""

        element_out = self.algebra.array(np.empty_like(self.left_values))
        returned = self.algebra.multiply(self.left, self.right, out=element_out)
        self.assertIs(returned, element_out)
        self.assertCoefficientsClose(element_out.coefficients, self.expected)

        ndarray_out = np.empty_like(self.left_values)
        returned_wrapper = self.algebra.multiply(self.left, self.right, out=ndarray_out)
        self.assertIsInstance(returned_wrapper, fcd.ElementArray)
        self.assertTrue(np.shares_memory(returned_wrapper.coefficients, ndarray_out))
        self.assertCoefficientsClose(ndarray_out, self.expected)

        basis_out = self.algebra.array(np.empty_like(self.left_values))
        basis_result = self.algebra.left_basis_multiply(2, self.left, out=basis_out)
        self.assertIs(basis_result, basis_out)
        basis_values = np.zeros(4)
        basis_values[2] = 1
        self.assertCoefficientsClose(
            basis_out.coefficients,
            np.stack([reference.multiply(basis_values, x) for x in self.left_values]),
        )

    def test_out_validation_and_overlap_rejection(self) -> None:
        """Reject invalid outputs and unsafe input or output overlap."""

        invalid_outs = [
            np.empty((1, 4), dtype=np.float64),
            np.empty((2, 4), dtype=np.float32),
            np.empty((2, 8), dtype=np.float64)[:, ::2],
            object(),
        ]
        readonly = np.empty_like(self.left_values)
        readonly.flags.writeable = False
        invalid_outs.append(readonly)
        for out in invalid_outs:
            with self.subTest(out=type(out)), self.assertRaises(
                (TypeError, ValueError)
            ):
                self.algebra.multiply(self.left, self.right, out=out)
        with self.assertRaises(ValueError):
            self.algebra.multiply(self.left, self.right, out=self.left)
        other_out = fcd.Algebra(dimension=8).array(np.empty((2, 8)))
        with self.assertRaises(ValueError):
            self.algebra.multiply(self.left, self.right, out=other_out)

        storage = np.arange(48, dtype=np.float64).reshape(4, 3, 4)
        strided_left = self.algebra.asarray(storage[:3:2], copy=False)
        strided_right = self.algebra.asarray(storage[1:3], copy=False)
        self.assertTrue(
            np.shares_memory(
                strided_left.coefficients, strided_right.coefficients
            )
        )
        # Records pair as (0, 1) and (2, 2): disjoint or identical, so the
        # shared storage is accepted.
        self.assertCoefficientsClose(
            self.algebra.multiply(strided_left, strided_right).coefficients,
            reference.multiply_batch(storage[:3:2], storage[1:3]),
        )
        self.assertCoefficientsClose(
            (strided_left * strided_left).coefficients,
            reference.multiply_batch(storage[:3:2], storage[:3:2]),
        )
        shifted_right = self.algebra.asarray(
            storage.reshape(-1)[1:25].reshape(2, 3, 4), copy=False
        )
        with self.assertRaises(ValueError):
            self.algebra.multiply(
                self.algebra.asarray(storage[:2], copy=False), shifted_right
            )

    def test_supported_element_ufuncs(self) -> None:
        """Verify the intentional NumPy ufunc subset for scalar Elements."""

        left = self.algebra(self.left_values[0])
        right = self.algebra(self.right_values[0])
        self.assertCoefficientsEqual(np.add(left, 2), left + 2)
        self.assertCoefficientsEqual(np.subtract(2, left), 2 - left)
        self.assertCoefficientsEqual(np.negative(left), -left)
        self.assertCoefficientsClose(np.multiply(left, right), left * right)
        self.assertCoefficientsClose(np.square(left), left.square())
        self.assertCoefficientsEqual(np.conjugate(left), left.conjugate())
        self.assertEqual(np.absolute(left), abs(left))
        self.assertTrue(np.equal(left, self.algebra(left)))
        self.assertTrue(np.not_equal(left, right))
        self.assertCoefficientsEqual(np.positive(left), +left)
        self.assertCoefficientsEqual(np.true_divide(left, 2), left / 2)
        self.assertCoefficientsEqual(np.divide(left, 2), left / 2)
        self.assertCoefficientsEqual(np.power(left, 3), left**3)
        with self.assertRaises(TypeError):
            np.true_divide(2, left)
        with self.assertRaises(TypeError):
            np.power(2, left)
        element_out = self.algebra.array(np.empty(4, dtype=np.float64))
        self.assertIs(np.multiply(left, right, out=(element_out,)), element_out)
        self.assertCoefficientsClose(
            element_out.coefficients,
            reference.multiply(self.left_values[0], self.right_values[0]),
        )
        raw_out = np.empty(4, dtype=np.float64)
        self.assertIs(np.multiply(left, right, out=(raw_out,)), raw_out)
        self.assertCoefficientsClose(
            raw_out,
            reference.multiply(self.left_values[0], self.right_values[0]),
        )
        with self.assertRaises(TypeError):
            np.sin(left)
        with self.assertRaises(TypeError):
            np.add(left, right, out=np.empty(4))

    def test_supported_array_ufuncs_and_multiply_out(self) -> None:
        """Verify array ufunc dispatch and multiplication output reuse."""

        self.assertCoefficientsEqual(
            np.add(self.left, 2).coefficients, (self.left + 2).coefficients
        )
        self.assertCoefficientsEqual(
            np.subtract(2, self.left).coefficients,
            (2 - self.left).coefficients,
        )
        self.assertCoefficientsEqual(
            np.negative(self.left).coefficients, (-self.left).coefficients
        )
        self.assertCoefficientsClose(
            np.multiply(self.left, self.right).coefficients, self.expected
        )
        self.assertCoefficientsClose(
            np.square(self.left).coefficients, self.left.square().coefficients
        )
        self.assertCoefficientsEqual(
            np.conjugate(self.left).coefficients,
            self.left.conjugate().coefficients,
        )
        self.assertCoefficientsClose(np.absolute(self.left), self.left.norm())
        self.assertTrue(np.all(np.equal(self.left, self.left.copy())))
        self.assertTrue(np.all(np.not_equal(self.left, self.right)))
        self.assertCoefficientsEqual(
            np.positive(self.left).coefficients, self.left.coefficients
        )
        self.assertCoefficientsEqual(
            np.true_divide(self.left, 2).coefficients, (self.left / 2).coefficients
        )
        self.assertCoefficientsClose(
            np.power(self.left, 3).coefficients, (self.left**3).coefficients
        )
        with self.assertRaises(TypeError):
            np.power(2, self.left)

        out = self.algebra.array(np.empty_like(self.left_values))
        returned = np.multiply(self.left, self.right, out=(out,))
        self.assertIs(returned, out)
        self.assertCoefficientsClose(out.coefficients, self.expected)
        raw_out = np.empty_like(self.left_values)
        raw_returned = np.multiply(self.left, self.right, out=(raw_out,))
        self.assertIs(raw_returned, raw_out)
        self.assertCoefficientsClose(raw_out, self.expected)
        with self.assertRaises(TypeError):
            np.add(self.left, self.right, out=(out,))
        with self.assertRaises(TypeError):
            np.sin(self.left)
        with self.assertRaises(TypeError):
            np.add.reduce(self.left)


class BatchOperatorTests(FastCDTestCase):
    def test_basis_and_operator_broadcasting_with_out(self) -> None:
        """Broadcast basis and operator actions with optional reused output."""

        algebra = fcd.Algebra(dimension=8)
        rng = np.random.default_rng(8118)
        a = rng.normal(size=(2, 1, 8))
        b = rng.normal(size=(1, 3, 8))
        x = rng.normal(size=(2, 3, 8))

        left_expected = np.empty_like(x)
        right_expected = np.empty_like(x)
        for i in range(2):
            for j in range(3):
                left_expected[i, j] = reference.left_operator_commutator(
                    a[i, 0], b[0, j], x[i, j]
                )
                right_expected[i, j] = reference.right_operator_commutator(
                    a[i, 0], b[0, j], x[i, j]
                )

        left_out = algebra.array(np.empty_like(x))
        left_actual = algebra.left_operator_commutator(a, b, x, out=left_out)
        self.assertIs(left_actual, left_out)
        self.assertCoefficientsClose(
            left_actual.coefficients, left_expected, rtol=2.0e-11, atol=2.0e-11
        )
        right_actual = algebra.right_operator_commutator(a, b, x)
        self.assertCoefficientsClose(
            right_actual.coefficients, right_expected, rtol=2.0e-11, atol=2.0e-11
        )

        basis = np.zeros(8)
        basis[5] = 1.0
        left_basis = algebra.left_basis_multiply(5, x)
        right_basis = algebra.right_basis_multiply(x, 5)
        self.assertCoefficientsClose(
            left_basis.coefficients,
            np.stack(
                [reference.multiply(basis, value) for value in x.reshape(-1, 8)]
            ).reshape(x.shape),
        )
        self.assertCoefficientsClose(
            right_basis.coefficients,
            np.stack(
                [reference.multiply(value, basis) for value in x.reshape(-1, 8)]
            ).reshape(x.shape),
        )


class RemainingBranchTests(FastCDTestCase):
    def test_array_and_asarray_accept_fastcd_objects(self) -> None:
        """Copy arrays from arrays and view elements without copying."""

        algebra = fcd.Algebra(dimension=4)
        source = algebra.array(np.arange(8.0).reshape(2, 4))
        copied = algebra.array(source)
        self.assertFalse(np.shares_memory(copied.coefficients, source.coefficients))
        np.testing.assert_array_equal(copied.coefficients, source.coefficients)
        element = algebra([1.0, 2.0, 3.0, 4.0])
        view = algebra.asarray(element)
        self.assertTrue(np.shares_memory(view.coefficients, np.asarray(element)))
        self.assertFalse(view.coefficients.flags.writeable)
        with self.assertRaises(ValueError):
            algebra.array(fcd.Algebra(dimension=8).array(np.ones((1, 8))))

    def test_imag_positive_and_dtype_conversion(self) -> None:
        """Exercise the imaginary part, unary plus, and dtype conversion."""

        algebra = fcd.Algebra(dimension=4)
        batch = algebra.array(np.arange(8.0).reshape(2, 4))
        imag = batch.imag
        np.testing.assert_array_equal(imag.coefficients[..., 0], 0.0)
        np.testing.assert_array_equal(
            imag.coefficients[..., 1:], batch.coefficients[..., 1:]
        )
        positive = +batch
        self.assertIsNot(positive, batch)
        np.testing.assert_array_equal(positive.coefficients, batch.coefficients)
        converted = np.asarray(batch, dtype=np.float32)
        self.assertEqual(converted.dtype, np.float32)
        with self.assertRaises(ValueError):
            batch.__array__(dtype=np.float32, copy=False)

    def test_setitem_rejects_foreign_parent(self) -> None:
        """Refuse to assign records from another algebra."""

        algebra = fcd.Algebra(dimension=4)
        batch = algebra.array(np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            batch[0] = fcd.Algebra(dimension=8).array(np.ones((8,)))

    def test_validation_helpers_reject_bad_inputs(self) -> None:
        """Cover index, dimension, and DLPack pair validation errors."""

        from pyfastcd import _validation

        with self.assertRaises(TypeError):
            _validation._basis_index("x", 4)
        with self.assertRaises(TypeError):
            _validation._basis_index(True, 4)
        with self.assertRaises(OverflowError):
            fcd.Algebra(dimension=2 ** (sys.maxsize.bit_length() - 1))
        for bad in ((1,), ("1", 0), (-1, 0), [1, 0]):
            with self.subTest(pair=bad), self.assertRaises(TypeError):
                array_module._dlpack_pair(bad, name="max_version")

    def test_dlpack_import_rejects_non_host_devices(self) -> None:
        """Refuse producers whose data does not live in host memory."""

        class Producer:
            def __dlpack__(self, **kwargs: Any) -> Any:
                raise AssertionError("device check must come first")

            def __dlpack_device__(self) -> tuple[int, int]:
                return (2, 0)

        with self.assertRaises(ValueError):
            fcd.Algebra(dimension=4).array(Producer())

    def test_ufunc_dispatch_declines_extra_options(self) -> None:
        """Return NotImplemented for ufunc keywords outside the supported set."""

        algebra = fcd.Algebra(dimension=4)
        for value in (algebra([1.0, 2.0, 3.0, 4.0]), algebra.array(np.ones((2, 4)))):
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(TypeError):
                    np.add(value, 1, where=True)
                with self.assertRaises(TypeError):
                    np.negative(value, out=np.empty((2, 4)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
