#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "fastcd/fastcd.h"

/* Messages shared verbatim with pyfastcd._validation so that the C and Python
 * layers report identical conditions identically. */
#define PYFASTCD_DIFFERENT_ALGEBRA_MESSAGE "operands belong to different algebras"
#define PYFASTCD_COMPLEX_SCALAR_MESSAGE "complex scalars are not supported"
#define PYFASTCD_MODULAR_POWER_MESSAGE \
    "modular exponentiation is not defined for Cayley--Dickson elements"
#define PYFASTCD_INTEGER_EXPONENT_MESSAGE \
    "Cayley--Dickson powers require an integer exponent"

#define PYFASTCD_STACK_WORKSPACE_LENGTH ((size_t)1024)
#define PYFASTCD_GIL_RELEASE_COUNT ((size_t)32)

_Static_assert(sizeof(double) == 8 && FLT_RADIX == 2
        && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024
        && DBL_MIN_EXP == -1021,
        "pyfastcd requires an eight-byte IEC 60559 binary64-compatible double");

typedef struct {
    PyObject_VAR_HEAD
    PyObject* algebra;
    Py_ssize_t buffer_shape;
    Py_ssize_t buffer_stride;
    double coefficients[];
} PyFastCDElement;

typedef enum {
    PYFASTCD_UNARY_CONJUGATE = 1,
    PYFASTCD_UNARY_IMAG_PART,
    PYFASTCD_UNARY_NORMALIZE,
    PYFASTCD_UNARY_RECIPROCAL,
    PYFASTCD_UNARY_SQUARE
} PyFastCDUnaryOperation;

typedef enum {
    PYFASTCD_BINARY_MULTIPLY = 1,
    PYFASTCD_BINARY_IMAG_CROSS,
    PYFASTCD_BINARY_COMMUTATOR,
    PYFASTCD_BINARY_ANTICOMMUTATOR
} PyFastCDBinaryOperation;

typedef enum {
    PYFASTCD_SCALAR_INNER = 1,
    PYFASTCD_SCALAR_NORM_SQUARED,
    PYFASTCD_SCALAR_NORM
} PyFastCDScalarOperation;

typedef struct {
    double stack[PYFASTCD_STACK_WORKSPACE_LENGTH];
    FastCDWorkspaceF64 value;
    bool heap_allocated;
} PyFastCDWorkspace;

typedef struct {
    uintptr_t first;
    uintptr_t last;
    bool empty;
} PyFastCDBufferSpan;

typedef struct {
    uintptr_t first;
    uintptr_t last;
} PyFastCDBufferInterval;

static PyTypeObject PyFastCDElementType;

/* ---------- Error and allocation helpers ---------- */

// Translate a fastCD status into the corresponding Python exception
static int pyfastcd_set_status_error(FastCDStatus status) {
    PyObject* exception = PyExc_RuntimeError;

    switch (status) {
        case FASTCD_SUCCESS:
            return 0;
        case FASTCD_INVALID_DIMENSION:
        case FASTCD_INVALID_LEVEL:
        case FASTCD_INVALID_STRIDE:
        case FASTCD_OVERLAPPING_BUFFERS:
        case FASTCD_UNSUPPORTED_ALGORITHM:
            exception = PyExc_ValueError;
            break;
        case FASTCD_INVALID_BASIS_INDEX:
            exception = PyExc_IndexError;
            break;
        case FASTCD_ZERO_NORM:
            exception = PyExc_ZeroDivisionError;
            break;
        case FASTCD_SIZE_OVERFLOW:
            exception = PyExc_OverflowError;
            break;
        case FASTCD_INSUFFICIENT_WORKSPACE:
        case FASTCD_NULL_POINTER:
            exception = PyExc_RuntimeError;
            break;
        default:
            exception = PyExc_RuntimeError;
            break;
    }
    PyErr_SetString(exception, fastcd_status_string(status));
    return -1;
}

// Validate and convert a Python-sized Cayley--Dickson dimension
static int pyfastcd_dimension_from_ssize(Py_ssize_t value, size_t* dimension) {
    if (dimension == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "internal null dimension pointer");
        return -1;
    }
    if (value <= 0 || (uintmax_t)value > (uintmax_t)SIZE_MAX) {
        PyErr_SetString(PyExc_ValueError,
                "dimension must be a positive power of two");
        return -1;
    }
    *dimension = (size_t)value;
    if (!fastcd_valid_dimension(*dimension)) {
        PyErr_SetString(PyExc_ValueError,
                "dimension must be a positive power of two");
        return -1;
    }
    return 0;
}

// Allocate an element and initialize its parent and buffer metadata
static PyFastCDElement* pyfastcd_element_allocate(PyObject* algebra,
        size_t dimension) {
    PyFastCDElement* result;

    if (algebra == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "element algebra is null");
        return NULL;
    }
    if (dimension > (size_t)PY_SSIZE_T_MAX
            || dimension > ((size_t)PY_SSIZE_T_MAX
                - sizeof(PyFastCDElement)) / sizeof(double)) {
        PyErr_SetString(PyExc_OverflowError,
                "element dimension exceeds Python container limits");
        return NULL;
    }
    result = (PyFastCDElement*)PyFastCDElementType.tp_alloc(
            &PyFastCDElementType, (Py_ssize_t)dimension);
    if (result == NULL) return NULL;
    result->algebra = Py_NewRef(algebra);
    result->buffer_shape = (Py_ssize_t)dimension;
    result->buffer_stride = (Py_ssize_t)sizeof(double);
    return result;
}

// Copy an element while retaining its algebra parent
static PyFastCDElement* pyfastcd_element_copy(const PyFastCDElement* source) {
    size_t dimension = (size_t)Py_SIZE(source);
    PyFastCDElement* result = pyfastcd_element_allocate(source->algebra,
            dimension);

    if (result == NULL) return NULL;
    memcpy(result->coefficients, source->coefficients,
            dimension * sizeof(double));
    return result;
}

// Allocate stack or heap storage for a requested workspace length
static int pyfastcd_workspace_allocate(PyFastCDWorkspace* workspace,
        size_t length, bool force_heap) {
    if (workspace == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "internal null workspace");
        return -1;
    }
    workspace->heap_allocated = false;
    workspace->value.length = length;
    if (length == 0) {
        workspace->value.data = NULL;
        return 0;
    }
    if (!force_heap && length <= PYFASTCD_STACK_WORKSPACE_LENGTH) {
        workspace->value.data = workspace->stack;
        return 0;
    }
    if (length > SIZE_MAX / sizeof(double)) {
        PyErr_SetString(PyExc_OverflowError, "workspace size overflow");
        return -1;
    }
    workspace->value.data = PyMem_Malloc(length * sizeof(double));
    if (workspace->value.data == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    workspace->heap_allocated = true;
    return 0;
}

// Release heap storage owned by a temporary workspace
static void pyfastcd_workspace_release(PyFastCDWorkspace* workspace) {
    if (workspace != NULL && workspace->heap_allocated) {
        PyMem_Free(workspace->value.data);
        workspace->value.data = NULL;
        workspace->value.length = 0;
        workspace->heap_allocated = false;
    }
}

// Query and allocate workspace for a dimension-dependent operation
static int pyfastcd_workspace_for_query(PyFastCDWorkspace* workspace,
        size_t dimension,
        FastCDStatus (*query)(size_t, size_t*)) {
    size_t length = 0;
    FastCDStatus status = query(dimension, &length);

    if (status != FASTCD_SUCCESS) return pyfastcd_set_status_error(status);
    return pyfastcd_workspace_allocate(workspace, length, false);
}

/* ---------- Multiplication algorithm selection ---------- */

static const FastCDAlgorithm pyfastcd_algorithms[] = {
    FASTCD_ALGORITHM_AUTO,
    FASTCD_ALGORITHM_DIRECT,
    FASTCD_ALGORITHM_CARIOW_CARIOWA,
    FASTCD_ALGORITHM_QUASILINEAR
};

#define PYFASTCD_ALGORITHM_COUNT \
    (sizeof(pyfastcd_algorithms) / sizeof(pyfastcd_algorithms[0]))

// Validate an integer algorithm code received from the Python layer
static int pyfastcd_algorithm_from_code(int code, FastCDAlgorithm* algorithm) {
    for (size_t i = 0; i < PYFASTCD_ALGORITHM_COUNT; i++) {
        if (code == (int)pyfastcd_algorithms[i]) {
            *algorithm = pyfastcd_algorithms[i];
            return 0;
        }
    }
    PyErr_SetString(PyExc_ValueError, "unknown multiplication algorithm code");
    return -1;
}

// Map a public algorithm name onto the matching FastCDAlgorithm value
static int pyfastcd_algorithm_from_name(PyObject* name,
        FastCDAlgorithm* algorithm) {
    const char* text;
    Py_ssize_t length;

    if (!PyUnicode_Check(name)) {
        PyErr_Format(PyExc_TypeError,
                "algorithm must be a str, not %.100s", Py_TYPE(name)->tp_name);
        return -1;
    }
    text = PyUnicode_AsUTF8AndSize(name, &length);
    if (text == NULL) return -1;
    for (size_t i = 0; i < PYFASTCD_ALGORITHM_COUNT; i++) {
        const char* candidate = fastcd_algorithm_string(pyfastcd_algorithms[i]);

        if (strlen(candidate) == (size_t)length
                && memcmp(text, candidate, (size_t)length) == 0) {
            *algorithm = pyfastcd_algorithms[i];
            return 0;
        }
    }
    PyErr_Format(PyExc_ValueError,
            "unknown multiplication algorithm %R; expected one of "
            "'%s', '%s', '%s', or '%s'", name,
            fastcd_algorithm_string(FASTCD_ALGORITHM_AUTO),
            fastcd_algorithm_string(FASTCD_ALGORITHM_DIRECT),
            fastcd_algorithm_string(FASTCD_ALGORITHM_CARIOW_CARIOWA),
            fastcd_algorithm_string(FASTCD_ALGORITHM_QUASILINEAR));
    return -1;
}

// Require an algorithm to multiply elements of a validated dimension
static int pyfastcd_require_algorithm_support(FastCDAlgorithm algorithm,
        size_t dimension) {
    if (!fastcd_algorithm_supports_dimension(algorithm, dimension)) {
        PyErr_Format(PyExc_ValueError,
                "multiplication algorithm '%s' does not support dimension %zu",
                fastcd_algorithm_string(algorithm), dimension);
        return -1;
    }
    return 0;
}

// Allocate the workspace required by a selected multiplication algorithm
static int pyfastcd_algorithm_workspace(PyFastCDWorkspace* workspace,
        size_t dimension, FastCDAlgorithm algorithm) {
    size_t length = 0;
    FastCDStatus status = fastcd_mul_algorithm_workspace_size(dimension,
            algorithm, &length);

    if (status != FASTCD_SUCCESS) return pyfastcd_set_status_error(status);
    return pyfastcd_workspace_allocate(workspace, length, false);
}

// Require two elements to have the same coefficient dimension
static int pyfastcd_check_same_dimension(const PyFastCDElement* left,
        const PyFastCDElement* right) {
    if (Py_SIZE(left) != Py_SIZE(right)) {
        PyErr_SetString(PyExc_ValueError,
                PYFASTCD_DIFFERENT_ALGEBRA_MESSAGE);
        return -1;
    }
    return 0;
}

// Report whether a PEP 3118 format denotes a complex scalar
static bool pyfastcd_complex_buffer_format(const char* format) {
    if (format == NULL) return false;
    if (*format == '@' || *format == '=' || *format == '<'
            || *format == '>' || *format == '!') {
        format++;
    }
    return *format == 'Z';
}

// Report whether a PEP 3118 format denotes one real numeric scalar
static bool pyfastcd_real_numeric_buffer_format(const char* format) {
    const char* real_formats = "?bBhHiIlLqQnNefdg";

    if (format == NULL) return false;
    if (*format == '@' || *format == '=' || *format == '<'
            || *format == '>' || *format == '!') {
        format++;
    }
    return format[0] != '\0' && format[1] == '\0'
        && strchr(real_formats, format[0]) != NULL;
}

// Report whether a PEP 3118 format denotes a native double
static bool pyfastcd_native_double_format(const char* format) {
    const uint16_t one = 1;
    bool little_endian = *(const unsigned char*)&one == 1;

    if (format == NULL) return false;
    if (strcmp(format, "d") == 0 || strcmp(format, "@d") == 0
            || strcmp(format, "=d") == 0) {
        return true;
    }
    return (little_endian && strcmp(format, "<d") == 0)
        || (!little_endian && (strcmp(format, ">d") == 0
            || strcmp(format, "!d") == 0));
}

// Recognize and convert one unambiguous real scalar object
static int pyfastcd_as_real_scalar(PyObject* value, double* scalar) {
    PyObject* converted;
    Py_buffer view = {0};

    if (PyObject_TypeCheck(value, &PyFastCDElementType)) return 0;
    if (PyComplex_Check(value)) {
        PyErr_SetString(PyExc_TypeError, PYFASTCD_COMPLEX_SCALAR_MESSAGE);
        return 0;
    }
    if (!PyFloat_Check(value) && !PyLong_Check(value)
            && PyObject_GetBuffer(value, &view,
                PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) == 0) {
        bool is_complex = pyfastcd_complex_buffer_format(view.format);
        bool is_real_numeric = pyfastcd_real_numeric_buffer_format(
                view.format);

        if (view.ndim != 0) {
            PyBuffer_Release(&view);
            return 0;
        }
        PyBuffer_Release(&view);
        if (is_complex) {
            PyErr_SetString(PyExc_TypeError, PYFASTCD_COMPLEX_SCALAR_MESSAGE);
            return 0;
        }
        if (!is_real_numeric) {
            PyErr_SetString(PyExc_TypeError,
                    "scalar must have a real numeric dtype");
            return 0;
        }
    } else if (PyErr_Occurred()) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)
                || PyErr_ExceptionMatches(PyExc_BufferError)) {
            PyErr_Clear();
        } else if (PyErr_ExceptionMatches(PyExc_ValueError)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError,
                    "scalar must have a real numeric dtype");
            return 0;
        } else {
            return 0;
        }
    }
    if (!PyNumber_Check(value)) return 0;
    converted = PyNumber_Float(value);
    if (converted == NULL) return 0;
    *scalar = PyFloat_AS_DOUBLE(converted);
    Py_DECREF(converted);
    return 1;
}

/* ---------- Element construction ---------- */

// Copy compatible one-dimensional buffer coefficients into an element
static int pyfastcd_copy_buffer_coefficients(PyFastCDElement* element,
        PyObject* source) {
    Py_buffer view;
    size_t dimension = (size_t)Py_SIZE(element);

    if (PyObject_GetBuffer(source, &view,
                PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)
                || PyErr_ExceptionMatches(PyExc_BufferError)) {
            PyErr_Clear();
            return 0;
        }
        return -1;
    }
    if (view.ndim != 1 || view.shape == NULL
            || view.shape[0] != (Py_ssize_t)dimension) {
        PyBuffer_Release(&view);
        PyErr_Format(PyExc_ValueError,
                "coefficient input must have shape (%zu,)", dimension);
        return -1;
    }
    if (pyfastcd_complex_buffer_format(view.format)) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError,
                "complex coefficients are not supported");
        return -1;
    }
    if (view.itemsize != (Py_ssize_t)sizeof(double)
            || !pyfastcd_native_double_format(view.format)) {
        PyBuffer_Release(&view);
        return 0;
    }
    if (view.strides == NULL) {
        memcpy(element->coefficients, view.buf,
                dimension * sizeof(double));
    } else {
        const char* pointer = (const char*)view.buf;
        for (size_t i = 0; i < dimension; i++) {
            memcpy(&element->coefficients[i], pointer,
                    sizeof(double));
            pointer += view.strides[0];
        }
    }
    PyBuffer_Release(&view);
    return 1;
}

// Convert a stable snapshot of a coefficient sequence into an element
static int pyfastcd_copy_sequence_coefficients(PyFastCDElement* element,
        PyObject* source) {
    PyObject* sequence;
    size_t dimension = (size_t)Py_SIZE(element);

    if (PyUnicode_Check(source) || PyBytes_Check(source)
            || PyByteArray_Check(source)) {
        PyErr_SetString(PyExc_TypeError,
                "coefficients must be real numeric values");
        return -1;
    }
    /* The input must define the coordinate order.  Sequences and iterators
     * do; a set, mapping, or other unordered iterable would silently assign
     * coordinates in hash order. */
    if (!PySequence_Check(source) && !PyIter_Check(source)) {
        PyErr_SetString(PyExc_TypeError,
                "coefficients must be a sequence, iterator, or buffer");
        return -1;
    }
    /* Conversions may call Python and mutate source.  A tuple owns stable
     * references to every coefficient for the whole conversion loop. */
    sequence = PySequence_Tuple(source);
    if (sequence == NULL) return -1;
    if (PyTuple_GET_SIZE(sequence) != (Py_ssize_t)dimension) {
        Py_DECREF(sequence);
        PyErr_Format(PyExc_ValueError,
                "coefficient input must contain exactly %zu values",
                dimension);
        return -1;
    }
    for (size_t i = 0; i < dimension; i++) {
        PyObject* item = PyTuple_GET_ITEM(sequence, (Py_ssize_t)i);
        double value;
        int is_scalar = pyfastcd_as_real_scalar(item, &value);

        if (!is_scalar) {
            Py_DECREF(sequence);
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_TypeError,
                        "coefficients must be real numeric values");
            }
            return -1;
        }
        element->coefficients[i] = value;
    }
    Py_DECREF(sequence);
    return 0;
}

// Construct an element from another element, a buffer, or a sequence
static PyObject* pyfastcd_element_from_object(PyObject* Py_UNUSED(module),
        PyObject* args) {
    PyObject* algebra;
    PyObject* source;
    Py_ssize_t dimension_value;
    size_t dimension;
    PyFastCDElement* result;
    int buffer_result;

    if (!PyArg_ParseTuple(args, "OnO:_element",
                &algebra, &dimension_value, &source)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    if (PyObject_TypeCheck(source, &PyFastCDElementType)) {
        PyFastCDElement* other = (PyFastCDElement*)source;
        if ((size_t)Py_SIZE(other) != dimension) {
            PyErr_SetString(PyExc_ValueError,
                    PYFASTCD_DIFFERENT_ALGEBRA_MESSAGE);
            return NULL;
        }
        result = pyfastcd_element_allocate(algebra, dimension);
        if (result == NULL) return NULL;
        memcpy(result->coefficients, other->coefficients,
                dimension * sizeof(double));
        return (PyObject*)result;
    }

    {
        double scalar;
        int is_scalar = pyfastcd_as_real_scalar(source, &scalar);
        FastCDStatus status;

        if (!is_scalar && PyErr_Occurred()) return NULL;
        if (is_scalar) {
            result = pyfastcd_element_allocate(algebra, dimension);
            if (result == NULL) return NULL;
            status = fastcd_scalar_f64(result->coefficients, dimension,
                    scalar);
            if (status != FASTCD_SUCCESS) {
                Py_DECREF(result);
                pyfastcd_set_status_error(status);
                return NULL;
            }
            return (PyObject*)result;
        }
    }

    result = pyfastcd_element_allocate(algebra, dimension);
    if (result == NULL) return NULL;
    buffer_result = pyfastcd_copy_buffer_coefficients(result, source);
    if (buffer_result < 0) {
        Py_DECREF(result);
        return NULL;
    }
    if (buffer_result == 0
            && pyfastcd_copy_sequence_coefficients(result, source) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    return (PyObject*)result;
}

// Construct an element whose only nonzero coefficient is a real scalar
static PyObject* pyfastcd_scalar_element(PyObject* Py_UNUSED(module),
        PyObject* args) {
    PyObject* algebra;
    Py_ssize_t dimension_value;
    double scalar;
    size_t dimension;
    PyFastCDElement* result;
    FastCDStatus status;

    if (!PyArg_ParseTuple(args, "Ond:_scalar",
                &algebra, &dimension_value, &scalar)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    result = pyfastcd_element_allocate(algebra, dimension);
    if (result == NULL) return NULL;
    status = fastcd_scalar_f64(result->coefficients, dimension, scalar);
    if (status != FASTCD_SUCCESS) {
        Py_DECREF(result);
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return (PyObject*)result;
}

// Construct one standard-basis element for a validated algebra dimension
static PyObject* pyfastcd_basis_element(PyObject* Py_UNUSED(module),
        PyObject* args) {
    PyObject* algebra;
    Py_ssize_t dimension_value;
    Py_ssize_t index_value;
    size_t dimension;
    PyFastCDElement* result;
    FastCDStatus status;

    if (!PyArg_ParseTuple(args, "Onn:_basis",
                &algebra, &dimension_value, &index_value)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    if (index_value < 0 || (size_t)index_value >= dimension) {
        PyErr_SetString(PyExc_IndexError, "basis index is out of range");
        return NULL;
    }
    result = pyfastcd_element_allocate(algebra, dimension);
    if (result == NULL) return NULL;
    status = fastcd_basis_f64(result->coefficients, dimension,
            (size_t)index_value);
    if (status != FASTCD_SUCCESS) {
        Py_DECREF(result);
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return (PyObject*)result;
}

/* ---------- Element protocol methods ---------- */

// Visit the algebra reference held by an element for cyclic garbage collection
static int pyfastcd_element_traverse(PyFastCDElement* self, visitproc visit,
        void* arg) {
    Py_VISIT(self->algebra);
    return 0;
}

// Clear the algebra reference held by an element
static int pyfastcd_element_clear(PyFastCDElement* self) {
    Py_CLEAR(self->algebra);
    return 0;
}

// Untrack, clear, and free an element object
static void pyfastcd_element_dealloc(PyFastCDElement* self) {
    PyObject_GC_UnTrack(self);
    pyfastcd_element_clear(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// Reject direct Element construction outside an Algebra parent
static PyObject* pyfastcd_element_new_forbidden(PyTypeObject* type,
        PyObject* Py_UNUSED(args), PyObject* Py_UNUSED(keywords)) {
    PyErr_Format(PyExc_TypeError,
            "%s objects are created by calling an Algebra",
            type->tp_name);
    return NULL;
}

// Return the number of real coefficients in an element
static Py_ssize_t pyfastcd_element_length(PyObject* self) {
    return Py_SIZE((PyFastCDElement*)self);
}

// Return one coefficient using Python sequence index semantics
static PyObject* pyfastcd_element_item(PyObject* self, Py_ssize_t index) {
    PyFastCDElement* element = (PyFastCDElement*)self;
    Py_ssize_t dimension = Py_SIZE(element);

    if (index < 0) index += dimension;
    if (index < 0 || index >= dimension) {
        PyErr_SetString(PyExc_IndexError, "element index is out of range");
        return NULL;
    }
    return PyFloat_FromDouble(element->coefficients[index]);
}

// Return one coefficient through the sequence protocol slot
static PyObject* pyfastcd_element_sequence_item(PyObject* self,
        Py_ssize_t index) {
    /* PySequence_GetItem has already added the length to a negative index,
     * so a negative value here is out of range rather than a wraparound. */
    if (index < 0) {
        PyErr_SetString(PyExc_IndexError, "element index is out of range");
        return NULL;
    }
    return pyfastcd_element_item(self, index);
}

// Iterate over the coefficients in coordinate order
static PyObject* pyfastcd_element_iter(PyObject* self) {
    return PySeqIter_New(self);
}

// Return an indexed coefficient or a tuple for a coefficient slice
static PyObject* pyfastcd_element_subscript(PyObject* self, PyObject* key) {
    PyFastCDElement* element = (PyFastCDElement*)self;
    Py_ssize_t dimension = Py_SIZE(element);

    if (PyIndex_Check(key)) {
        Py_ssize_t index = PyNumber_AsSsize_t(key, PyExc_IndexError);
        if (index == -1 && PyErr_Occurred()) return NULL;
        return pyfastcd_element_item(self, index);
    }
    if (PySlice_Check(key)) {
        Py_ssize_t start;
        Py_ssize_t stop;
        Py_ssize_t step;
        Py_ssize_t length;
        PyObject* tuple;

        if (PySlice_GetIndicesEx(key, dimension, &start, &stop, &step,
                    &length) < 0) {
            return NULL;
        }
        tuple = PyTuple_New(length);
        if (tuple == NULL) return NULL;
        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject* value = PyFloat_FromDouble(
                    element->coefficients[start + i * step]);
            if (value == NULL) {
                Py_DECREF(tuple);
                return NULL;
            }
            PyTuple_SET_ITEM(tuple, i, value);
        }
        return tuple;
    }
    PyErr_SetString(PyExc_TypeError,
            "element indices must be integers or slices");
    return NULL;
}

// Export immutable native-double coefficient storage through the buffer protocol
static int pyfastcd_element_getbuffer(PyObject* exporter, Py_buffer* view,
        int flags) {
    PyFastCDElement* self = (PyFastCDElement*)exporter;

    if (view == NULL) {
        PyErr_SetString(PyExc_BufferError, "buffer view is null");
        return -1;
    }
    if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) {
        PyErr_SetString(PyExc_BufferError,
                "Cayley--Dickson elements are immutable");
        return -1;
    }
    view->obj = Py_NewRef(exporter);
    view->buf = self->coefficients;
    view->len = Py_SIZE(self) * (Py_ssize_t)sizeof(double);
    view->readonly = 1;
    view->itemsize = (Py_ssize_t)sizeof(double);
    view->format = (flags & PyBUF_FORMAT) ? (char*)"d" : NULL;
    view->ndim = 1;
    view->shape = (flags & PyBUF_ND) ? &self->buffer_shape : NULL;
    view->strides = ((flags & PyBUF_STRIDES) == PyBUF_STRIDES)
        ? &self->buffer_stride : NULL;
    view->suboffsets = NULL;
    view->internal = NULL;
    return 0;
}

// Format an element with its coefficients and algebra parent
static PyObject* pyfastcd_element_repr(PyObject* self) {
    PyFastCDElement* element = (PyFastCDElement*)self;
    PyObject* list = PyList_New(Py_SIZE(element));
    PyObject* list_repr;
    PyObject* result;

    if (list == NULL) return NULL;
    for (Py_ssize_t i = 0; i < Py_SIZE(element); i++) {
        PyObject* value = PyFloat_FromDouble(element->coefficients[i]);
        if (value == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, value);
    }
    list_repr = PyObject_Repr(list);
    Py_DECREF(list);
    if (list_repr == NULL) return NULL;
    result = PyUnicode_FromFormat("Element(%U, algebra=%R)", list_repr,
            element->algebra);
    Py_DECREF(list_repr);
    return result;
}

// Compare an element exactly with another element or an embedded real scalar
static PyObject* pyfastcd_element_richcompare(PyObject* left_object,
        PyObject* right_object, int operation) {
    PyFastCDElement* left;
    int equal = 0;

    if (operation != Py_EQ && operation != Py_NE) Py_RETURN_NOTIMPLEMENTED;
    if (!PyObject_TypeCheck(left_object, &PyFastCDElementType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    left = (PyFastCDElement*)left_object;
    if (PyObject_TypeCheck(right_object, &PyFastCDElementType)) {
        PyFastCDElement* right = (PyFastCDElement*)right_object;
        if (Py_SIZE(left) == Py_SIZE(right)) {
            equal = 1;
            for (Py_ssize_t i = 0; i < Py_SIZE(left); i++) {
                if (!(left->coefficients[i] == right->coefficients[i])) {
                    equal = 0;
                    break;
                }
            }
        }
    } else {
        double scalar;
        int scalar_result = pyfastcd_as_real_scalar(right_object, &scalar);
        if (scalar_result) {
            equal = left->coefficients[0] == scalar;
            for (Py_ssize_t i = 1; equal && i < Py_SIZE(left); i++) {
                if (!(left->coefficients[i] == 0.0)) equal = 0;
            }
        } else if (PyErr_Occurred()) {
            /* An operand of an unsupported type compares unequal instead of
             * raising, as for ElementArray; other conversion failures such
             * as a raising __float__ still propagate. */
            if (!PyErr_ExceptionMatches(PyExc_TypeError)) return NULL;
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        } else {
            Py_RETURN_NOTIMPLEMENTED;
        }
    }
    if (operation == Py_NE) equal = !equal;
    return PyBool_FromLong(equal);
}

// Hash an element consistently with equality against elements and real scalars
static Py_hash_t pyfastcd_element_hash(PyObject* self_object) {
    PyFastCDElement* self = (PyFastCDElement*)self_object;
    Py_ssize_t dimension = Py_SIZE(self);
    bool embedded_real = true;
    PyObject* key;
    Py_hash_t result;

    for (Py_ssize_t i = 0; i < dimension; i++) {
        if (isnan(self->coefficients[i])) {
            /* NaN never compares equal, so any fixed value is consistent
             * with equality; a fresh float would hash by identity and
             * change between calls. */
            return (Py_hash_t)dimension;
        }
        if (i > 0 && self->coefficients[i] != 0.0) embedded_real = false;
    }
    if (embedded_real) {
        /* x == r holds for the real scalar r = x[0], so hash like that
         * float; Python hashes equal ints, floats, and NumPy scalars alike. */
        key = PyFloat_FromDouble(self->coefficients[0]);
    } else {
        key = PyTuple_New(dimension + 1);
        if (key == NULL) return -1;
        PyTuple_SET_ITEM(key, 0, PyLong_FromSsize_t(dimension));
        for (Py_ssize_t i = 0; i < dimension; i++) {
            PyTuple_SET_ITEM(key, i + 1,
                    PyFloat_FromDouble(self->coefficients[i]));
        }
        for (Py_ssize_t i = 0; i <= dimension; i++) {
            if (PyTuple_GET_ITEM(key, i) == NULL) {
                Py_DECREF(key);
                return -1;
            }
        }
    }
    if (key == NULL) return -1;
    result = PyObject_Hash(key);
    Py_DECREF(key);
    return result;
}

// Report whether any coefficient of an element is nonzero
static int pyfastcd_element_bool(PyObject* self) {
    PyFastCDElement* element = (PyFastCDElement*)self;

    for (Py_ssize_t i = 0; i < Py_SIZE(element); i++) {
        if (element->coefficients[i] != 0.0) return 1;
    }
    return 0;
}

/* ---------- Scalar arithmetic slots ---------- */

// Add compatible elements or add an embedded real scalar
static PyObject* pyfastcd_element_add(PyObject* left_object,
        PyObject* right_object) {
    PyFastCDElement* element;
    PyFastCDElement* result;
    double scalar;
    int scalar_result;
    bool element_on_left;

    if (PyObject_TypeCheck(left_object, &PyFastCDElementType)) {
        element = (PyFastCDElement*)left_object;
        element_on_left = true;
    } else if (PyObject_TypeCheck(right_object, &PyFastCDElementType)) {
        element = (PyFastCDElement*)right_object;
        element_on_left = false;
    } else {
        Py_RETURN_NOTIMPLEMENTED;
    }
    if (PyObject_TypeCheck(left_object, &PyFastCDElementType)
            && PyObject_TypeCheck(right_object, &PyFastCDElementType)) {
        PyFastCDElement* left = (PyFastCDElement*)left_object;
        PyFastCDElement* right = (PyFastCDElement*)right_object;
        FastCDStatus status;

        if (pyfastcd_check_same_dimension(left, right) < 0) return NULL;
        result = pyfastcd_element_allocate(left->algebra,
                (size_t)Py_SIZE(left));
        if (result == NULL) return NULL;
        status = fastcd_add_f64(result->coefficients, left->coefficients,
                right->coefficients, (size_t)Py_SIZE(left));
        if (status != FASTCD_SUCCESS) {
            Py_DECREF(result);
            pyfastcd_set_status_error(status);
            return NULL;
        }
        return (PyObject*)result;
    }
    scalar_result = pyfastcd_as_real_scalar(element_on_left
            ? right_object : left_object, &scalar);
    if (!scalar_result) {
        if (PyErr_Occurred()) return NULL;
        Py_RETURN_NOTIMPLEMENTED;
    }
    result = pyfastcd_element_copy(element);
    if (result == NULL) return NULL;
    result->coefficients[0] += scalar;
    return (PyObject*)result;
}

// Subtract compatible elements or an embedded real scalar in either order
static PyObject* pyfastcd_element_subtract(PyObject* left_object,
        PyObject* right_object) {
    if (PyObject_TypeCheck(left_object, &PyFastCDElementType)
            && PyObject_TypeCheck(right_object, &PyFastCDElementType)) {
        PyFastCDElement* left = (PyFastCDElement*)left_object;
        PyFastCDElement* right = (PyFastCDElement*)right_object;
        PyFastCDElement* result;
        FastCDStatus status;

        if (pyfastcd_check_same_dimension(left, right) < 0) return NULL;
        result = pyfastcd_element_allocate(left->algebra,
                (size_t)Py_SIZE(left));
        if (result == NULL) return NULL;
        status = fastcd_sub_f64(result->coefficients, left->coefficients,
                right->coefficients, (size_t)Py_SIZE(left));
        if (status != FASTCD_SUCCESS) {
            Py_DECREF(result);
            pyfastcd_set_status_error(status);
            return NULL;
        }
        return (PyObject*)result;
    }
    if (PyObject_TypeCheck(left_object, &PyFastCDElementType)) {
        PyFastCDElement* left = (PyFastCDElement*)left_object;
        PyFastCDElement* result;
        double scalar;
        int scalar_result = pyfastcd_as_real_scalar(right_object, &scalar);

        if (!scalar_result) {
            if (PyErr_Occurred()) return NULL;
            Py_RETURN_NOTIMPLEMENTED;
        }
        result = pyfastcd_element_copy(left);
        if (result == NULL) return NULL;
        result->coefficients[0] -= scalar;
        return (PyObject*)result;
    }
    if (PyObject_TypeCheck(right_object, &PyFastCDElementType)) {
        PyFastCDElement* right = (PyFastCDElement*)right_object;
        PyFastCDElement* result;
        double scalar;
        int scalar_result = pyfastcd_as_real_scalar(left_object, &scalar);

        if (!scalar_result) {
            if (PyErr_Occurred()) return NULL;
            Py_RETURN_NOTIMPLEMENTED;
        }
        result = pyfastcd_element_allocate(right->algebra,
                (size_t)Py_SIZE(right));
        if (result == NULL) return NULL;
        result->coefficients[0] = scalar - right->coefficients[0];
        for (Py_ssize_t i = 1; i < Py_SIZE(right); i++) {
            result->coefficients[i] = -right->coefficients[i];
        }
        return (PyObject*)result;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

// Multiply compatible elements or scale an element by a real scalar
static PyObject* pyfastcd_element_multiply(PyObject* left_object,
        PyObject* right_object) {
    if (PyObject_TypeCheck(left_object, &PyFastCDElementType)
            && PyObject_TypeCheck(right_object, &PyFastCDElementType)) {
        PyFastCDElement* left = (PyFastCDElement*)left_object;
        PyFastCDElement* right = (PyFastCDElement*)right_object;
        size_t dimension = (size_t)Py_SIZE(left);
        PyFastCDElement* result;
        PyFastCDWorkspace workspace;
        FastCDStatus status;

        if (pyfastcd_check_same_dimension(left, right) < 0) return NULL;
        result = pyfastcd_element_allocate(left->algebra, dimension);
        if (result == NULL) return NULL;
        if (pyfastcd_workspace_for_query(&workspace, dimension,
                    fastcd_mul_workspace_size) < 0) {
            Py_DECREF(result);
            return NULL;
        }
        status = fastcd_mul_f64(result->coefficients, left->coefficients,
                right->coefficients, dimension, &workspace.value);
        pyfastcd_workspace_release(&workspace);
        if (status != FASTCD_SUCCESS) {
            Py_DECREF(result);
            pyfastcd_set_status_error(status);
            return NULL;
        }
        return (PyObject*)result;
    }
    {
        PyFastCDElement* element;
        PyFastCDElement* result;
        double scalar;
        int scalar_result;
        FastCDStatus status;

        if (PyObject_TypeCheck(left_object, &PyFastCDElementType)) {
            element = (PyFastCDElement*)left_object;
            scalar_result = pyfastcd_as_real_scalar(right_object, &scalar);
        } else if (PyObject_TypeCheck(right_object, &PyFastCDElementType)) {
            element = (PyFastCDElement*)right_object;
            scalar_result = pyfastcd_as_real_scalar(left_object, &scalar);
        } else {
            Py_RETURN_NOTIMPLEMENTED;
        }
        if (!scalar_result) {
            if (PyErr_Occurred()) return NULL;
            Py_RETURN_NOTIMPLEMENTED;
        }
        result = pyfastcd_element_allocate(element->algebra,
                (size_t)Py_SIZE(element));
        if (result == NULL) return NULL;
        status = fastcd_scale_f64(result->coefficients,
                element->coefficients, scalar, (size_t)Py_SIZE(element));
        if (status != FASTCD_SUCCESS) {
            Py_DECREF(result);
            pyfastcd_set_status_error(status);
            return NULL;
        }
        return (PyObject*)result;
    }
}

// Return the coordinatewise negation of an element
static PyObject* pyfastcd_element_negative(PyObject* self_object) {
    PyFastCDElement* self = (PyFastCDElement*)self_object;
    PyFastCDElement* result = pyfastcd_element_allocate(self->algebra,
            (size_t)Py_SIZE(self));
    FastCDStatus status;

    if (result == NULL) return NULL;
    status = fastcd_neg_f64(result->coefficients, self->coefficients,
            (size_t)Py_SIZE(self));
    if (status != FASTCD_SUCCESS) {
        Py_DECREF(result);
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return (PyObject*)result;
}

// Return an independent copy for unary positive
static PyObject* pyfastcd_element_positive(PyObject* self_object) {
    return (PyObject*)pyfastcd_element_copy(
            (PyFastCDElement*)self_object);
}

// Return the Euclidean norm used by Python absolute value
static PyObject* pyfastcd_element_absolute(PyObject* self_object) {
    PyFastCDElement* self = (PyFastCDElement*)self_object;
    double value;
    FastCDStatus status = fastcd_norm_f64(&value, self->coefficients,
            (size_t)Py_SIZE(self));

    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return PyFloat_FromDouble(value);
}

// Divide an element coordinatewise by a nonzero real scalar
static PyObject* pyfastcd_element_true_divide(PyObject* left_object,
        PyObject* right_object) {
    PyFastCDElement* left;
    PyFastCDElement* result;
    double scalar;
    int scalar_result;

    if (!PyObject_TypeCheck(left_object, &PyFastCDElementType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    left = (PyFastCDElement*)left_object;
    scalar_result = pyfastcd_as_real_scalar(right_object, &scalar);
    if (!scalar_result) {
        if (PyErr_Occurred()) return NULL;
        Py_RETURN_NOTIMPLEMENTED;
    }
    if (scalar == 0.0) {
        PyErr_SetString(PyExc_ZeroDivisionError, "division by zero scalar");
        return NULL;
    }
    result = pyfastcd_element_allocate(left->algebra,
            (size_t)Py_SIZE(left));
    if (result == NULL) return NULL;
    for (Py_ssize_t i = 0; i < Py_SIZE(left); i++) {
        result->coefficients[i] = left->coefficients[i] / scalar;
    }
    return (PyObject*)result;
}

// Evaluate an integer element power by repeated squaring
static PyObject* pyfastcd_element_power(PyObject* base_object,
        PyObject* exponent_object, PyObject* modulus) {
    PyFastCDElement* base_element;
    PyFastCDElement* result;
    PyObject* indexed_exponent;
    long long signed_exponent;
    unsigned long long exponent;
    size_t dimension;
    size_t workspace_length;
    double* storage;
    double* accumulator;
    double* base;
    double* temporary;
    PyFastCDWorkspace workspace;
    FastCDStatus status;
    bool accumulator_initialized = false;

    if (!PyObject_TypeCheck(base_object, &PyFastCDElementType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    if (modulus != Py_None) {
        PyErr_SetString(PyExc_TypeError,
                PYFASTCD_MODULAR_POWER_MESSAGE);
        return NULL;
    }
    if (!PyIndex_Check(exponent_object)) {
        PyErr_SetString(PyExc_TypeError,
                PYFASTCD_INTEGER_EXPONENT_MESSAGE);
        return NULL;
    }
    indexed_exponent = PyNumber_Index(exponent_object);
    if (indexed_exponent == NULL) return NULL;
    signed_exponent = PyLong_AsLongLong(indexed_exponent);
    Py_DECREF(indexed_exponent);
    if (signed_exponent == -1 && PyErr_Occurred()) return NULL;
    base_element = (PyFastCDElement*)base_object;
    dimension = (size_t)Py_SIZE(base_element);

    if (signed_exponent == 0) {
        result = pyfastcd_element_allocate(base_element->algebra, dimension);
        if (result == NULL) return NULL;
        status = fastcd_scalar_f64(result->coefficients, dimension, 1.0);
        if (status != FASTCD_SUCCESS) {
            Py_DECREF(result);
            pyfastcd_set_status_error(status);
            return NULL;
        }
        return (PyObject*)result;
    }
    if (signed_exponent == 1) {
        return (PyObject*)pyfastcd_element_copy(base_element);
    }
    if (signed_exponent == 2) {
        result = pyfastcd_element_allocate(base_element->algebra, dimension);
        if (result == NULL) return NULL;
        status = fastcd_square_f64(result->coefficients,
                base_element->coefficients, dimension);
        if (status != FASTCD_SUCCESS) {
            Py_DECREF(result);
            pyfastcd_set_status_error(status);
            return NULL;
        }
        return (PyObject*)result;
    }

    status = fastcd_mul_workspace_size(dimension, &workspace_length);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    if (dimension > SIZE_MAX / (3 * sizeof(double))) {
        PyErr_SetString(PyExc_OverflowError,
                "power temporary-storage size overflow");
        return NULL;
    }
    storage = PyMem_Malloc(3 * dimension * sizeof(double));
    if (storage == NULL) return PyErr_NoMemory();
    accumulator = storage;
    base = storage + dimension;
    temporary = storage + 2 * dimension;
    if (pyfastcd_workspace_allocate(&workspace, workspace_length, false) < 0) {
        PyMem_Free(storage);
        return NULL;
    }
    memcpy(base, base_element->coefficients, dimension * sizeof(double));
    if (signed_exponent < 0) {
        status = fastcd_reciprocal_f64(base, base, dimension);
        if (status != FASTCD_SUCCESS) goto power_error;
        exponent = signed_exponent == LLONG_MIN
            ? (unsigned long long)LLONG_MAX + 1ULL
            : (unsigned long long)(-signed_exponent);
    } else {
        exponent = (unsigned long long)signed_exponent;
    }

    while (exponent != 0) {
        if ((exponent & 1ULL) != 0) {
            if (!accumulator_initialized) {
                memcpy(accumulator, base, dimension * sizeof(double));
                accumulator_initialized = true;
            } else {
                status = fastcd_mul_f64(temporary, accumulator, base,
                        dimension, &workspace.value);
                if (status != FASTCD_SUCCESS) goto power_error;
                memcpy(accumulator, temporary, dimension * sizeof(double));
            }
        }
        exponent >>= 1;
        if (exponent != 0) {
            status = fastcd_square_f64(base, base, dimension);
            if (status != FASTCD_SUCCESS) goto power_error;
        }
    }
    result = pyfastcd_element_allocate(base_element->algebra, dimension);
    if (result == NULL) goto power_python_error;
    memcpy(result->coefficients, accumulator, dimension * sizeof(double));
    pyfastcd_workspace_release(&workspace);
    PyMem_Free(storage);
    return (PyObject*)result;

power_error:
    pyfastcd_set_status_error(status);
power_python_error:
    pyfastcd_workspace_release(&workspace);
    PyMem_Free(storage);
    return NULL;
}

/* ---------- Named scalar operations ---------- */

// Dispatch a named unary operation and return a new element
static PyObject* pyfastcd_unary_element_result(PyFastCDElement* self,
        PyFastCDUnaryOperation operation) {
    size_t dimension = (size_t)Py_SIZE(self);
    PyFastCDElement* result = pyfastcd_element_allocate(self->algebra,
            dimension);
    FastCDStatus status = FASTCD_SUCCESS;

    if (result == NULL) return NULL;
    switch (operation) {
        case PYFASTCD_UNARY_CONJUGATE:
            status = fastcd_conjugate_f64(result->coefficients,
                    self->coefficients, dimension);
            break;
        case PYFASTCD_UNARY_IMAG_PART:
            status = fastcd_imag_part_f64(result->coefficients,
                    self->coefficients, dimension);
            break;
        case PYFASTCD_UNARY_NORMALIZE:
            status = fastcd_normalize_f64(result->coefficients,
                    self->coefficients, dimension);
            break;
        case PYFASTCD_UNARY_RECIPROCAL:
            status = fastcd_reciprocal_f64(result->coefficients,
                    self->coefficients, dimension);
            break;
        case PYFASTCD_UNARY_SQUARE:
            status = fastcd_square_f64(result->coefficients,
                    self->coefficients, dimension);
            break;
        default:
            Py_DECREF(result);
            PyErr_SetString(PyExc_RuntimeError,
                    "unknown fastCD unary operation");
            return NULL;
    }
    if (status != FASTCD_SUCCESS) {
        Py_DECREF(result);
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return (PyObject*)result;
}

// Dispatch a named binary operation and return a new element
static PyObject* pyfastcd_binary_element_result(PyFastCDElement* self,
        PyObject* other_object, PyFastCDBinaryOperation operation) {
    PyFastCDElement* other;
    PyFastCDElement* result;
    PyFastCDWorkspace workspace;
    size_t dimension = (size_t)Py_SIZE(self);
    FastCDStatus status;

    if (!PyObject_TypeCheck(other_object, &PyFastCDElementType)) {
        PyErr_SetString(PyExc_TypeError,
                "operand must be a pyfastcd.Element");
        return NULL;
    }
    other = (PyFastCDElement*)other_object;
    if (pyfastcd_check_same_dimension(self, other) < 0) return NULL;
    result = pyfastcd_element_allocate(self->algebra, dimension);
    if (result == NULL) return NULL;

    switch (operation) {
        case PYFASTCD_BINARY_MULTIPLY:
            if (pyfastcd_workspace_for_query(&workspace, dimension,
                        fastcd_mul_workspace_size) < 0) {
                Py_DECREF(result);
                return NULL;
            }
            status = fastcd_mul_f64(result->coefficients,
                    self->coefficients, other->coefficients, dimension,
                    &workspace.value);
            break;
        case PYFASTCD_BINARY_IMAG_CROSS:
        case PYFASTCD_BINARY_COMMUTATOR:
            if (pyfastcd_workspace_for_query(&workspace, dimension,
                        fastcd_imag_cross_workspace_size) < 0) {
                Py_DECREF(result);
                return NULL;
            }
            status = operation == PYFASTCD_BINARY_IMAG_CROSS
                ? fastcd_imag_cross_f64(result->coefficients,
                        self->coefficients, other->coefficients, dimension,
                        &workspace.value)
                : fastcd_commutator_f64(result->coefficients,
                        self->coefficients, other->coefficients, dimension,
                        &workspace.value);
            break;
        case PYFASTCD_BINARY_ANTICOMMUTATOR:
            workspace.heap_allocated = false;
            workspace.value.data = NULL;
            workspace.value.length = 0;
            status = fastcd_anticommutator_f64(result->coefficients,
                    self->coefficients, other->coefficients, dimension);
            break;
        default:
            Py_DECREF(result);
            PyErr_SetString(PyExc_RuntimeError,
                    "unknown fastCD binary operation");
            return NULL;
    }
    pyfastcd_workspace_release(&workspace);
    if (status != FASTCD_SUCCESS) {
        Py_DECREF(result);
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return (PyObject*)result;
}

// Expose element copying through the Python method table
static PyObject* pyfastcd_element_copy_method(PyObject* self,
        PyObject* Py_UNUSED(arguments)) {
    return (PyObject*)pyfastcd_element_copy((PyFastCDElement*)self);
}

PyDoc_STRVAR(pyfastcd_element_copy_doc,
"copy($self, /)\n"
"--\n\n"
"Return an independent copy of this element.\n\n"
"Returns\n"
"-------\n"
"Element\n"
"    A new immutable element with a distinct inline coefficient buffer.");

// Expose Cayley--Dickson conjugation as an element method
static PyObject* pyfastcd_element_conjugate(PyObject* self,
        PyObject* Py_UNUSED(arguments)) {
    return pyfastcd_unary_element_result((PyFastCDElement*)self,
            PYFASTCD_UNARY_CONJUGATE);
}

PyDoc_STRVAR(pyfastcd_element_conjugate_doc,
"conjugate($self, /)\n"
"--\n\n"
"Return the standard Cayley--Dickson conjugate.\n\n"
"Returns\n"
"-------\n"
"Element\n"
"    A new element with the scalar coordinate preserved and every\n"
"    imaginary coordinate negated.");

// Expose imaginary-part extraction as an element method
static PyObject* pyfastcd_element_imag_part(PyObject* self,
        PyObject* Py_UNUSED(arguments)) {
    return pyfastcd_unary_element_result((PyFastCDElement*)self,
            PYFASTCD_UNARY_IMAG_PART);
}

PyDoc_STRVAR(pyfastcd_element_imag_part_doc,
"imag_part($self, /)\n"
"--\n\n"
"Return the imaginary part of this element.\n\n"
"Returns\n"
"-------\n"
"Element\n"
"    A new element with coordinate zero replaced by zero and all other\n"
"    coordinates preserved.");

// Return the Euclidean coefficient inner product of two elements
static PyObject* pyfastcd_element_inner(PyObject* self_object,
        PyObject* other_object) {
    PyFastCDElement* self = (PyFastCDElement*)self_object;
    PyFastCDElement* other;
    double value;
    FastCDStatus status;

    if (!PyObject_TypeCheck(other_object, &PyFastCDElementType)) {
        PyErr_SetString(PyExc_TypeError,
                "inner-product operand must be a pyfastcd.Element");
        return NULL;
    }
    other = (PyFastCDElement*)other_object;
    if (pyfastcd_check_same_dimension(self, other) < 0) return NULL;
    status = fastcd_inner_f64(&value, self->coefficients,
            other->coefficients, (size_t)Py_SIZE(self));
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return PyFloat_FromDouble(value);
}

PyDoc_STRVAR(pyfastcd_element_inner_doc,
"inner($self, other, /)\n"
"--\n\n"
"Return the Euclidean coefficient inner product with ``other``.\n\n"
"Parameters\n"
"----------\n"
"other : Element\n"
"    Element of the same algebra.\n\n"
"Returns\n"
"-------\n"
"float\n"
"    The sequential double-precision sum of coordinate products.\n\n"
"Raises\n"
"------\n"
"TypeError\n"
"    If ``other`` is not an Element.\n"
"ValueError\n"
"    If the two elements have different dimensions.\n\n"
"Notes\n"
"-----\n"
"Accumulation is not compensated or scaled.");

// Return the squared Euclidean coefficient norm of an element
static PyObject* pyfastcd_element_norm_squared(PyObject* self_object,
        PyObject* Py_UNUSED(arguments)) {
    PyFastCDElement* self = (PyFastCDElement*)self_object;
    double value;
    FastCDStatus status = fastcd_norm_squared_f64(&value,
            self->coefficients, (size_t)Py_SIZE(self));

    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return PyFloat_FromDouble(value);
}

PyDoc_STRVAR(pyfastcd_element_norm_squared_doc,
"norm_squared($self, /)\n"
"--\n\n"
"Return the squared Euclidean coefficient norm.\n\n"
"Returns\n"
"-------\n"
"float\n"
"    The value ``sum(self[i]**2)``.\n\n"
"Notes\n"
"-----\n"
"The native routine uses scaled accumulation to reduce avoidable overflow\n"
"and underflow.  NaN and infinity follow IEEE-754 behavior.");

// Return the Euclidean coefficient norm of an element
static PyObject* pyfastcd_element_norm(PyObject* self_object,
        PyObject* Py_UNUSED(arguments)) {
    return pyfastcd_element_absolute(self_object);
}

PyDoc_STRVAR(pyfastcd_element_norm_doc,
"norm($self, /)\n"
"--\n\n"
"Return the Euclidean coefficient norm.\n\n"
"Returns\n"
"-------\n"
"float\n"
"    The same value returned by ``abs(self)``.\n\n"
"Notes\n"
"-----\n"
"Scaled accumulation reduces avoidable finite overflow and underflow.");

// Return a normalized copy of an element
static PyObject* pyfastcd_element_normalized(PyObject* self,
        PyObject* Py_UNUSED(arguments)) {
    return pyfastcd_unary_element_result((PyFastCDElement*)self,
            PYFASTCD_UNARY_NORMALIZE);
}

PyDoc_STRVAR(pyfastcd_element_normalized_doc,
"normalized($self, /)\n"
"--\n\n"
"Return this element divided by its Euclidean norm.\n\n"
"Returns\n"
"-------\n"
"Element\n"
"    A newly allocated normalized element.\n\n"
"Raises\n"
"------\n"
"ZeroDivisionError\n"
"    If every coefficient is exactly zero.  No near-zero tolerance is\n"
"    applied.\n\n"
"Notes\n"
"-----\n"
"Scaled norm accumulation reduces avoidable finite overflow and underflow.\n"
"NaN and infinity follow the native IEEE-754 path.");

// Return the canonical conjugate-over-norm-squared reciprocal
static PyObject* pyfastcd_element_reciprocal(PyObject* self,
        PyObject* Py_UNUSED(arguments)) {
    return pyfastcd_unary_element_result((PyFastCDElement*)self,
            PYFASTCD_UNARY_RECIPROCAL);
}

PyDoc_STRVAR(pyfastcd_element_reciprocal_doc,
"reciprocal($self, /)\n"
"--\n\n"
"Return ``conjugate(self) / norm_squared(self)``.\n\n"
"Returns\n"
"-------\n"
"Element\n"
"    The newly allocated canonical reciprocal.\n\n"
"Raises\n"
"------\n"
"ZeroDivisionError\n"
"    If every coefficient is exactly zero.  No near-zero tolerance is\n"
"    applied.\n\n"
"Notes\n"
"-----\n"
"The result satisfies the individual products ``self*r == 1`` and\n"
"``r*self == 1`` in exact arithmetic.  In higher nonassociative algebras it\n"
"is not a general left- or right-division solver.");

// Return an element square through the specialized native kernel
static PyObject* pyfastcd_element_square(PyObject* self,
        PyObject* Py_UNUSED(arguments)) {
    return pyfastcd_unary_element_result((PyFastCDElement*)self,
            PYFASTCD_UNARY_SQUARE);
}

PyDoc_STRVAR(pyfastcd_element_square_doc,
"square($self, /)\n"
"--\n\n"
"Return the square using the specialized linear-time identity.\n\n"
"Returns\n"
"-------\n"
"Element\n"
"    A new element equal in exact arithmetic to ``self**2``.\n\n"
"Notes\n"
"-----\n"
"For ``self = alpha + u``, the kernel evaluates\n"
"``(alpha**2 - inner(u, u)) + 2*alpha*u``.  Its operation order may round\n"
"differently from general ``self*self``.");

// Return the generalized imaginary cross product of two elements
static PyObject* pyfastcd_element_imag_cross(PyObject* self,
        PyObject* other) {
    return pyfastcd_binary_element_result((PyFastCDElement*)self, other,
            PYFASTCD_BINARY_IMAG_CROSS);
}

PyDoc_STRVAR(pyfastcd_element_imag_cross_doc,
"imag_cross($self, other, /)\n"
"--\n\n"
"Return the generalized imaginary cross product with ``other``.\n\n"
"Parameters\n"
"----------\n"
"other : Element\n"
"    Element of the same algebra.\n\n"
"Returns\n"
"-------\n"
"Element\n"
"    The value ``(u*v - v*u)/2``, where ``u`` and ``v`` are the imaginary\n"
"    parts.  Its scalar coordinate is zero.\n\n"
"Raises\n"
"------\n"
"TypeError\n"
"    If ``other`` is not an Element.\n"
"ValueError\n"
"    If the two elements have different dimensions.\n\n"
"Notes\n"
"-----\n"
"Scalar coordinates are ignored.  At dimension four this agrees with the\n"
"ordinary vector cross product under fastCD's basis orientation; higher-\n"
"dimensional vector identities need not hold.");

// Return the multiplication commutator of two elements
static PyObject* pyfastcd_element_commutator(PyObject* self,
        PyObject* other) {
    return pyfastcd_binary_element_result((PyFastCDElement*)self, other,
            PYFASTCD_BINARY_COMMUTATOR);
}

PyDoc_STRVAR(pyfastcd_element_commutator_doc,
"commutator($self, other, /)\n"
"--\n\n"
"Return the algebra commutator with ``other``.\n\n"
"Parameters\n"
"----------\n"
"other : Element\n"
"    Element of the same algebra.\n\n"
"Returns\n"
"-------\n"
"Element\n"
"    A new value ``self*other - other*self``.\n\n"
"Raises\n"
"------\n"
"TypeError\n"
"    If ``other`` is not an Element.\n"
"ValueError\n"
"    If the two elements have different dimensions.\n\n"
"Notes\n"
"-----\n"
"The optimized matrix-free alternating-product schedule does not form and\n"
"subtract two independent complete products.");

// Return the multiplication anticommutator of two elements
static PyObject* pyfastcd_element_anticommutator(PyObject* self,
        PyObject* other) {
    return pyfastcd_binary_element_result((PyFastCDElement*)self, other,
            PYFASTCD_BINARY_ANTICOMMUTATOR);
}

PyDoc_STRVAR(pyfastcd_element_anticommutator_doc,
"anticommutator($self, other, /)\n"
"--\n\n"
"Return the algebra anticommutator with ``other``.\n\n"
"Parameters\n"
"----------\n"
"other : Element\n"
"    Element of the same algebra.\n\n"
"Returns\n"
"-------\n"
"Element\n"
"    A new value ``self*other + other*self``.\n\n"
"Raises\n"
"------\n"
"TypeError\n"
"    If ``other`` is not an Element.\n"
"ValueError\n"
"    If the two elements have different dimensions.\n\n"
"Notes\n"
"-----\n"
"The native implementation uses a linear-time scalar--imaginary identity\n"
"rather than forming two complete products.");

// Return the associator of three compatible elements
static PyObject* pyfastcd_element_associator(PyObject* self_object,
        PyObject* args) {
    PyFastCDElement* self = (PyFastCDElement*)self_object;
    PyObject* y_object;
    PyObject* z_object;
    PyFastCDElement* y;
    PyFastCDElement* z;
    PyFastCDElement* result;
    PyFastCDWorkspace workspace;
    size_t dimension = (size_t)Py_SIZE(self);
    FastCDStatus status;

    if (!PyArg_ParseTuple(args, "OO:associator", &y_object, &z_object)) {
        return NULL;
    }
    if (!PyObject_TypeCheck(y_object, &PyFastCDElementType)
            || !PyObject_TypeCheck(z_object, &PyFastCDElementType)) {
        PyErr_SetString(PyExc_TypeError,
                "associator operands must be pyfastcd.Element objects");
        return NULL;
    }
    y = (PyFastCDElement*)y_object;
    z = (PyFastCDElement*)z_object;
    if (pyfastcd_check_same_dimension(self, y) < 0
            || pyfastcd_check_same_dimension(self, z) < 0) {
        return NULL;
    }
    result = pyfastcd_element_allocate(self->algebra, dimension);
    if (result == NULL) return NULL;
    if (pyfastcd_workspace_for_query(&workspace, dimension,
                fastcd_associator_workspace_size) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    status = fastcd_associator_f64(result->coefficients,
            self->coefficients, y->coefficients, z->coefficients,
            dimension, &workspace.value);
    pyfastcd_workspace_release(&workspace);
    if (status != FASTCD_SUCCESS) {
        Py_DECREF(result);
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return (PyObject*)result;
}

PyDoc_STRVAR(pyfastcd_element_associator_doc,
"associator($self, y, z, /)\n"
"--\n\n"
"Return the associator ``(self*y)*z - self*(y*z)``.\n\n"
"Parameters\n"
"----------\n"
"y, z : Element\n"
"    Elements of the same algebra as ``self``.\n\n"
"Returns\n"
"-------\n"
"Element\n"
"    The newly allocated associator.\n\n"
"Raises\n"
"------\n"
"TypeError\n"
"    If either operand is not an Element.\n"
"ValueError\n"
"    If an element has a different dimension.\n\n"
"Notes\n"
"-----\n"
"The displayed parenthesization is part of the numerical contract once\n"
"multiplication is nonassociative.");

// Export element coefficients to a copied or zero-copy NumPy array
static PyObject* pyfastcd_element_to_numpy(PyObject* self,
        PyObject* args, PyObject* keywords) {
    static char* keyword_names[] = {"copy", NULL};
    PyObject* copy_object = Py_True;
    PyObject* numpy;
    PyObject* array;
    PyObject* result;

    if (!PyArg_ParseTupleAndKeywords(args, keywords, "|$O:to_numpy",
                keyword_names, &copy_object)) {
        return NULL;
    }
    if (!PyBool_Check(copy_object)) {
        PyErr_SetString(PyExc_TypeError, "copy must be True or False");
        return NULL;
    }
    numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) return NULL;
    array = PyObject_CallMethod(numpy, "asarray", "O", self);
    Py_DECREF(numpy);
    if (array == NULL) return NULL;
    if (copy_object == Py_False) return array;
    result = PyObject_CallMethod(array, "copy", NULL);
    Py_DECREF(array);
    return result;
}

PyDoc_STRVAR(pyfastcd_element_to_numpy_doc,
"to_numpy($self, /, *, copy=True)\n"
"--\n\n"
"Return the coefficients as a NumPy array.\n\n"
"Parameters\n"
"----------\n"
"copy : bool, optional\n"
"    If true, return independent writable storage.  If false, return a\n"
"    zero-copy read-only view.\n\n"
"Returns\n"
"-------\n"
"numpy.ndarray\n"
"    Native ``float64`` coefficients with shape ``(dimension,)``.\n\n"
"Raises\n"
"------\n"
"TypeError\n"
"    If ``copy`` is not ``True`` or ``False``.\n\n"
"Notes\n"
"-----\n"
"A zero-copy view retains this Element as its owner.  ``np.asarray(self)``\n"
"and ``memoryview(self)`` expose the same read-only native storage.");

// Return the algebra call and coefficients needed to pickle an element
static PyObject* pyfastcd_element_reduce(PyObject* self_object,
        PyObject* Py_UNUSED(arguments)) {
    PyFastCDElement* self = (PyFastCDElement*)self_object;
    PyObject* coefficients = PyTuple_New(Py_SIZE(self));
    PyObject* constructor_arguments;
    PyObject* result;

    if (coefficients == NULL) return NULL;
    for (Py_ssize_t i = 0; i < Py_SIZE(self); i++) {
        PyObject* value = PyFloat_FromDouble(self->coefficients[i]);
        if (value == NULL) {
            Py_DECREF(coefficients);
            return NULL;
        }
        PyTuple_SET_ITEM(coefficients, i, value);
    }
    constructor_arguments = PyTuple_Pack(1, coefficients);
    Py_DECREF(coefficients);
    if (constructor_arguments == NULL) return NULL;
    result = PyTuple_Pack(2, self->algebra, constructor_arguments);
    Py_DECREF(constructor_arguments);
    return result;
}

// Return the algebra parent exposed by the element descriptor
static PyObject* pyfastcd_element_get_algebra(PyObject* self, void* closure) {
    (void)closure;
    return Py_NewRef(((PyFastCDElement*)self)->algebra);
}

// Return the element coefficient dimension exposed by the descriptor
static PyObject* pyfastcd_element_get_dimension(PyObject* self,
        void* closure) {
    (void)closure;
    return PyLong_FromSsize_t(Py_SIZE((PyFastCDElement*)self));
}

// Return the doubling level exposed by the element descriptor
static PyObject* pyfastcd_element_get_level(PyObject* self, void* closure) {
    size_t level;
    FastCDStatus status;

    (void)closure;
    status = fastcd_level((size_t)Py_SIZE((PyFastCDElement*)self), &level);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return PyLong_FromSize_t(level);
}

// Return the scalar coordinate exposed by the element descriptor
static PyObject* pyfastcd_element_get_real(PyObject* self, void* closure) {
    (void)closure;
    return PyFloat_FromDouble(((PyFastCDElement*)self)->coefficients[0]);
}

// Return the imaginary part exposed by the element descriptor
static PyObject* pyfastcd_element_get_imag(PyObject* self, void* closure) {
    (void)closure;
    return pyfastcd_unary_element_result((PyFastCDElement*)self,
            PYFASTCD_UNARY_IMAG_PART);
}

// Return a read-only NumPy view exposed by the coefficients descriptor
static PyObject* pyfastcd_element_get_coefficients(PyObject* self,
        void* closure) {
    PyObject* numpy;
    PyObject* result;

    (void)closure;
    numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) return NULL;
    result = PyObject_CallMethod(numpy, "asarray", "O", self);
    Py_DECREF(numpy);
    return result;
}

// Return the one-dimensional coefficient shape exposed to NumPy
static PyObject* pyfastcd_element_get_shape(PyObject* self, void* closure) {
    (void)closure;
    return Py_BuildValue("(n)", Py_SIZE((PyFastCDElement*)self));
}

// Return the native float64 dtype exposed to NumPy
static PyObject* pyfastcd_element_get_dtype(PyObject* self, void* closure) {
    PyObject* numpy;
    PyObject* result;

    (void)self;
    (void)closure;
    numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) return NULL;
    result = PyObject_CallMethod(numpy, "dtype", "s", "float64");
    Py_DECREF(numpy);
    return result;
}

// Return the priority used to route mixed NumPy operations to fastCD
static PyObject* pyfastcd_element_get_array_priority(PyObject* self,
        void* closure) {
    (void)self;
    (void)closure;
    return PyFloat_FromDouble(1000.0);
}

// Delegate supported NumPy ufunc dispatch to the Python operator layer
static PyObject* pyfastcd_element_array_ufunc(PyObject* self,
        PyObject* args, PyObject* keywords) {
    PyObject* module;
    PyObject* function;
    PyObject* full_arguments;
    PyObject* result;
    Py_ssize_t argument_count = PyTuple_GET_SIZE(args);

    module = PyImport_ImportModule("pyfastcd._operators");
    if (module == NULL) return NULL;
    function = PyObject_GetAttrString(module, "_element_array_ufunc");
    Py_DECREF(module);
    if (function == NULL) return NULL;
    full_arguments = PyTuple_New(argument_count + 1);
    if (full_arguments == NULL) {
        Py_DECREF(function);
        return NULL;
    }
    PyTuple_SET_ITEM(full_arguments, 0, Py_NewRef(self));
    for (Py_ssize_t i = 0; i < argument_count; i++) {
        PyTuple_SET_ITEM(full_arguments, i + 1,
                Py_NewRef(PyTuple_GET_ITEM(args, i)));
    }
    result = PyObject_Call(function, full_arguments, keywords);
    Py_DECREF(full_arguments);
    Py_DECREF(function);
    return result;
}

PyDoc_STRVAR(pyfastcd_element_array_ufunc_doc,
"__array_ufunc__($self, ufunc, method, *inputs, **kwargs)\n"
"--\n\n"
"Dispatch the intentional NumPy ufunc subset algebraically.\n\n"
"Supported ``__call__`` ufuncs are ``add``, ``subtract``, ``negative``,\n"
"``multiply``, ``square``, ``conjugate``, ``absolute``, ``equal``, and\n"
"``not_equal``.  Multiplication remains Cayley--Dickson multiplication, never\n"
"coordinatewise multiplication.  Only multiplication accepts ``out=``; the\n"
"output must be an exact-shape ElementArray or native ``float64`` ndarray.\n\n"
"Unsupported ufuncs, nonscalar raw ndarray operands, methods such as\n"
"``reduce``, and extra keyword options return ``NotImplemented`` and normally\n"
"cause NumPy to raise ``TypeError``.  Numeric zero-dimensional ndarrays are\n"
"treated as real scalars.");

PyDoc_STRVAR(pyfastcd_element_algebra_doc,
"Algebra: The cached parent algebra.\n\n"
"All element-valued results retain this parent.");

PyDoc_STRVAR(pyfastcd_element_dimension_doc,
"int: Number of real coefficients in this element.");

PyDoc_STRVAR(pyfastcd_element_level_doc,
"int: Doubling level ``n`` satisfying ``dimension == 2**n``.");

PyDoc_STRVAR(pyfastcd_element_real_doc,
"float: Scalar coordinate, stored at index zero.");

PyDoc_STRVAR(pyfastcd_element_imag_doc,
"Element: Imaginary part in newly allocated storage.\n\n"
"Coordinate zero is replaced by zero.  This is equivalent to\n"
"``self.imag_part()`` and never aliases ``self``.");

PyDoc_STRVAR(pyfastcd_element_coefficients_doc,
"numpy.ndarray: Read-only zero-copy view of the coefficients.\n\n"
"The native ``float64`` view has shape ``(dimension,)`` and retains this\n"
"Element as its owner.  Use ``to_numpy(copy=True)`` for writable storage.");

PyDoc_STRVAR(pyfastcd_element_shape_doc,
"tuple[int]: Coefficient shape, always ``(dimension,)``.");

PyDoc_STRVAR(pyfastcd_element_dtype_doc,
"numpy.dtype: Coefficient dtype, always native ``float64``.");

PyDoc_STRVAR(pyfastcd_element_array_priority_doc,
"float: NumPy dispatch priority, currently ``1000.0``.\n\n"
"This lets supported mixed ufunc calls reach fastCD before ordinary ndarray\n"
"coordinate semantics are selected.");

static PyGetSetDef pyfastcd_element_getset[] = {
    {"algebra", pyfastcd_element_get_algebra, NULL,
        pyfastcd_element_algebra_doc, NULL},
    {"dimension", pyfastcd_element_get_dimension, NULL,
        pyfastcd_element_dimension_doc, NULL},
    {"level", pyfastcd_element_get_level, NULL,
        pyfastcd_element_level_doc, NULL},
    {"real", pyfastcd_element_get_real, NULL,
        pyfastcd_element_real_doc, NULL},
    {"imag", pyfastcd_element_get_imag, NULL,
        pyfastcd_element_imag_doc, NULL},
    {"coefficients", pyfastcd_element_get_coefficients, NULL,
        pyfastcd_element_coefficients_doc, NULL},
    {"shape", pyfastcd_element_get_shape, NULL,
        pyfastcd_element_shape_doc, NULL},
    {"dtype", pyfastcd_element_get_dtype, NULL,
        pyfastcd_element_dtype_doc, NULL},
    {"__array_priority__", pyfastcd_element_get_array_priority, NULL,
        pyfastcd_element_array_priority_doc, NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyMethodDef pyfastcd_element_methods[] = {
    {"copy", pyfastcd_element_copy_method, METH_NOARGS,
        pyfastcd_element_copy_doc},
    {"conjugate", pyfastcd_element_conjugate, METH_NOARGS,
        pyfastcd_element_conjugate_doc},
    {"imag_part", pyfastcd_element_imag_part, METH_NOARGS,
        pyfastcd_element_imag_part_doc},
    {"inner", pyfastcd_element_inner, METH_O, pyfastcd_element_inner_doc},
    {"norm_squared", pyfastcd_element_norm_squared, METH_NOARGS,
        pyfastcd_element_norm_squared_doc},
    {"norm", pyfastcd_element_norm, METH_NOARGS,
        pyfastcd_element_norm_doc},
    {"normalized", pyfastcd_element_normalized, METH_NOARGS,
        pyfastcd_element_normalized_doc},
    {"reciprocal", pyfastcd_element_reciprocal, METH_NOARGS,
        pyfastcd_element_reciprocal_doc},
    {"square", pyfastcd_element_square, METH_NOARGS,
        pyfastcd_element_square_doc},
    {"imag_cross", pyfastcd_element_imag_cross, METH_O,
        pyfastcd_element_imag_cross_doc},
    {"commutator", pyfastcd_element_commutator, METH_O,
        pyfastcd_element_commutator_doc},
    {"anticommutator", pyfastcd_element_anticommutator, METH_O,
        pyfastcd_element_anticommutator_doc},
    {"associator", pyfastcd_element_associator, METH_VARARGS,
        pyfastcd_element_associator_doc},
    {"to_numpy", (PyCFunction)pyfastcd_element_to_numpy,
        METH_VARARGS | METH_KEYWORDS, pyfastcd_element_to_numpy_doc},
    {"__reduce__", pyfastcd_element_reduce, METH_NOARGS,
        "Return the pickle reconstruction data."},
    {"__array_ufunc__", (PyCFunction)pyfastcd_element_array_ufunc,
        METH_VARARGS | METH_KEYWORDS, pyfastcd_element_array_ufunc_doc},
    {NULL, NULL, 0, NULL}
};

/* ---------- Private scalar entry points used by Algebra ---------- */

// Map an algorithm name onto its code after checking dimension support
static PyObject* pyfastcd_algorithm_entry(PyObject* Py_UNUSED(module),
        PyObject* args) {
    PyObject* name;
    Py_ssize_t dimension_value;
    size_t dimension;
    FastCDAlgorithm algorithm;

    if (!PyArg_ParseTuple(args, "On:_algorithm", &name, &dimension_value)) {
        return NULL;
    }
    if (pyfastcd_algorithm_from_name(name, &algorithm) < 0
            || pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0
            || pyfastcd_require_algorithm_support(algorithm, dimension) < 0) {
        return NULL;
    }
    return PyLong_FromLong((long)algorithm);
}

// Return the name of the algorithm that automatic selection uses
static PyObject* pyfastcd_default_algorithm_entry(PyObject* Py_UNUSED(module),
        PyObject* args) {
    Py_ssize_t dimension_value;
    size_t dimension;
    FastCDAlgorithm algorithm;
    FastCDStatus status;

    if (!PyArg_ParseTuple(args, "n:_default_algorithm", &dimension_value)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    status = fastcd_default_algorithm(dimension, &algorithm);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return PyUnicode_FromString(fastcd_algorithm_string(algorithm));
}

// Return the name of the algorithm that automatic selection uses for batches
static PyObject* pyfastcd_default_batch_algorithm_entry(
        PyObject* Py_UNUSED(module), PyObject* args) {
    Py_ssize_t dimension_value;
    size_t dimension;
    FastCDAlgorithm algorithm;
    FastCDStatus status;

    if (!PyArg_ParseTuple(args, "n:_default_batch_algorithm",
                &dimension_value)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    status = fastcd_default_batch_algorithm(dimension, &algorithm);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return PyUnicode_FromString(fastcd_algorithm_string(algorithm));
}

// Multiply two elements with an explicitly selected algorithm
static PyObject* pyfastcd_multiply_element_entry(PyObject* Py_UNUSED(module),
        PyObject* args) {
    int algorithm_code;
    PyObject* left_object;
    PyObject* right_object;
    PyFastCDElement* left;
    PyFastCDElement* right;
    PyFastCDElement* result;
    FastCDAlgorithm algorithm;
    size_t dimension;
    PyFastCDWorkspace workspace;
    FastCDStatus status;

    if (!PyArg_ParseTuple(args, "iOO:_multiply_element", &algorithm_code,
                &left_object, &right_object)) {
        return NULL;
    }
    if (pyfastcd_algorithm_from_code(algorithm_code, &algorithm) < 0) {
        return NULL;
    }
    if (!PyObject_TypeCheck(left_object, &PyFastCDElementType)
            || !PyObject_TypeCheck(right_object, &PyFastCDElementType)) {
        PyErr_SetString(PyExc_TypeError,
                "multiplication requires two pyfastcd.Element operands");
        return NULL;
    }
    left = (PyFastCDElement*)left_object;
    right = (PyFastCDElement*)right_object;
    if (pyfastcd_check_same_dimension(left, right) < 0) return NULL;
    dimension = (size_t)Py_SIZE(left);
    if (pyfastcd_require_algorithm_support(algorithm, dimension) < 0
            || pyfastcd_algorithm_workspace(&workspace, dimension,
                algorithm) < 0) {
        return NULL;
    }
    result = pyfastcd_element_allocate(left->algebra, dimension);
    if (result == NULL) {
        pyfastcd_workspace_release(&workspace);
        return NULL;
    }
    status = fastcd_mul_with_algorithm_f64(result->coefficients,
            left->coefficients, right->coefficients, dimension, algorithm,
            &workspace.value);
    pyfastcd_workspace_release(&workspace);
    if (status != FASTCD_SUCCESS) {
        Py_DECREF(result);
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return (PyObject*)result;
}

// Return the index and sign of a standard-basis product
static PyObject* pyfastcd_basis_product_entry(PyObject* Py_UNUSED(module),
        PyObject* args) {
    Py_ssize_t dimension_value;
    Py_ssize_t left_value;
    Py_ssize_t right_value;
    size_t dimension;
    FastCDBasisProduct product;
    FastCDStatus status;

    if (!PyArg_ParseTuple(args, "nnn:_basis_product", &dimension_value,
                &left_value, &right_value)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    if (left_value < 0 || right_value < 0
            || (size_t)left_value >= dimension
            || (size_t)right_value >= dimension) {
        PyErr_SetString(PyExc_IndexError, "basis index is out of range");
        return NULL;
    }
    status = fastcd_basis_product(&product, dimension, (size_t)left_value,
            (size_t)right_value);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return Py_BuildValue("(Ni)", PyLong_FromSize_t(product.index),
            product.sign);
}

// Apply left or right standard-basis multiplication to an element
static PyObject* pyfastcd_basis_action_entry(PyObject* Py_UNUSED(module),
        PyObject* args, bool left_action) {
    PyObject* element_object;
    Py_ssize_t index_value;
    PyFastCDElement* element;
    PyFastCDElement* result;
    size_t dimension;
    FastCDStatus status;

    if (left_action) {
        if (!PyArg_ParseTuple(args, "nO:_basis_lmul", &index_value,
                    &element_object)) {
            return NULL;
        }
    } else if (!PyArg_ParseTuple(args, "On:_basis_rmul", &element_object,
                &index_value)) {
        return NULL;
    }
    if (!PyObject_TypeCheck(element_object, &PyFastCDElementType)) {
        PyErr_SetString(PyExc_TypeError,
                "basis action requires a pyfastcd.Element");
        return NULL;
    }
    element = (PyFastCDElement*)element_object;
    dimension = (size_t)Py_SIZE(element);
    if (index_value < 0 || (size_t)index_value >= dimension) {
        PyErr_SetString(PyExc_IndexError, "basis index is out of range");
        return NULL;
    }
    result = pyfastcd_element_allocate(element->algebra, dimension);
    if (result == NULL) return NULL;
    status = left_action
        ? fastcd_basis_lmul_f64(result->coefficients, (size_t)index_value,
                element->coefficients, dimension)
        : fastcd_basis_rmul_f64(result->coefficients, element->coefficients,
                (size_t)index_value, dimension);
    if (status != FASTCD_SUCCESS) {
        Py_DECREF(result);
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return (PyObject*)result;
}

// Expose left standard-basis multiplication to the Python layer
static PyObject* pyfastcd_basis_lmul_entry(PyObject* module,
        PyObject* args) {
    return pyfastcd_basis_action_entry(module, args, true);
}

// Expose right standard-basis multiplication to the Python layer
static PyObject* pyfastcd_basis_rmul_entry(PyObject* module,
        PyObject* args) {
    return pyfastcd_basis_action_entry(module, args, false);
}

// Apply a left- or right-multiplication operator commutator
static PyObject* pyfastcd_operator_commutator_entry(
        PyObject* Py_UNUSED(module), PyObject* args, bool left_operator) {
    PyObject* a_object;
    PyObject* b_object;
    PyObject* x_object;
    PyFastCDElement* a;
    PyFastCDElement* b;
    PyFastCDElement* x;
    PyFastCDElement* result;
    PyFastCDWorkspace workspace;
    size_t dimension;
    FastCDStatus status;

    if (!PyArg_ParseTuple(args, "OOO:_operator_commutator", &a_object,
                &b_object, &x_object)) {
        return NULL;
    }
    if (!PyObject_TypeCheck(a_object, &PyFastCDElementType)
            || !PyObject_TypeCheck(b_object, &PyFastCDElementType)
            || !PyObject_TypeCheck(x_object, &PyFastCDElementType)) {
        PyErr_SetString(PyExc_TypeError,
                "operator commutator operands must be pyfastcd.Element objects");
        return NULL;
    }
    a = (PyFastCDElement*)a_object;
    b = (PyFastCDElement*)b_object;
    x = (PyFastCDElement*)x_object;
    if (pyfastcd_check_same_dimension(a, b) < 0
            || pyfastcd_check_same_dimension(a, x) < 0) {
        return NULL;
    }
    dimension = (size_t)Py_SIZE(a);
    result = pyfastcd_element_allocate(a->algebra, dimension);
    if (result == NULL) return NULL;
    if (pyfastcd_workspace_for_query(&workspace, dimension,
                left_operator ? fastcd_lop_commutator_workspace_size
                : fastcd_rop_commutator_workspace_size) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    status = left_operator
        ? fastcd_lop_commutator_apply_f64(result->coefficients,
                a->coefficients, b->coefficients, x->coefficients,
                dimension, &workspace.value)
        : fastcd_rop_commutator_apply_f64(result->coefficients,
                a->coefficients, b->coefficients, x->coefficients,
                dimension, &workspace.value);
    pyfastcd_workspace_release(&workspace);
    if (status != FASTCD_SUCCESS) {
        Py_DECREF(result);
        pyfastcd_set_status_error(status);
        return NULL;
    }
    return (PyObject*)result;
}

// Expose a left multiplication-operator commutator to Python
static PyObject* pyfastcd_lop_commutator_entry(PyObject* module,
        PyObject* args) {
    return pyfastcd_operator_commutator_entry(module, args, true);
}

// Expose a right multiplication-operator commutator to Python
static PyObject* pyfastcd_rop_commutator_entry(PyObject* module,
        PyObject* args) {
    return pyfastcd_operator_commutator_entry(module, args, false);
}

/* ---------- Buffer helpers for vectorized operations ---------- */

// Acquire and validate a strided coefficient-array buffer
static int pyfastcd_acquire_element_buffer(PyObject* object,
        Py_buffer* view, size_t dimension, bool writable,
        const char* argument_name) {
    int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES;

    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(object, view, flags) < 0) return -1;
    if (!pyfastcd_native_double_format(view->format)
            || view->itemsize != (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError,
                "%s must have native float64 storage", argument_name);
        return -1;
    }
    if (view->suboffsets != NULL) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError,
                "%s must use direct strided storage", argument_name);
        return -1;
    }
    if (view->ndim < 1 || view->shape == NULL || view->strides == NULL
            || view->shape[view->ndim - 1] != (Py_ssize_t)dimension) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError,
                "%s must have a final axis of length %zu",
                argument_name, dimension);
        return -1;
    }
    if (view->strides[view->ndim - 1]
            != (Py_ssize_t)sizeof(double)
            && !(dimension == 1
                && view->strides[view->ndim - 1] == 0)) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError,
                "%s must be contiguous along its final axis",
                argument_name);
        return -1;
    }
    if (view->buf != NULL
            && (uintptr_t)view->buf % _Alignof(double) != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError,
                "%s must be aligned for float64 access", argument_name);
        return -1;
    }
    for (int axis = 0; axis < view->ndim - 1; axis++) {
        if (view->shape[axis] > 1
                && view->strides[axis]
                    % (Py_ssize_t)_Alignof(double) != 0) {
            PyBuffer_Release(view);
            PyErr_Format(PyExc_ValueError,
                    "%s has an unsupported unaligned stride", argument_name);
            return -1;
        }
    }
    return 0;
}

// Acquire and validate a strided scalar-output buffer
static int pyfastcd_acquire_scalar_buffer(PyObject* object,
        Py_buffer* view, bool writable, const char* argument_name) {
    int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES;

    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(object, view, flags) < 0) return -1;
    if (!pyfastcd_native_double_format(view->format)
            || view->itemsize != (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError,
                "%s must have native float64 storage", argument_name);
        return -1;
    }
    if (view->suboffsets != NULL) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError,
                "%s must use direct strided storage", argument_name);
        return -1;
    }
    if (view->buf != NULL
            && (uintptr_t)view->buf % _Alignof(double) != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError,
                "%s must be aligned for float64 access", argument_name);
        return -1;
    }
    if (view->strides != NULL) {
        for (int axis = 0; axis < view->ndim; axis++) {
            if (view->shape[axis] > 1
                    && view->strides[axis]
                        % (Py_ssize_t)_Alignof(double) != 0) {
                PyBuffer_Release(view);
                PyErr_Format(PyExc_ValueError,
                        "%s has an unsupported unaligned stride",
                        argument_name);
                return -1;
            }
        }
    }
    return 0;
}

// Require two element buffers to have identical shapes
static int pyfastcd_matching_element_buffers(const Py_buffer* left,
        const Py_buffer* right, const char* argument_name) {
    if (left->ndim != right->ndim) {
        PyErr_Format(PyExc_ValueError,
                "%s has the wrong number of axes", argument_name);
        return -1;
    }
    for (int axis = 0; axis < left->ndim; axis++) {
        if (left->shape[axis] != right->shape[axis]) {
            PyErr_Format(PyExc_ValueError,
                    "%s has an incompatible shape", argument_name);
            return -1;
        }
    }
    return 0;
}

// Require a scalar output to match an element buffer's batch shape
static int pyfastcd_matching_scalar_output(const Py_buffer* input,
        const Py_buffer* output) {
    if (output->ndim != input->ndim - 1) {
        PyErr_SetString(PyExc_ValueError,
                "scalar output has the wrong number of axes");
        return -1;
    }
    for (int axis = 0; axis < output->ndim; axis++) {
        if (output->shape[axis] != input->shape[axis]) {
            PyErr_SetString(PyExc_ValueError,
                    "scalar output has an incompatible shape");
            return -1;
        }
    }
    return 0;
}

// Compute the number of coefficient records in an element buffer
static int pyfastcd_batch_count(const Py_buffer* input, size_t* count) {
    size_t result = 1;

    for (int axis = 0; axis < input->ndim - 1; axis++) {
        size_t extent = (size_t)input->shape[axis];
        if (extent != 0 && result > SIZE_MAX / extent) {
            PyErr_SetString(PyExc_OverflowError,
                    "batch element count overflow");
            return -1;
        }
        result *= extent;
    }
    *count = result;
    return 0;
}

// Resolve a flat batch index to its record address in a strided buffer
static char* pyfastcd_record_pointer(const Py_buffer* view, size_t flat_index,
        int batch_ndim, const Py_ssize_t* batch_shape) {
    Py_ssize_t offset = 0;
    size_t remaining = flat_index;

    for (int axis = batch_ndim - 1; axis >= 0; axis--) {
        size_t extent = (size_t)batch_shape[axis];
        size_t index = extent == 0 ? 0 : remaining % extent;
        if (extent != 0) remaining /= extent;
        offset += (Py_ssize_t)index * view->strides[axis];
    }
    return (char*)view->buf + offset;
}

// Report whether every batch axis broadcasts one coefficient record
static bool pyfastcd_whole_buffer_broadcast(const Py_buffer* view) {
    for (int axis = 0; axis < view->ndim - 1; axis++) {
        if (view->strides[axis] != 0) return false;
    }
    return true;
}

// Return the inclusive address envelope touched by a strided buffer
static int pyfastcd_buffer_span(const Py_buffer* view,
        PyFastCDBufferSpan* span, const char* argument_name) {
    uintmax_t negative_extent = 0;
    uintmax_t positive_extent = 0;
    uintptr_t base;

    if (view == NULL || span == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "internal null buffer span");
        return -1;
    }
    if (view->ndim < 0 || view->itemsize <= 0
            || (view->ndim > 0
                && (view->shape == NULL || view->strides == NULL))) {
        PyErr_Format(PyExc_ValueError,
                "%s has invalid buffer metadata", argument_name);
        return -1;
    }
    span->empty = false;
    for (int axis = 0; axis < view->ndim; axis++) {
        Py_ssize_t extent = view->shape[axis];
        Py_ssize_t stride = view->strides[axis];
        uintmax_t magnitude;
        uintmax_t contribution;
        uintmax_t* total;

        if (extent < 0) {
            PyErr_Format(PyExc_ValueError,
                    "%s has a negative shape extent", argument_name);
            return -1;
        }
        if (extent == 0) {
            span->empty = true;
            return 0;
        }
        magnitude = stride < 0
            ? (uintmax_t)(-(stride + 1)) + UINTMAX_C(1)
            : (uintmax_t)stride;
        if (magnitude != 0
                && (uintmax_t)(extent - 1) > UINTMAX_MAX / magnitude) {
            PyErr_Format(PyExc_OverflowError,
                    "%s buffer span is too large", argument_name);
            return -1;
        }
        contribution = (uintmax_t)(extent - 1) * magnitude;
        total = stride < 0 ? &negative_extent : &positive_extent;
        if (*total > UINTMAX_MAX - contribution) {
            PyErr_Format(PyExc_OverflowError,
                    "%s buffer span is too large", argument_name);
            return -1;
        }
        *total += contribution;
    }
    if (negative_extent > (uintmax_t)PY_SSIZE_T_MAX
            || positive_extent > (uintmax_t)PY_SSIZE_T_MAX) {
        PyErr_Format(PyExc_OverflowError,
                "%s buffer displacement exceeds Python index limits",
                argument_name);
        return -1;
    }
    if (view->buf == NULL) {
        PyErr_Format(PyExc_ValueError,
                "%s has null storage for a nonempty buffer", argument_name);
        return -1;
    }
    base = (uintptr_t)view->buf;
    if (negative_extent > (uintmax_t)base
            || positive_extent > (uintmax_t)(UINTPTR_MAX - base)
            || (uintmax_t)(view->itemsize - 1)
                > (uintmax_t)(UINTPTR_MAX - base) - positive_extent) {
        PyErr_Format(PyExc_OverflowError,
                "%s buffer address span overflow", argument_name);
        return -1;
    }
    span->first = base - (uintptr_t)negative_extent;
    span->last = base + (uintptr_t)positive_extent
        + (uintptr_t)(view->itemsize - 1);
    return 0;
}

// Return one contiguous record interval within a validated buffer span
static int pyfastcd_record_interval(const Py_buffer* view, size_t flat_index,
        int batch_ndim, size_t record_bytes,
        PyFastCDBufferInterval* interval, const char* argument_name) {
    uintmax_t negative_offset = 0;
    uintmax_t positive_offset = 0;
    size_t remaining = flat_index;
    uintptr_t base = (uintptr_t)view->buf;
    uintmax_t delta;

    for (int axis = batch_ndim - 1; axis >= 0; axis--) {
        size_t extent = (size_t)view->shape[axis];
        size_t index = extent == 0 ? 0 : remaining % extent;
        Py_ssize_t stride = view->strides[axis];
        uintmax_t magnitude = stride < 0
            ? (uintmax_t)(-(stride + 1)) + UINTMAX_C(1)
            : (uintmax_t)stride;
        uintmax_t contribution;
        uintmax_t* total;

        if (extent != 0) remaining /= extent;
        if (magnitude != 0 && (uintmax_t)index > UINTMAX_MAX / magnitude) {
            PyErr_Format(PyExc_OverflowError,
                    "%s record address overflow", argument_name);
            return -1;
        }
        contribution = (uintmax_t)index * magnitude;
        total = stride < 0 ? &negative_offset : &positive_offset;
        if (*total > UINTMAX_MAX - contribution) {
            PyErr_Format(PyExc_OverflowError,
                    "%s record address overflow", argument_name);
            return -1;
        }
        *total += contribution;
    }
    if (positive_offset >= negative_offset) {
        delta = positive_offset - negative_offset;
        if (delta > (uintmax_t)(UINTPTR_MAX - base)) {
            PyErr_Format(PyExc_OverflowError,
                    "%s record address overflow", argument_name);
            return -1;
        }
        interval->first = base + (uintptr_t)delta;
    } else {
        delta = negative_offset - positive_offset;
        if (delta > (uintmax_t)base) {
            PyErr_Format(PyExc_OverflowError,
                    "%s record address overflow", argument_name);
            return -1;
        }
        interval->first = base - (uintptr_t)delta;
    }
    if (record_bytes == 0
            || record_bytes - 1 > UINTPTR_MAX - interval->first) {
        PyErr_Format(PyExc_OverflowError,
                "%s record extent overflow", argument_name);
        return -1;
    }
    interval->last = interval->first + record_bytes - 1;
    return 0;
}

// Order touched memory intervals by their first and last addresses
static int pyfastcd_interval_compare(const void* left_object,
        const void* right_object) {
    const PyFastCDBufferInterval* left = left_object;
    const PyFastCDBufferInterval* right = right_object;

    if (left->first < right->first) return -1;
    if (left->first > right->first) return 1;
    if (left->last < right->last) return -1;
    if (left->last > right->last) return 1;
    return 0;
}

// Test records against one continuous address span
static int pyfastcd_records_overlap_span(const Py_buffer* view, size_t count,
        int batch_ndim, size_t record_bytes,
        const PyFastCDBufferSpan* span, const char* argument_name,
        bool* overlap) {
    for (size_t record = 0; record < count; record++) {
        PyFastCDBufferInterval interval;

        if (pyfastcd_record_interval(view, record, batch_ndim, record_bytes,
                    &interval, argument_name) < 0) {
            return -1;
        }
        if (interval.first <= span->last && span->first <= interval.last) {
            *overlap = true;
            return 0;
        }
    }
    *overlap = false;
    return 0;
}

// Test the exact touched record intervals after intersecting envelopes
static int pyfastcd_exact_buffer_overlap(const Py_buffer* output,
        bool scalar_output, const Py_buffer* input, size_t count,
        size_t dimension, const char* argument_name, bool* overlap) {
    PyFastCDBufferSpan output_span;
    PyFastCDBufferSpan input_span;
    PyFastCDBufferInterval* intervals;
    size_t element_bytes;
    size_t output_record_bytes;
    size_t merged_count;
    bool output_contiguous;
    bool input_contiguous;

    if (dimension > SIZE_MAX / sizeof(double)) {
        PyErr_SetString(PyExc_OverflowError, "element byte extent overflow");
        return -1;
    }
    element_bytes = dimension * sizeof(double);
    output_record_bytes = scalar_output ? sizeof(double) : element_bytes;
    if (pyfastcd_buffer_span(output, &output_span, "output") < 0
            || pyfastcd_buffer_span(input, &input_span, argument_name) < 0) {
        return -1;
    }
    if (output_span.empty || input_span.empty
            || output_span.first > input_span.last
            || input_span.first > output_span.last) {
        *overlap = false;
        return 0;
    }
    output_contiguous = PyBuffer_IsContiguous(output, 'C')
        || PyBuffer_IsContiguous(output, 'F');
    input_contiguous = PyBuffer_IsContiguous(input, 'C')
        || PyBuffer_IsContiguous(input, 'F');
    if (output_contiguous && input_contiguous) {
        *overlap = true;
        return 0;
    }
    if (output_contiguous) {
        return pyfastcd_records_overlap_span(input, count, input->ndim - 1,
                element_bytes, &output_span, argument_name, overlap);
    }
    if (input_contiguous) {
        return pyfastcd_records_overlap_span(output, count,
                scalar_output ? output->ndim : output->ndim - 1,
                output_record_bytes, &input_span, "output", overlap);
    }
    if (count > SIZE_MAX / sizeof(*intervals)) {
        PyErr_SetString(PyExc_OverflowError,
                "buffer overlap interval count overflow");
        return -1;
    }
    intervals = PyMem_Malloc(count * sizeof(*intervals));
    if (intervals == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    for (size_t record = 0; record < count; record++) {
        if (pyfastcd_record_interval(input, record, input->ndim - 1,
                    element_bytes, &intervals[record], argument_name) < 0) {
            PyMem_Free(intervals);
            return -1;
        }
    }
    qsort(intervals, count, sizeof(*intervals), pyfastcd_interval_compare);
    merged_count = 0;
    for (size_t record = 0; record < count; record++) {
        if (merged_count == 0
                || intervals[record].first
                    > intervals[merged_count - 1].last) {
            intervals[merged_count++] = intervals[record];
        } else if (intervals[record].last
                > intervals[merged_count - 1].last) {
            intervals[merged_count - 1].last = intervals[record].last;
        }
    }
    for (size_t record = 0; record < count; record++) {
        PyFastCDBufferInterval candidate;
        size_t lower = 0;
        size_t upper = merged_count;

        if (pyfastcd_record_interval(output, record,
                    scalar_output ? output->ndim : output->ndim - 1,
                    output_record_bytes, &candidate, "output") < 0) {
            PyMem_Free(intervals);
            return -1;
        }
        while (lower < upper) {
            size_t middle = lower + (upper - lower) / 2;
            if (intervals[middle].first <= candidate.last) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        if (lower > 0 && intervals[lower - 1].last >= candidate.first) {
            PyMem_Free(intervals);
            *overlap = true;
            return 0;
        }
    }
    PyMem_Free(intervals);
    *overlap = false;
    return 0;
}

// Reject exact overlap before an output can overwrite a later input record
static int pyfastcd_require_disjoint_output(const Py_buffer* output,
        bool scalar_output, const Py_buffer* input, size_t count,
        size_t dimension, const char* argument_name) {
    bool overlap;

    if (pyfastcd_exact_buffer_overlap(output, scalar_output, input, count,
                dimension, argument_name, &overlap) < 0) {
        return -1;
    }
    if (overlap) {
        PyErr_Format(PyExc_ValueError,
                "output must not overlap %s", argument_name);
        return -1;
    }
    return 0;
}

// Allocate the workspace required by a vectorized unary operation
static int pyfastcd_unary_workspace(size_t dimension,
        PyFastCDUnaryOperation operation, PyFastCDWorkspace* workspace) {
    (void)dimension;
    (void)operation;
    return pyfastcd_workspace_allocate(workspace, 0, false);
}

// Allocate the workspace required by a vectorized binary operation
static int pyfastcd_binary_workspace(size_t dimension,
        PyFastCDBinaryOperation operation, PyFastCDWorkspace* workspace) {
    switch (operation) {
        case PYFASTCD_BINARY_MULTIPLY:
            return pyfastcd_workspace_for_query(workspace, dimension,
                    fastcd_mul_workspace_size);
        case PYFASTCD_BINARY_IMAG_CROSS:
        case PYFASTCD_BINARY_COMMUTATOR:
            return pyfastcd_workspace_for_query(workspace, dimension,
                    fastcd_imag_cross_workspace_size);
        case PYFASTCD_BINARY_ANTICOMMUTATOR:
            return pyfastcd_workspace_allocate(workspace, 0, false);
        default:
            PyErr_SetString(PyExc_ValueError,
                    "unknown vectorized binary operation");
            return -1;
    }
}

// Apply a selected unary kernel to one coefficient record
static FastCDStatus pyfastcd_apply_unary_record(double* out,
        const double* x, size_t dimension,
        PyFastCDUnaryOperation operation,
        const FastCDWorkspaceF64* workspace) {
    (void)workspace;
    switch (operation) {
        case PYFASTCD_UNARY_CONJUGATE:
            return fastcd_conjugate_f64(out, x, dimension);
        case PYFASTCD_UNARY_IMAG_PART:
            return fastcd_imag_part_f64(out, x, dimension);
        case PYFASTCD_UNARY_NORMALIZE:
            return fastcd_normalize_f64(out, x, dimension);
        case PYFASTCD_UNARY_RECIPROCAL:
            return fastcd_reciprocal_f64(out, x, dimension);
        case PYFASTCD_UNARY_SQUARE:
            return fastcd_square_f64(out, x, dimension);
        default:
            return FASTCD_INVALID_DIMENSION;
    }
}

// Apply a selected binary kernel to one pair of coefficient records
static FastCDStatus pyfastcd_apply_binary_record(double* out,
        const double* x, const double* y, size_t dimension,
        PyFastCDBinaryOperation operation,
        const FastCDWorkspaceF64* workspace) {
    switch (operation) {
        case PYFASTCD_BINARY_MULTIPLY:
            return fastcd_mul_f64(out, x, y, dimension, workspace);
        case PYFASTCD_BINARY_IMAG_CROSS:
            return fastcd_imag_cross_f64(out, x, y, dimension, workspace);
        case PYFASTCD_BINARY_COMMUTATOR:
            return fastcd_commutator_f64(out, x, y, dimension, workspace);
        case PYFASTCD_BINARY_ANTICOMMUTATOR:
            return fastcd_anticommutator_f64(out, x, y, dimension);
        default:
            return FASTCD_INVALID_DIMENSION;
    }
}

// Apply a unary kernel across validated strided coefficient buffers
static PyObject* pyfastcd_apply_unary_into(PyObject* Py_UNUSED(module),
        PyObject* args) {
    int operation_value;
    PyObject* out_object;
    PyObject* x_object;
    Py_ssize_t dimension_value;
    size_t dimension;
    size_t count;
    Py_buffer out = {0};
    Py_buffer x = {0};
    PyFastCDWorkspace workspace;
    FastCDStatus status = FASTCD_SUCCESS;

    if (!PyArg_ParseTuple(args, "iOOn:_apply_unary_into",
                &operation_value, &out_object, &x_object,
                &dimension_value)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    if (pyfastcd_acquire_element_buffer(out_object, &out, dimension, true,
                "output") < 0) {
        return NULL;
    }
    if (pyfastcd_acquire_element_buffer(x_object, &x, dimension, false,
                "input") < 0) {
        PyBuffer_Release(&out);
        return NULL;
    }
    if (pyfastcd_matching_element_buffers(&out, &x, "input") < 0
            || pyfastcd_batch_count(&x, &count) < 0
            || pyfastcd_require_disjoint_output(&out, false, &x, count,
                dimension, "input") < 0
            || pyfastcd_unary_workspace(dimension,
                (PyFastCDUnaryOperation)operation_value, &workspace) < 0) {
        PyBuffer_Release(&x);
        PyBuffer_Release(&out);
        return NULL;
    }

#define PYFASTCD_RUN_UNARY_LOOP() \
    do { \
        for (size_t record = 0; record < count; record++) { \
            double* out_pointer = (double*)pyfastcd_record_pointer(&out, \
                    record, x.ndim - 1, x.shape); \
            const double* x_pointer = (const double*) \
                pyfastcd_record_pointer(&x, record, x.ndim - 1, x.shape); \
            status = pyfastcd_apply_unary_record(out_pointer, x_pointer, \
                    dimension, (PyFastCDUnaryOperation)operation_value, \
                    &workspace.value); \
            if (status != FASTCD_SUCCESS) break; \
        } \
    } while (0)

    if (count >= PYFASTCD_GIL_RELEASE_COUNT) {
        Py_BEGIN_ALLOW_THREADS
        PYFASTCD_RUN_UNARY_LOOP();
        Py_END_ALLOW_THREADS
    } else {
        PYFASTCD_RUN_UNARY_LOOP();
    }
#undef PYFASTCD_RUN_UNARY_LOOP

    pyfastcd_workspace_release(&workspace);
    PyBuffer_Release(&x);
    PyBuffer_Release(&out);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    Py_RETURN_NONE;
}

// Validate and execute every coefficient-array multiplication consistently
static PyObject* pyfastcd_multiply_into(PyObject* out_object,
        PyObject* x_object, PyObject* y_object, size_t dimension,
        FastCDAlgorithm algorithm) {
    size_t count;
    Py_buffer out = {0};
    Py_buffer x = {0};
    Py_buffer y = {0};
    PyFastCDWorkspace workspace;
    FastCDStatus status = FASTCD_SUCCESS;

    if (pyfastcd_acquire_element_buffer(out_object, &out, dimension, true,
                "output") < 0) {
        return NULL;
    }
    if (pyfastcd_acquire_element_buffer(x_object, &x, dimension, false,
                "left input") < 0) {
        PyBuffer_Release(&out);
        return NULL;
    }
    if (pyfastcd_acquire_element_buffer(y_object, &y, dimension, false,
                "right input") < 0) {
        PyBuffer_Release(&x);
        PyBuffer_Release(&out);
        return NULL;
    }
    if (pyfastcd_matching_element_buffers(&out, &x, "left input") < 0
            || pyfastcd_matching_element_buffers(&out, &y,
                "right input") < 0
            || pyfastcd_batch_count(&x, &count) < 0
            || pyfastcd_require_disjoint_output(&out, false, &x, count,
                dimension, "left input") < 0
            || pyfastcd_require_disjoint_output(&out, false, &y, count,
                dimension, "right input") < 0
            || pyfastcd_require_algorithm_support(algorithm, dimension) < 0
            || pyfastcd_algorithm_workspace(&workspace, dimension,
                algorithm) < 0) {
        PyBuffer_Release(&y);
        PyBuffer_Release(&x);
        PyBuffer_Release(&out);
        return NULL;
    }
    /* Batches of two or more records use the batch selection on every
     * layout, so results never depend on strides.  The automatic workspace
     * query already covers this choice. */
    if (algorithm == FASTCD_ALGORITHM_AUTO && count >= 2) {
        (void)fastcd_default_batch_algorithm(dimension, &algorithm);
    }

    if (PyBuffer_IsContiguous(&out, 'C')
            && (PyBuffer_IsContiguous(&x, 'C')
                || pyfastcd_whole_buffer_broadcast(&x))
            && (PyBuffer_IsContiguous(&y, 'C')
                || pyfastcd_whole_buffer_broadcast(&y))) {
        size_t x_stride = pyfastcd_whole_buffer_broadcast(&x)
            ? 0 : dimension;
        size_t y_stride = pyfastcd_whole_buffer_broadcast(&y)
            ? 0 : dimension;

        if (count >= PYFASTCD_GIL_RELEASE_COUNT) {
            Py_BEGIN_ALLOW_THREADS
            status = fastcd_mul_batch_with_algorithm_f64((double*)out.buf,
                    dimension, (const double*)x.buf, x_stride,
                    (const double*)y.buf, y_stride, count, dimension,
                    algorithm, &workspace.value);
            Py_END_ALLOW_THREADS
        } else {
            status = fastcd_mul_batch_with_algorithm_f64((double*)out.buf,
                    dimension, (const double*)x.buf, x_stride,
                    (const double*)y.buf, y_stride, count, dimension,
                    algorithm, &workspace.value);
        }
    } else {
#define PYFASTCD_RUN_MULTIPLY_LOOP() \
        do { \
            for (size_t record = 0; record < count; record++) { \
                double* out_pointer = (double*) \
                    pyfastcd_record_pointer(&out, record, x.ndim - 1, \
                            x.shape); \
                const double* x_pointer = (const double*) \
                    pyfastcd_record_pointer(&x, record, x.ndim - 1, \
                            x.shape); \
                const double* y_pointer = (const double*) \
                    pyfastcd_record_pointer(&y, record, x.ndim - 1, \
                            x.shape); \
                status = fastcd_mul_with_algorithm_f64(out_pointer, \
                        x_pointer, y_pointer, dimension, algorithm, \
                        &workspace.value); \
                if (status != FASTCD_SUCCESS) break; \
            } \
        } while (0)

        if (count >= PYFASTCD_GIL_RELEASE_COUNT) {
            Py_BEGIN_ALLOW_THREADS
            PYFASTCD_RUN_MULTIPLY_LOOP();
            Py_END_ALLOW_THREADS
        } else {
            PYFASTCD_RUN_MULTIPLY_LOOP();
        }
#undef PYFASTCD_RUN_MULTIPLY_LOOP
    }

    pyfastcd_workspace_release(&workspace);
    PyBuffer_Release(&y);
    PyBuffer_Release(&x);
    PyBuffer_Release(&out);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    Py_RETURN_NONE;
}

// Apply a binary kernel across validated strided coefficient buffers
static PyObject* pyfastcd_apply_binary_into(PyObject* Py_UNUSED(module),
        PyObject* args) {
    int operation_value;
    PyObject* out_object;
    PyObject* x_object;
    PyObject* y_object;
    Py_ssize_t dimension_value;
    size_t dimension;
    size_t count;
    Py_buffer out = {0};
    Py_buffer x = {0};
    Py_buffer y = {0};
    PyFastCDWorkspace workspace;
    FastCDStatus status = FASTCD_SUCCESS;

    if (!PyArg_ParseTuple(args, "iOOOn:_apply_binary_into",
                &operation_value, &out_object, &x_object, &y_object,
                &dimension_value)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    if ((PyFastCDBinaryOperation)operation_value
            == PYFASTCD_BINARY_MULTIPLY) {
        return pyfastcd_multiply_into(out_object, x_object, y_object,
                dimension, FASTCD_ALGORITHM_AUTO);
    }
    if (pyfastcd_acquire_element_buffer(out_object, &out, dimension, true,
                "output") < 0) {
        return NULL;
    }
    if (pyfastcd_acquire_element_buffer(x_object, &x, dimension, false,
                "left input") < 0) {
        PyBuffer_Release(&out);
        return NULL;
    }
    if (pyfastcd_acquire_element_buffer(y_object, &y, dimension, false,
                "right input") < 0) {
        PyBuffer_Release(&x);
        PyBuffer_Release(&out);
        return NULL;
    }
    if (pyfastcd_matching_element_buffers(&out, &x, "left input") < 0
            || pyfastcd_matching_element_buffers(&out, &y,
                "right input") < 0
            || pyfastcd_batch_count(&x, &count) < 0
            || pyfastcd_require_disjoint_output(&out, false, &x, count,
                dimension, "left input") < 0
            || pyfastcd_require_disjoint_output(&out, false, &y, count,
                dimension, "right input") < 0
            || pyfastcd_binary_workspace(dimension,
                (PyFastCDBinaryOperation)operation_value, &workspace) < 0) {
        PyBuffer_Release(&y);
        PyBuffer_Release(&x);
        PyBuffer_Release(&out);
        return NULL;
    }

#define PYFASTCD_RUN_BINARY_LOOP() \
        do { \
            for (size_t record = 0; record < count; record++) { \
                double* out_pointer = (double*) \
                    pyfastcd_record_pointer(&out, record, x.ndim - 1, \
                            x.shape); \
                const double* x_pointer = (const double*) \
                    pyfastcd_record_pointer(&x, record, x.ndim - 1, \
                            x.shape); \
                const double* y_pointer = (const double*) \
                    pyfastcd_record_pointer(&y, record, x.ndim - 1, \
                            x.shape); \
                status = pyfastcd_apply_binary_record(out_pointer, x_pointer, \
                        y_pointer, dimension, \
                        (PyFastCDBinaryOperation)operation_value, \
                        &workspace.value); \
                if (status != FASTCD_SUCCESS) break; \
            } \
        } while (0)

        if (count >= PYFASTCD_GIL_RELEASE_COUNT) {
            Py_BEGIN_ALLOW_THREADS
            PYFASTCD_RUN_BINARY_LOOP();
            Py_END_ALLOW_THREADS
        } else {
            PYFASTCD_RUN_BINARY_LOOP();
        }
#undef PYFASTCD_RUN_BINARY_LOOP

    pyfastcd_workspace_release(&workspace);
    PyBuffer_Release(&y);
    PyBuffer_Release(&x);
    PyBuffer_Release(&out);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    Py_RETURN_NONE;
}

// Apply an explicitly selected multiplication algorithm to coefficient buffers
static PyObject* pyfastcd_apply_multiply_into(PyObject* Py_UNUSED(module),
        PyObject* args) {
    int algorithm_code;
    PyObject* out_object;
    PyObject* x_object;
    PyObject* y_object;
    Py_ssize_t dimension_value;
    size_t dimension;
    FastCDAlgorithm algorithm;

    if (!PyArg_ParseTuple(args, "iOOOn:_apply_multiply_into",
                &algorithm_code, &out_object, &x_object, &y_object,
                &dimension_value)) {
        return NULL;
    }
    if (pyfastcd_algorithm_from_code(algorithm_code, &algorithm) < 0
            || pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    return pyfastcd_multiply_into(out_object, x_object, y_object, dimension,
            algorithm);
}

// Apply a scalar-valued kernel across validated coefficient buffers
static PyObject* pyfastcd_apply_scalar_into(PyObject* Py_UNUSED(module),
        PyObject* args) {
    int operation_value;
    PyObject* out_object;
    PyObject* x_object;
    PyObject* y_object = Py_None;
    Py_ssize_t dimension_value;
    size_t dimension;
    size_t count;
    Py_buffer out = {0};
    Py_buffer x = {0};
    Py_buffer y = {0};
    bool has_y;
    FastCDStatus status = FASTCD_SUCCESS;

    if (!PyArg_ParseTuple(args, "iOOOn:_apply_scalar_into",
                &operation_value, &out_object, &x_object, &y_object,
                &dimension_value)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    has_y = (PyFastCDScalarOperation)operation_value == PYFASTCD_SCALAR_INNER;
    if (pyfastcd_acquire_scalar_buffer(out_object, &out, true,
                "output") < 0) {
        return NULL;
    }
    if (pyfastcd_acquire_element_buffer(x_object, &x, dimension, false,
                "left input") < 0) {
        PyBuffer_Release(&out);
        return NULL;
    }
    if (has_y && pyfastcd_acquire_element_buffer(y_object, &y, dimension,
                false, "right input") < 0) {
        PyBuffer_Release(&x);
        PyBuffer_Release(&out);
        return NULL;
    }
    if (pyfastcd_matching_scalar_output(&x, &out) < 0
            || (has_y && pyfastcd_matching_element_buffers(&x, &y,
                "right input") < 0)
            || pyfastcd_batch_count(&x, &count) < 0
            || pyfastcd_require_disjoint_output(&out, true, &x, count,
                dimension, "left input") < 0
            || (has_y && pyfastcd_require_disjoint_output(&out, true, &y,
                count, dimension, "right input") < 0)) {
        if (has_y) PyBuffer_Release(&y);
        PyBuffer_Release(&x);
        PyBuffer_Release(&out);
        return NULL;
    }

#define PYFASTCD_RUN_SCALAR_LOOP() \
    do { \
        for (size_t record = 0; record < count; record++) { \
            double* out_pointer = (double*)pyfastcd_record_pointer(&out, \
                    record, x.ndim - 1, x.shape); \
            const double* x_pointer = (const double*) \
                pyfastcd_record_pointer(&x, record, x.ndim - 1, x.shape); \
            if ((PyFastCDScalarOperation)operation_value \
                    == PYFASTCD_SCALAR_INNER) { \
                const double* y_pointer = (const double*) \
                    pyfastcd_record_pointer(&y, record, x.ndim - 1, \
                            x.shape); \
                status = fastcd_inner_f64(out_pointer, x_pointer, y_pointer, \
                        dimension); \
            } else if ((PyFastCDScalarOperation)operation_value \
                    == PYFASTCD_SCALAR_NORM_SQUARED) { \
                status = fastcd_norm_squared_f64(out_pointer, x_pointer, \
                        dimension); \
            } else if ((PyFastCDScalarOperation)operation_value \
                    == PYFASTCD_SCALAR_NORM) { \
                status = fastcd_norm_f64(out_pointer, x_pointer, dimension); \
            } else { \
                status = FASTCD_INVALID_DIMENSION; \
            } \
            if (status != FASTCD_SUCCESS) break; \
        } \
    } while (0)

    if (count >= PYFASTCD_GIL_RELEASE_COUNT) {
        Py_BEGIN_ALLOW_THREADS
        PYFASTCD_RUN_SCALAR_LOOP();
        Py_END_ALLOW_THREADS
    } else {
        PYFASTCD_RUN_SCALAR_LOOP();
    }
#undef PYFASTCD_RUN_SCALAR_LOOP

    if (has_y) PyBuffer_Release(&y);
    PyBuffer_Release(&x);
    PyBuffer_Release(&out);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    Py_RETURN_NONE;
}

// Apply the associator across validated coefficient buffers
static PyObject* pyfastcd_apply_associator_into(
        PyObject* Py_UNUSED(module), PyObject* args) {
    PyObject* out_object;
    PyObject* x_object;
    PyObject* y_object;
    PyObject* z_object;
    Py_ssize_t dimension_value;
    size_t dimension;
    size_t count;
    Py_buffer out = {0};
    Py_buffer x = {0};
    Py_buffer y = {0};
    Py_buffer z = {0};
    PyFastCDWorkspace workspace;
    FastCDStatus status = FASTCD_SUCCESS;

    if (!PyArg_ParseTuple(args, "OOOOn:_apply_associator_into", &out_object,
                &x_object, &y_object, &z_object, &dimension_value)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    if (pyfastcd_acquire_element_buffer(out_object, &out, dimension, true,
                "output") < 0) return NULL;
    if (pyfastcd_acquire_element_buffer(x_object, &x, dimension, false,
                "first input") < 0) goto associator_release_out;
    if (pyfastcd_acquire_element_buffer(y_object, &y, dimension, false,
                "second input") < 0) goto associator_release_x;
    if (pyfastcd_acquire_element_buffer(z_object, &z, dimension, false,
                "third input") < 0) goto associator_release_y;
    if (pyfastcd_matching_element_buffers(&out, &x, "first input") < 0
            || pyfastcd_matching_element_buffers(&out, &y,
                "second input") < 0
            || pyfastcd_matching_element_buffers(&out, &z,
                "third input") < 0
            || pyfastcd_batch_count(&x, &count) < 0
            || pyfastcd_require_disjoint_output(&out, false, &x, count,
                dimension, "first input") < 0
            || pyfastcd_require_disjoint_output(&out, false, &y, count,
                dimension, "second input") < 0
            || pyfastcd_require_disjoint_output(&out, false, &z, count,
                dimension, "third input") < 0
            || pyfastcd_workspace_for_query(&workspace, dimension,
                fastcd_associator_workspace_size) < 0) {
        goto associator_release_z;
    }

#define PYFASTCD_RUN_ASSOCIATOR_LOOP() \
    do { \
        for (size_t record = 0; record < count; record++) { \
            double* out_pointer = (double*)pyfastcd_record_pointer(&out, \
                    record, x.ndim - 1, x.shape); \
            const double* x_pointer = (const double*) \
                pyfastcd_record_pointer(&x, record, x.ndim - 1, x.shape); \
            const double* y_pointer = (const double*) \
                pyfastcd_record_pointer(&y, record, x.ndim - 1, x.shape); \
            const double* z_pointer = (const double*) \
                pyfastcd_record_pointer(&z, record, x.ndim - 1, x.shape); \
            status = fastcd_associator_f64(out_pointer, x_pointer, y_pointer, \
                    z_pointer, dimension, &workspace.value); \
            if (status != FASTCD_SUCCESS) break; \
        } \
    } while (0)

    if (count >= PYFASTCD_GIL_RELEASE_COUNT) {
        Py_BEGIN_ALLOW_THREADS
        PYFASTCD_RUN_ASSOCIATOR_LOOP();
        Py_END_ALLOW_THREADS
    } else {
        PYFASTCD_RUN_ASSOCIATOR_LOOP();
    }
#undef PYFASTCD_RUN_ASSOCIATOR_LOOP

    pyfastcd_workspace_release(&workspace);
    PyBuffer_Release(&z);
    PyBuffer_Release(&y);
    PyBuffer_Release(&x);
    PyBuffer_Release(&out);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    Py_RETURN_NONE;

associator_release_z:
    PyBuffer_Release(&z);
associator_release_y:
    PyBuffer_Release(&y);
associator_release_x:
    PyBuffer_Release(&x);
associator_release_out:
    PyBuffer_Release(&out);
    return NULL;
}

// Apply left or right basis multiplication across coefficient buffers
static PyObject* pyfastcd_apply_basis_into(PyObject* Py_UNUSED(module),
        PyObject* args) {
    int left_action;
    Py_ssize_t index_value;
    PyObject* out_object;
    PyObject* x_object;
    Py_ssize_t dimension_value;
    size_t dimension;
    size_t count;
    Py_buffer out = {0};
    Py_buffer x = {0};
    FastCDStatus status = FASTCD_SUCCESS;

    if (!PyArg_ParseTuple(args, "pnOOn:_apply_basis_into", &left_action,
                &index_value, &out_object, &x_object, &dimension_value)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    if (index_value < 0 || (size_t)index_value >= dimension) {
        PyErr_SetString(PyExc_IndexError, "basis index is out of range");
        return NULL;
    }
    if (pyfastcd_acquire_element_buffer(out_object, &out, dimension, true,
                "output") < 0) return NULL;
    if (pyfastcd_acquire_element_buffer(x_object, &x, dimension, false,
                "input") < 0) {
        PyBuffer_Release(&out);
        return NULL;
    }
    if (pyfastcd_matching_element_buffers(&out, &x, "input") < 0
            || pyfastcd_batch_count(&x, &count) < 0
            || pyfastcd_require_disjoint_output(&out, false, &x, count,
                dimension, "input") < 0) {
        PyBuffer_Release(&x);
        PyBuffer_Release(&out);
        return NULL;
    }

#define PYFASTCD_RUN_BASIS_LOOP() \
    do { \
        for (size_t record = 0; record < count; record++) { \
            double* out_pointer = (double*)pyfastcd_record_pointer(&out, \
                    record, x.ndim - 1, x.shape); \
            const double* x_pointer = (const double*) \
                pyfastcd_record_pointer(&x, record, x.ndim - 1, x.shape); \
            status = left_action \
                ? fastcd_basis_lmul_f64(out_pointer, (size_t)index_value, \
                        x_pointer, dimension) \
                : fastcd_basis_rmul_f64(out_pointer, x_pointer, \
                        (size_t)index_value, dimension); \
            if (status != FASTCD_SUCCESS) break; \
        } \
    } while (0)

    if (count >= PYFASTCD_GIL_RELEASE_COUNT) {
        Py_BEGIN_ALLOW_THREADS
        PYFASTCD_RUN_BASIS_LOOP();
        Py_END_ALLOW_THREADS
    } else {
        PYFASTCD_RUN_BASIS_LOOP();
    }
#undef PYFASTCD_RUN_BASIS_LOOP

    PyBuffer_Release(&x);
    PyBuffer_Release(&out);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    Py_RETURN_NONE;
}

// Apply an operator commutator across validated coefficient buffers
static PyObject* pyfastcd_apply_operator_into(PyObject* Py_UNUSED(module),
        PyObject* args) {
    int left_operator;
    PyObject* out_object;
    PyObject* a_object;
    PyObject* b_object;
    PyObject* x_object;
    Py_ssize_t dimension_value;
    size_t dimension;
    size_t count;
    Py_buffer out = {0};
    Py_buffer a = {0};
    Py_buffer b = {0};
    Py_buffer x = {0};
    PyFastCDWorkspace workspace;
    FastCDStatus status = FASTCD_SUCCESS;

    if (!PyArg_ParseTuple(args, "pOOOOn:_apply_operator_into",
                &left_operator, &out_object, &a_object, &b_object, &x_object,
                &dimension_value)) {
        return NULL;
    }
    if (pyfastcd_dimension_from_ssize(dimension_value, &dimension) < 0) {
        return NULL;
    }
    if (pyfastcd_acquire_element_buffer(out_object, &out, dimension, true,
                "output") < 0) return NULL;
    if (pyfastcd_acquire_element_buffer(a_object, &a, dimension, false,
                "a") < 0) goto operator_release_out;
    if (pyfastcd_acquire_element_buffer(b_object, &b, dimension, false,
                "b") < 0) goto operator_release_a;
    if (pyfastcd_acquire_element_buffer(x_object, &x, dimension, false,
                "x") < 0) goto operator_release_b;
    if (pyfastcd_matching_element_buffers(&out, &a, "a") < 0
            || pyfastcd_matching_element_buffers(&out, &b, "b") < 0
            || pyfastcd_matching_element_buffers(&out, &x, "x") < 0
            || pyfastcd_batch_count(&x, &count) < 0
            || pyfastcd_require_disjoint_output(&out, false, &a, count,
                dimension, "a") < 0
            || pyfastcd_require_disjoint_output(&out, false, &b, count,
                dimension, "b") < 0
            || pyfastcd_require_disjoint_output(&out, false, &x, count,
                dimension, "x") < 0
            || pyfastcd_workspace_for_query(&workspace, dimension,
                left_operator ? fastcd_lop_commutator_workspace_size
                : fastcd_rop_commutator_workspace_size) < 0) {
        goto operator_release_x;
    }

#define PYFASTCD_RUN_OPERATOR_LOOP() \
    do { \
        for (size_t record = 0; record < count; record++) { \
            double* out_pointer = (double*)pyfastcd_record_pointer(&out, \
                    record, x.ndim - 1, x.shape); \
            const double* a_pointer = (const double*) \
                pyfastcd_record_pointer(&a, record, x.ndim - 1, x.shape); \
            const double* b_pointer = (const double*) \
                pyfastcd_record_pointer(&b, record, x.ndim - 1, x.shape); \
            const double* x_pointer = (const double*) \
                pyfastcd_record_pointer(&x, record, x.ndim - 1, x.shape); \
            status = left_operator \
                ? fastcd_lop_commutator_apply_f64(out_pointer, a_pointer, \
                        b_pointer, x_pointer, dimension, &workspace.value) \
                : fastcd_rop_commutator_apply_f64(out_pointer, a_pointer, \
                        b_pointer, x_pointer, dimension, &workspace.value); \
            if (status != FASTCD_SUCCESS) break; \
        } \
    } while (0)

    if (count >= PYFASTCD_GIL_RELEASE_COUNT) {
        Py_BEGIN_ALLOW_THREADS
        PYFASTCD_RUN_OPERATOR_LOOP();
        Py_END_ALLOW_THREADS
    } else {
        PYFASTCD_RUN_OPERATOR_LOOP();
    }
#undef PYFASTCD_RUN_OPERATOR_LOOP

    pyfastcd_workspace_release(&workspace);
    PyBuffer_Release(&x);
    PyBuffer_Release(&b);
    PyBuffer_Release(&a);
    PyBuffer_Release(&out);
    if (status != FASTCD_SUCCESS) {
        pyfastcd_set_status_error(status);
        return NULL;
    }
    Py_RETURN_NONE;

operator_release_x:
    PyBuffer_Release(&x);
operator_release_b:
    PyBuffer_Release(&b);
operator_release_a:
    PyBuffer_Release(&a);
operator_release_out:
    PyBuffer_Release(&out);
    return NULL;
}

/* ---------- Type and module definitions ---------- */

static PySequenceMethods pyfastcd_element_sequence = {
    .sq_length = pyfastcd_element_length,
    .sq_item = pyfastcd_element_sequence_item
};

static PyMappingMethods pyfastcd_element_mapping = {
    .mp_length = pyfastcd_element_length,
    .mp_subscript = pyfastcd_element_subscript
};

static PyBufferProcs pyfastcd_element_buffer = {
    .bf_getbuffer = pyfastcd_element_getbuffer,
    .bf_releasebuffer = NULL
};

static PyNumberMethods pyfastcd_element_number = {
    .nb_add = pyfastcd_element_add,
    .nb_subtract = pyfastcd_element_subtract,
    .nb_multiply = pyfastcd_element_multiply,
    .nb_power = pyfastcd_element_power,
    .nb_negative = pyfastcd_element_negative,
    .nb_positive = pyfastcd_element_positive,
    .nb_bool = pyfastcd_element_bool,
    .nb_true_divide = pyfastcd_element_true_divide,
    .nb_absolute = pyfastcd_element_absolute
};

#ifdef Py_TPFLAGS_ITEMS_AT_END
#  define PYFASTCD_ITEMS_AT_END Py_TPFLAGS_ITEMS_AT_END
#else
#  define PYFASTCD_ITEMS_AT_END 0
#endif

PyDoc_STRVAR(pyfastcd_element_type_doc,
"An immutable element of a standard real Cayley--Dickson algebra.\n\n"
"Elements are constructed by calling a :class:`pyfastcd.Algebra`; direct\n"
"``Element(...)`` construction is unsupported.  Coefficients are contiguous\n"
"native ``double`` values stored inline in the same Python allocation.\n\n"
"Raises\n"
"------\n"
"TypeError\n"
"    If this type is constructed directly or subclassed.\n\n"
"Notes\n"
"-----\n"
"Elements are immutable and hashable: equal elements hash equal, and an\n"
"element equal to a real scalar hashes like that scalar.  Every\n"
"element-valued operation,\n"
"including unary plus, conjugation, and powers zero or one, returns a newly\n"
"allocated value.  The read-only Python buffer enables zero-copy\n"
"``np.asarray(x)`` and ``memoryview(x)`` conversion.\n\n"
"Addition and subtraction are coordinatewise.  A real scalar is embedded in\n"
"coordinate zero.  ``x*y`` is the Cayley--Dickson product, while multiplication\n"
"or division by a real scalar scales every coordinate.  ``abs(x)`` is the\n"
"Euclidean coefficient norm.  Element operands must have the same dimension.\n\n"
"Integer powers use a deterministic exponentiation-by-squaring schedule;\n"
"``x**2`` selects the specialized linear-time square and negative exponents\n"
"first use ``x.reciprocal()``.  Scalar Element exponents must fit a signed C\n"
"``long long`` or ``OverflowError`` is raised.  Use ``**`` for powers: Python\n"
"reserves ``^`` for bitwise XOR, and Element does not override it.\n\n"
"All arithmetic uses native ``float64``.  Familiar norm-composition,\n"
"associativity, and cancellation laws do not hold throughout the full tower;\n"
"preserve written parentheses in higher-dimensional calculations.");

static PyTypeObject PyFastCDElementType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyfastcd.Element",
    .tp_basicsize = offsetof(PyFastCDElement, coefficients),
    .tp_itemsize = sizeof(double),
    .tp_dealloc = (destructor)pyfastcd_element_dealloc,
    .tp_repr = pyfastcd_element_repr,
    .tp_as_number = &pyfastcd_element_number,
    .tp_as_sequence = &pyfastcd_element_sequence,
    .tp_as_mapping = &pyfastcd_element_mapping,
    .tp_as_buffer = &pyfastcd_element_buffer,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC
        | PYFASTCD_ITEMS_AT_END,
    .tp_doc = pyfastcd_element_type_doc,
    .tp_richcompare = pyfastcd_element_richcompare,
    .tp_iter = pyfastcd_element_iter,
    .tp_hash = pyfastcd_element_hash,
    .tp_traverse = (traverseproc)pyfastcd_element_traverse,
    .tp_clear = (inquiry)pyfastcd_element_clear,
    .tp_methods = pyfastcd_element_methods,
    .tp_getset = pyfastcd_element_getset,
    .tp_new = pyfastcd_element_new_forbidden
};

_Static_assert(offsetof(PyFastCDElement, coefficients)
        % _Alignof(double) == 0,
        "Element coefficient storage must be aligned for double");

// Return the linked native library version to Python
static PyObject* pyfastcd_version(PyObject* Py_UNUSED(module),
        PyObject* Py_UNUSED(arguments)) {
    return PyUnicode_FromString(fastcd_version_string());
}

static PyMethodDef pyfastcd_module_methods[] = {
    {"_version", pyfastcd_version, METH_NOARGS,
        "Return the linked fastCD library version."},
    {"_element", pyfastcd_element_from_object, METH_VARARGS,
        "Construct an owned Element for an Algebra."},
    {"_scalar", pyfastcd_scalar_element, METH_VARARGS,
        "Construct a scalar Element for an Algebra."},
    {"_basis", pyfastcd_basis_element, METH_VARARGS,
        "Construct a standard-basis Element for an Algebra."},
    {"_algorithm", pyfastcd_algorithm_entry, METH_VARARGS,
        "Map an algorithm name onto its code and check dimension support."},
    {"_default_algorithm", pyfastcd_default_algorithm_entry, METH_VARARGS,
        "Return the name of the algorithm that automatic selection uses."},
    {"_default_batch_algorithm", pyfastcd_default_batch_algorithm_entry,
        METH_VARARGS,
        "Return the name of the algorithm automatic selection uses for batches."},
    {"_multiply_element", pyfastcd_multiply_element_entry, METH_VARARGS,
        "Multiply two Elements with an explicitly selected algorithm."},
    {"_basis_product", pyfastcd_basis_product_entry, METH_VARARGS,
        "Return the index and sign of a standard-basis product."},
    {"_basis_lmul", pyfastcd_basis_lmul_entry, METH_VARARGS,
        "Apply left standard-basis multiplication to an Element."},
    {"_basis_rmul", pyfastcd_basis_rmul_entry, METH_VARARGS,
        "Apply right standard-basis multiplication to an Element."},
    {"_lop_commutator", pyfastcd_lop_commutator_entry, METH_VARARGS,
        "Apply a left multiplication-operator commutator."},
    {"_rop_commutator", pyfastcd_rop_commutator_entry, METH_VARARGS,
        "Apply a right multiplication-operator commutator."},
    {"_apply_unary_into", pyfastcd_apply_unary_into, METH_VARARGS,
        "Apply a unary operation to coefficient buffers."},
    {"_apply_binary_into", pyfastcd_apply_binary_into, METH_VARARGS,
        "Apply a binary operation to coefficient buffers."},
    {"_apply_multiply_into", pyfastcd_apply_multiply_into, METH_VARARGS,
        "Apply a selected multiplication algorithm to coefficient buffers."},
    {"_apply_scalar_into", pyfastcd_apply_scalar_into, METH_VARARGS,
        "Apply a scalar-valued operation to coefficient buffers."},
    {"_apply_associator_into", pyfastcd_apply_associator_into, METH_VARARGS,
        "Apply associators to coefficient buffers."},
    {"_apply_basis_into", pyfastcd_apply_basis_into, METH_VARARGS,
        "Apply standard-basis multiplication to coefficient buffers."},
    {"_apply_operator_into", pyfastcd_apply_operator_into, METH_VARARGS,
        "Apply operator commutators to coefficient buffers."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pyfastcd_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pyfastcd._core",
    .m_doc = "Compiled scalar and buffer kernels for pyfastcd.",
    .m_size = -1,
    .m_methods = pyfastcd_module_methods
};

// Initialize the compiled pyfastcd core module and Element type
PyMODINIT_FUNC PyInit__core(void) {
    PyObject* module;

    if (PyType_Ready(&PyFastCDElementType) < 0) return NULL;
    module = PyModule_Create(&pyfastcd_module);
    if (module == NULL) return NULL;
    Py_INCREF(&PyFastCDElementType);
    if (PyModule_AddObject(module, "Element",
                (PyObject*)&PyFastCDElementType) < 0) {
        Py_DECREF(&PyFastCDElementType);
        Py_DECREF(module);
        return NULL;
    }

#define PYFASTCD_ADD_INT_CONSTANT(name, value) \
    do { \
        if (PyModule_AddIntConstant(module, name, (long)(value)) < 0) { \
            Py_DECREF(module); \
            return NULL; \
        } \
    } while (0)

    PYFASTCD_ADD_INT_CONSTANT("_UNARY_CONJUGATE",
            PYFASTCD_UNARY_CONJUGATE);
    PYFASTCD_ADD_INT_CONSTANT("_UNARY_IMAG_PART",
            PYFASTCD_UNARY_IMAG_PART);
    PYFASTCD_ADD_INT_CONSTANT("_UNARY_NORMALIZE",
            PYFASTCD_UNARY_NORMALIZE);
    PYFASTCD_ADD_INT_CONSTANT("_UNARY_RECIPROCAL",
            PYFASTCD_UNARY_RECIPROCAL);
    PYFASTCD_ADD_INT_CONSTANT("_UNARY_SQUARE", PYFASTCD_UNARY_SQUARE);
    PYFASTCD_ADD_INT_CONSTANT("_BINARY_MULTIPLY",
            PYFASTCD_BINARY_MULTIPLY);
    PYFASTCD_ADD_INT_CONSTANT("_BINARY_IMAG_CROSS",
            PYFASTCD_BINARY_IMAG_CROSS);
    PYFASTCD_ADD_INT_CONSTANT("_BINARY_COMMUTATOR",
            PYFASTCD_BINARY_COMMUTATOR);
    PYFASTCD_ADD_INT_CONSTANT("_BINARY_ANTICOMMUTATOR",
            PYFASTCD_BINARY_ANTICOMMUTATOR);
    PYFASTCD_ADD_INT_CONSTANT("_SCALAR_INNER", PYFASTCD_SCALAR_INNER);
    PYFASTCD_ADD_INT_CONSTANT("_SCALAR_NORM_SQUARED",
            PYFASTCD_SCALAR_NORM_SQUARED);
    PYFASTCD_ADD_INT_CONSTANT("_SCALAR_NORM", PYFASTCD_SCALAR_NORM);
#undef PYFASTCD_ADD_INT_CONSTANT

    return module;
}
