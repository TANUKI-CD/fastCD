#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "_domain.h"
#include "finite_backend.h"
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/*
 * Buffer-protocol bridge for the arrays of explicitly selected complex and
 * finite-field parents. Real parents use the _core extension and never reach
 * this module, and _core also runs Element arithmetic over C and finite
 * fields with the contexts created here. No NumPy C API is required: Python
 * adapters select parents and combine differing batch shapes, and this module
 * validates layouts, repeats an operand that holds one record, and runs
 * coefficient loops.
 *
 * A context owns a copied finite plan or complex dimension, never an operand
 * pointer. Each call pins its buffers, validates the entire batch before
 * writing and owns separate scratch, including while the GIL is released.
 */
typedef PyFastCDDomain Domain;

// Free the native value owned by a field or algebra-context capsule
static void capsule_free(PyObject* capsule) {
    const char* name = PyCapsule_GetName(capsule);
    if (name) PyMem_Free(PyCapsule_GetPointer(capsule, name));
}

// Translate finite status codes at the Python boundary, with the GIL held
static int ff_error(FastCDFFStatus s) {
    return pyfastcd_domain_ff_error(s);
}

// Translate complex status codes using the corresponding real-API exceptions
static int c_error(FastCDStatus s) {
    return pyfastcd_domain_complex_error(s);
}

// Construct a checked field and return its capsule plus actual immutable modulus
static PyObject* field_new(PyObject* module, PyObject* args) {
    (void)module;
    unsigned long p, d;
    PyObject* mod;
    if (!PyArg_ParseTuple(args, "kkO", &p, &d, &mod)) return NULL;
    if (p > INT32_MAX || p < 3 || d < 1 || d > 12) {
        PyErr_SetString(
                PyExc_ValueError, "unsupported characteristic or extension degree");
        return NULL;
    }
    uint32_t coefficients[13];
    const uint32_t* input = NULL;
    if (mod != Py_None) {
        PyObject* seq = PySequence_Fast(mod, "modulus must be a sequence");
        if (!seq) return NULL;
        if (PySequence_Fast_GET_SIZE(seq) != (Py_ssize_t)d + 1) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "modulus length does not match degree");
            return NULL;
        }
        for (size_t j = 0; j <= d; j++) {
            unsigned long v =
                    PyLong_AsUnsignedLong(PySequence_Fast_GET_ITEM(seq, (Py_ssize_t)j));
            if (PyErr_Occurred() || v >= p) {
                Py_DECREF(seq);
                if (!PyErr_Occurred())
                    PyErr_SetString(
                            PyExc_ValueError, "noncanonical modulus coefficient");
                return NULL;
            }
            coefficients[j] = (uint32_t)v;
        }
        Py_DECREF(seq);
        input = coefficients;
    }
    FastCDField* f = PyMem_Malloc(sizeof(*f));
    if (!f) return PyErr_NoMemory();
    if (ff_error(fastcd_field_init(f, (uint32_t)p, (uint32_t)d, input))) {
        PyMem_Free(f);
        return NULL;
    }
    PyObject* capsule = PyCapsule_New(f, "pyfastcd.field", capsule_free);
    if (!capsule) {
        PyMem_Free(f);
        return NULL;
    }
    PyObject* tuple = PyTuple_New((Py_ssize_t)d + 1);
    if (!tuple) {
        Py_DECREF(capsule);
        return NULL;
    }
    for (size_t j = 0; j <= d; j++) {
        PyObject* v = PyLong_FromUnsignedLong(f->modulus[j]);
        if (!v) {
            Py_DECREF(tuple);
            Py_DECREF(capsule);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, (Py_ssize_t)j, v);
    }
    PyObject* result = PyTuple_Pack(2, capsule, tuple);
    Py_DECREF(capsule);
    Py_DECREF(tuple);
    return result;
}

// Copy field state into a context whose lifetime is independent of that capsule
static PyObject* context_new(PyObject* module, PyObject* args) {
    (void)module;
    PyObject* field;
    Py_ssize_t n;
    int scalar;
    if (!PyArg_ParseTuple(args, "Oni", &field, &n, &scalar)) return NULL;
    if (n < 1 || !fastcd_valid_dimension((size_t)n)) {
        PyErr_SetString(PyExc_ValueError, "invalid dimension");
        return NULL;
    }
    Domain* c = PyMem_Calloc(1, sizeof(*c));
    if (!c) return PyErr_NoMemory();
    c->n = (size_t)n;
    c->d = 1;
    c->finite = field != Py_None;
    c->field_scalar = scalar;
    if (c->finite) {
        FastCDField* f = PyCapsule_GetPointer(field, "pyfastcd.field");
        if (!f) {
            PyMem_Free(c);
            return NULL;
        }
        c->d = f->degree;
        size_t level;
        fastcd_level(c->n, &level);
        if (ff_error(fastcd_ff_plan_init(&c->plan, f, level))) {
            PyMem_Free(c);
            return NULL;
        }
    }
    size_t item = c->finite ? sizeof(uint32_t) : sizeof(FastCDComplexF64);
    if (c->n > (size_t)PY_SSIZE_T_MAX / item / c->d) {
        PyMem_Free(c);
        PyErr_SetString(PyExc_OverflowError, "coefficient storage overflow");
        return NULL;
    }
    c->bytes = c->n * c->d * item;
    // Count the complete trailing storage axes of one record
    c->features = scalar ? (c->d > 1) : 1 + (c->d > 1);
    if (c->features == 2) {
        c->shape[0] = n;
        c->shape[1] = (Py_ssize_t)c->d;
    } else if (c->features == 1)
        c->shape[0] = (Py_ssize_t)(scalar ? c->d : c->n);
    PyObject* cap = PyCapsule_New(c, PYFASTCD_DOMAIN_CAPSULE, capsule_free);
    if (!cap) PyMem_Free(c);
    return cap;
}

// Pin native storage and validate record axes, alignment, and writable output
static int get_view(Py_buffer* v, PyObject* obj, Domain* c, int scalar, int output) {
    if (PyObject_GetBuffer(
                obj, v, PyBUF_FORMAT | PyBUF_STRIDES | (output ? PyBUF_WRITABLE : 0)))
        return -1;
    const char* fmt = v->format;
    if (fmt && (*fmt == '@' || *fmt == '=')) fmt++;
    size_t item = c->finite ? 4 : 16;
    // NumPy exports uint32 as "L" where long has 32 bits, as on Windows
    int known = fmt &&
            (c->finite ? !strcmp(fmt, "I") || !strcmp(fmt, "L") : !strcmp(fmt, "Zd"));
    if (v->itemsize != (Py_ssize_t)item || !known || v->ndim < 0 ||
            v->ndim > PyBUF_MAX_NDIM) {
        PyErr_SetString(PyExc_TypeError, "incompatible native coefficient dtype");
        return -1;
    }
    if (v->suboffsets) {
        for (int k = 0; k < v->ndim; k++)
            if (v->suboffsets[k] >= 0) {
                PyErr_SetString(PyExc_ValueError, "indirect buffers are unsupported");
                return -1;
            }
    }
    int features = scalar ? (c->d > 1) : c->features;
    if (v->ndim < features) {
        PyErr_SetString(PyExc_ValueError, "missing coefficient axes");
        return -1;
    }
    // Record axes must be contiguous even when leading batch strides are
    // negative or zero, and singleton axes need no particular physical stride
    Py_ssize_t stride = (Py_ssize_t)item;
    for (int j = features - 1; j >= 0; j--) {
        Py_ssize_t extent = scalar ? (Py_ssize_t)c->d : c->shape[j];
        int k = v->ndim - features + j;
        if (v->shape[k] != extent ||
                (v->len && v->strides[k] != stride && extent > 1)) {
            PyErr_SetString(PyExc_ValueError, "incompatible coefficient layout");
            return -1;
        }
        if (extent && stride > PY_SSIZE_T_MAX / extent) {
            PyErr_SetString(PyExc_OverflowError, "coefficient stride overflow");
            return -1;
        }
        stride *= extent;
    }
    Py_ssize_t alignment = c->finite ? 4 : 8;
    if (v->len && ((uintptr_t)v->buf % (uintptr_t)alignment)) {
        PyErr_SetString(PyExc_ValueError, "unaligned coefficients");
        return -1;
    }
    for (int k = 0; k < v->ndim - features; k++) {
        if (v->shape[k] < 0 || v->strides[k] % alignment) {
            PyErr_SetString(PyExc_ValueError, "invalid batch layout");
            return -1;
        }
        if (v->shape[k] > 1) {
            uint64_t a = v->strides[k] < 0 ? (uint64_t)(-(v->strides[k] + 1)) + 1
                                           : (uint64_t)v->strides[k];
            if (a && (uint64_t)(v->shape[k] - 1) > (uint64_t)PY_SSIZE_T_MAX / a) {
                PyErr_SetString(PyExc_OverflowError, "batch stride overflow");
                return -1;
            }
        }
    }
    if (output && !PyBuffer_IsContiguous(v, 'C')) {
        PyErr_SetString(PyExc_ValueError, "out must be C-contiguous");
        return -1;
    }
    return 0;
}

/* The record pointers of every active view at one C-order batch index. The
 * output fixes the batch shape. An input either has that shape or has no
 * batch axes, in which case its one record is reused at every index. Stepping
 * adds strides like an odometer, so broadcast and reversed layouts are
 * visited without any division. */
typedef struct {
    char* at[4];
    const Py_ssize_t* strides[4];
    const Py_ssize_t* shape;
    int active, rank;
    Py_ssize_t index[PyBUF_MAX_NDIM];
} Cursor;

// Strides of an input that repeats one record along every batch axis
static const Py_ssize_t repeat_strides[PyBUF_MAX_NDIM] = {0};

// Point a cursor at the first record of each validated view
static void cursor_start(
        Cursor* cursor, Py_buffer* views, const int* single, int active, int rank) {
    cursor->shape = views[0].shape;
    cursor->active = active;
    cursor->rank = rank;
    for (int j = 0; j < active; j++) {
        cursor->at[j] = views[j].buf;
        cursor->strides[j] = single[j] ? repeat_strides : views[j].strides;
    }
    for (int k = 0; k < rank; k++) cursor->index[k] = 0;
}

// Step a cursor to the next batch index in C order
static void cursor_next(Cursor* cursor) {
    for (int k = cursor->rank - 1; k >= 0; k--) {
        Py_ssize_t extent = cursor->shape[k];
        if (++cursor->index[k] < extent) {
            for (int j = 0; j < cursor->active; j++)
                cursor->at[j] += cursor->strides[j][k];
            return;
        }
        cursor->index[k] = 0;
        for (int j = 0; j < cursor->active; j++)
            cursor->at[j] -= (extent - 1) * cursor->strides[j][k];
    }
}

// Compare byte spans without undefined subtraction of unrelated pointers
static int overlap(const void* a, size_t an, const void* b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return an && bn && (x <= y ? y - x < an : x - y < bn);
}

/* A contiguous batch or one fully broadcast coefficient record can use the
 * public batch kernel, and general views use the validated strided loop. */
// Find the record stride of a contiguous or fully broadcast view, if any
static int flat_stride(
        Py_buffer* v, int rank, int single, size_t words, size_t* stride) {
    if (single) {
        *stride = 0;
        return 1;
    }
    if (PyBuffer_IsContiguous(v, 'C')) {
        *stride = words;
        return 1;
    }
    for (int k = 0; k < rank; k++)
        if (v->shape[k] > 1 && v->strides[k] != 0) return 0;
    *stride = 0;
    return 1;
}

// Preflight a complete broadcast batch, then execute it with one scratch buffer
static PyObject* apply(PyObject* module, PyObject* args) {
    (void)module;
    PyObject *cap, *objects[4];
    int operation, algorithm;
    Py_ssize_t basis;
    if (!PyArg_ParseTuple(args, "OiOOOOin", &cap, &operation, &objects[0], &objects[1],
                &objects[2], &objects[3], &algorithm, &basis))
        return NULL;
    Domain* c = PyCapsule_GetPointer(cap, PYFASTCD_DOMAIN_CAPSULE);
    if (!c) return NULL;
    if (operation < 0 || operation > FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR || basis < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid operation or basis index");
        return NULL;
    }
    if (algorithm != FASTCD_ALGORITHM_AUTO && algorithm != FASTCD_ALGORITHM_DIRECT &&
            algorithm != FASTCD_ALGORITHM_QUASILINEAR) {
        PyErr_SetString(PyExc_ValueError, "unsupported multiplication algorithm");
        return NULL;
    }
    FastCDOperation op = (FastCDOperation)operation;
    int reduction = op == FASTCD_OP_INNER || op == FASTCD_OP_QUADRATIC_NORM;
    int scale = op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE;
    int binary = op == FASTCD_OP_MULTIPLY || op == FASTCD_OP_ADD ||
            op == FASTCD_OP_SUBTRACT || scale || op == FASTCD_OP_INNER ||
            op == FASTCD_OP_IMAG_CROSS || op == FASTCD_OP_COMMUTATOR ||
            op == FASTCD_OP_ANTICOMMUTATOR || op == FASTCD_OP_ASSOCIATOR ||
            op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR ||
            op == FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR;
    int ternary = op == FASTCD_OP_ASSOCIATOR ||
            op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR ||
            op == FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR;
    if ((binary && objects[2] == Py_None) || (ternary && objects[3] == Py_None)) {
        PyErr_SetString(PyExc_TypeError, "missing operand");
        return NULL;
    }
    // Zero every view so all exits can release exactly the acquired exports
    Py_buffer views[4];
    memset(views, 0, sizeof(views));
    int active = ternary ? 4 : binary ? 3 : 2;
    if (get_view(&views[0], objects[0], c, reduction, 1) ||
            get_view(&views[1], objects[1], c, 0, 0))
        goto fail;
    if (binary && get_view(&views[2], objects[2], c, scale, 0)) goto fail;
    if (ternary && get_view(&views[3], objects[3], c, 0, 0)) goto fail;
    // The output has every batch axis, and get_view checked its record axes
    int rank = views[0].ndim - (reduction ? (c->d > 1) : c->features);
    size_t count = 1;
    for (int k = 0; k < rank; k++) {
        size_t extent = (size_t)views[0].shape[k];
        if (extent && count > (size_t)PY_SSIZE_T_MAX / extent) {
            PyErr_SetString(PyExc_OverflowError, "batch size overflow");
            goto fail;
        }
        count *= extent;
    }
    size_t lengths[4] = {reduction ? c->d * (c->finite ? 4 : 16) : c->bytes, c->bytes,
            scale ? c->d * (c->finite ? 4 : 16) : c->bytes, c->bytes};
    int single[4] = {0, 0, 0, 0};
    for (int j = 0; j < active; j++) {
        int features = j == 0 && reduction ? (c->d > 1)
                : j == 2 && scale          ? (c->d > 1)
                                           : c->features;
        // An input without batch axes is one record broadcast over the batch
        single[j] = j > 0 && rank > 0 && views[j].ndim == features;
        if (!single[j] && views[j].ndim - features != rank) {
            PyErr_SetString(PyExc_ValueError, "batch rank mismatch");
            goto fail;
        }
        for (int k = 0; !single[j] && k < rank; k++)
            if (views[j].shape[k] != views[0].shape[k]) {
                PyErr_SetString(PyExc_ValueError, "batch shape mismatch");
                goto fail;
            }
        if (count > (size_t)PY_SSIZE_T_MAX / lengths[j] ||
                (size_t)views[j].len != (single[j] ? 1 : count) * lengths[j]) {
            PyErr_SetString(PyExc_OverflowError, "buffer extent mismatch");
            goto fail;
        }
    }
    if (count == 0) {
        for (int j = 0; j < active; j++) PyBuffer_Release(&views[j]);
        Py_RETURN_NONE;
    }
    /* Validate every input record before the first output write: mutable
     * uint32 arrays need canonical-residue checks on every call, not only at
     * construction, and a failure late in a batch must leave all outputs
     * intact.
     *
     * Contiguous batches and single records admit the same checks on whole
     * spans first. A clean result proves that every record would pass. Any
     * other result, and every operation with a denominator, is left to the
     * record loop, which reports the first failing record. */
    int clean = op != FASTCD_OP_RECIPROCAL && op != FASTCD_OP_DIVIDE;
    size_t steps[4] = {0, 0, 0, 0}, spans[4] = {0, 0, 0, 0};
    for (int j = 1; clean && j < active; j++) {
        clean = flat_stride(&views[j], rank, single[j], lengths[j], &steps[j]);
        spans[j] = steps[j] ? count * lengths[j] : lengths[j];
        if (!clean || overlap(views[0].buf, count * lengths[0], views[j].buf, spans[j]))
            clean = 0;
        else if (c->finite) {
            const uint32_t* a = (const uint32_t*)views[j].buf;
            uint32_t prime = c->plan.field.prime, invalid = 0;
            for (size_t k = 0; k < spans[j] / 4; k++) invalid |= a[k] >= prime;
            clean = !invalid;
        }
    }
    // Two inputs must be the same records throughout or share no storage
    for (int j = 1; clean && !scale && j < active; j++)
        for (int k = j + 1; k < active; k++)
            if (!(views[j].buf == views[k].buf && steps[j] == steps[k]) &&
                    overlap(views[j].buf, spans[j], views[k].buf, spans[k]))
                clean = 0;
    Cursor cursor;
    cursor_start(&cursor, views, single, active, rank);
    for (size_t i = 0; !clean && i < count; i++, cursor_next(&cursor)) {
        char** ptr = cursor.at;
        for (int j = 1; j < active; j++) {
            // One broadcast record needs these checks only once
            if (single[j] && i) continue;
            if (overlap(views[0].buf, count * lengths[0], ptr[j], lengths[j])) {
                PyErr_SetString(PyExc_ValueError, "output overlaps an input");
                goto fail;
            }
            if (c->finite) {
                // Test a whole record without branching so the loop vectorizes
                const uint32_t* a = (const uint32_t*)ptr[j];
                uint32_t prime = c->plan.field.prime, invalid = 0;
                for (size_t k = 0; k < lengths[j] / 4; k++) invalid |= a[k] >= prime;
                if (invalid) {
                    PyErr_SetString(PyExc_ValueError,
                            "finite coefficients must be canonical residues");
                    goto fail;
                }
            }
        }
        if (!scale)
            for (int j = 1; j < active; j++)
                for (int k = j + 1; k < active; k++)
                    if (ptr[j] != ptr[k] &&
                            overlap(ptr[j], lengths[j], ptr[k], lengths[k])) {
                        PyErr_SetString(PyExc_ValueError,
                                "partially overlapping input records");
                        goto fail;
                    }
        if (op == FASTCD_OP_RECIPROCAL || op == FASTCD_OP_DIVIDE) {
            // Preflight every denominator too, since nonzero CD elements can
            // have zero quadratic norm over C or finite fields
            int iszero = 1;
            if (c->finite) {
                uint32_t q[12];
                if (scale)
                    memcpy(q, ptr[2], c->d * 4);
                else if (ff_error(fastcd_apply_ff(q, (uint32_t*)ptr[1], NULL, NULL,
                                 &c->plan, FASTCD_OP_QUADRATIC_NORM, 0,
                                 FASTCD_ALGORITHM_AUTO, NULL, 0)))
                    goto fail;
                for (size_t k = 0; k < c->d; k++)
                    if (q[k]) iszero = 0;
            } else {
                FastCDComplexF64 q;
                if (scale)
                    memcpy(&q, ptr[2], 16);
                else if (c_error(fastcd_quadratic_norm_c128(
                                 &q, (FastCDComplexF64*)ptr[1], c->n)))
                    goto fail;
                iszero = q.real == 0 && q.imag == 0;
            }
            if (iszero) {
                PyErr_SetString(PyExc_ZeroDivisionError, PYFASTCD_ZERO_DIVISOR_MESSAGE);
                goto fail;
            }
        }
    }
    // Finite queries count uint32 words and complex queries count doubles,
    // and one allocation is reused across all records
    size_t need = 0;
    if (c->finite) {
        if (ff_error(fastcd_apply_workspace_size_ff(&c->plan, op, &need))) goto fail;
    } else if (c_error(fastcd_workspace_size_c128(c->n, op, &need)))
        goto fail;
    size_t item = c->finite ? 4 : 8;
    if (need > SIZE_MAX / item) {
        PyErr_SetString(PyExc_OverflowError, "workspace size overflow");
        goto fail;
    }
    void* scratch = need ? PyMem_Malloc(need * item) : NULL;
    if (need && !scratch) {
        PyErr_NoMemory();
        goto fail;
    }
    int status = 0;
    FastCDWorkspaceF64 w = {scratch, need};
    size_t record_words = c->finite ? c->n * c->d : c->n, xs = 0, ys = 0;
    int batch_product = count >= 2 && op == FASTCD_OP_MULTIPLY &&
            flat_stride(&views[1], rank, single[1], record_words, &xs) &&
            flat_stride(&views[2], rank, single[2], record_words, &ys);
    /* Buffer exports keep storage alive while native code runs without the
     * GIL, and no Python API is called until that thread state is restored.
     * Exports do not lock data, so callers must synchronize concurrent
     * writers. */
    PyThreadState* thread = count >= 32 ? PyEval_SaveThread() : NULL;
    if (batch_product) {
        status = c->finite
                ? (int)fastcd_mul_batch_ff(views[0].buf, record_words, views[1].buf, xs,
                          views[2].buf, ys, count, &c->plan, (FastCDAlgorithm)algorithm,
                          scratch, need)
                : (int)fastcd_mul_batch_c128(views[0].buf, record_words, views[1].buf,
                          xs, views[2].buf, ys, count, c->n, (FastCDAlgorithm)algorithm,
                          &w);
    }
    cursor_start(&cursor, views, single, active, rank);
    for (size_t i = 0; !batch_product && i < count; i++, cursor_next(&cursor)) {
        void* out = cursor.at[0];
        const void* x = cursor.at[1];
        const void* y = binary ? cursor.at[2] : NULL;
        const void* z = ternary ? cursor.at[3] : NULL;
        status = c->finite
                ? (int)fastcd_apply_ff(out, x, y, z, &c->plan, op, (size_t)basis,
                          (FastCDAlgorithm)algorithm, scratch, need)
                : (int)fastcd_apply_c128(out, x, y, z, c->n, op, (size_t)basis,
                          (FastCDAlgorithm)algorithm, &w);
        if (status) break;
    }
    if (thread) PyEval_RestoreThread(thread);
    PyMem_Free(scratch);
    if (c->finite ? ff_error((FastCDFFStatus)status) : c_error((FastCDStatus)status))
        goto fail;
    for (int j = 0; j < 4; j++)
        if (views[j].obj) PyBuffer_Release(&views[j]);
    Py_RETURN_NONE;
fail:
    for (int j = 0; j < 4; j++)
        if (views[j].obj) PyBuffer_Release(&views[j]);
    return NULL;
}

// Report whether contiguous uint32 storage holds only canonical residues
static PyObject* canonical(PyObject* module, PyObject* args) {
    (void)module;
    PyObject *cap, *object;
    if (!PyArg_ParseTuple(args, "OO", &cap, &object)) return NULL;
    Domain* c = PyCapsule_GetPointer(cap, PYFASTCD_DOMAIN_CAPSULE);
    if (!c) return NULL;
    Py_buffer view;
    if (PyObject_GetBuffer(object, &view, PyBUF_C_CONTIGUOUS)) return NULL;
    if (!c->finite || view.itemsize != 4 || (uintptr_t)view.buf % 4) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "expected aligned uint32 residues");
        return NULL;
    }
    const uint32_t* a = view.buf;
    uint32_t prime = c->plan.field.prime, invalid = 0;
    for (Py_ssize_t k = 0; k < view.len / 4; k++) invalid |= a[k] >= prime;
    PyBuffer_Release(&view);
    return PyBool_FromLong(!invalid);
}

// Report private dispatch metadata for validation and benchmark provenance
static PyObject* backend(PyObject* module, PyObject* cap) {
    (void)module;
    Domain* c = PyCapsule_GetPointer(cap, PYFASTCD_DOMAIN_CAPSULE);
    if (!c) return NULL;
    if (!c->finite) return PyUnicode_FromString("complex_alternating");
    static const char* names[] = {"scalar", "byte_prime", "byte_quadratic", "ternary16",
            "ternary32", "ternary64", "medium_bar", "medium_mont", "medium_bar_prime",
            "short_prime", "medium_mont_prime", "mont4", "mont8", "mont16",
            "extension4", "extension8", "extension16", "small8", "small16", "small32",
            "small_prime32", "small_plain32", "short8", "short16", "short32"};
    size_t words;
    FastCDFFBackend b = fastcd_ff_finite_select(&c->plan, 0, &words);
    return PyUnicode_FromString(names[b]);
}

// Report the algorithm that AUTO selects for this domain and dimension
static PyObject* default_algorithm(PyObject* module, PyObject* cap) {
    (void)module;
    Domain* c = PyCapsule_GetPointer(cap, PYFASTCD_DOMAIN_CAPSULE);
    if (!c) return NULL;
    FastCDAlgorithm algorithm = fastcd_domain_auto_algorithm(c->n);
    return PyUnicode_FromString(fastcd_algorithm_string(algorithm));
}

static PyMethodDef methods[] = {
        {"field", field_new, METH_VARARGS,
                "Validate and construct an immutable native field context."},
        {"context", context_new, METH_VARARGS,
                "Construct an immutable algebra context."},
        {"apply", apply, METH_VARARGS,
                "Validate a complete broadcast batch before applying native "
                "arithmetic."},
        {"canonical", canonical, METH_VARARGS,
                "Report whether uint32 storage holds only canonical residues."},
        {"backend", backend, METH_O, "Report the selected native arithmetic backend."},
        {"default_algorithm", default_algorithm, METH_O,
                "Report the algorithm that AUTO selects for this domain."},
        {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_domains_core",
        "Native finite and complex CD arithmetic.", -1, methods, NULL, NULL, NULL,
        NULL};

// Initialize the private extension; the real _core module does not use it
PyMODINIT_FUNC PyInit__domains_core(void) {
    return PyModule_Create(&module);
}
