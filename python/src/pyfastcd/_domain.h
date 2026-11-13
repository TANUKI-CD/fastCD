#ifndef PYFASTCD_DOMAIN_H
#define PYFASTCD_DOMAIN_H

#include "fastcd/complex.h"
#include "fastcd/finite.h"

/*
 * Arithmetic context of a complex or finite-field parent. The _domains_core
 * module creates one context per parent and publishes it in a capsule, and
 * the _core module reads it to run Element arithmetic without leaving C.
 * Both modules are built from one source tree, so they agree on this layout.
 * A context is immutable after construction and owns no operand storage.
 * This header needs Python.h to be included first.
 */
#define PYFASTCD_DOMAIN_CAPSULE "pyfastcd.domain"

// Message for a zero quadratic norm in a reciprocal or a zero scalar divisor
#define PYFASTCD_ZERO_DIVISOR_MESSAGE "zero quadratic norm or scalar divisor"

/* The features of a context are the complete trailing storage axes of one
 * record, never batch axes. Scalars of a prime field have none and scalars of
 * an extension field have (d,). Records of a Cayley--Dickson algebra have
 * (N,) over a prime field or over C, and (N,d) over an extension field. */
typedef struct {
    int finite, field_scalar, features;
    size_t n, d, bytes;
    Py_ssize_t shape[2];
    FastCDPlanFF plan;
} PyFastCDDomain;

// Translate a finite-field status into a Python exception, with the GIL held
static inline int pyfastcd_domain_ff_error(FastCDFFStatus status) {
    if (status == FASTCD_FF_SUCCESS) return 0;
    PyErr_SetString(status == FASTCD_FF_ZERO_NORM             ? PyExc_ZeroDivisionError
                    : status == FASTCD_FF_SIZE_OVERFLOW       ? PyExc_OverflowError
                    : status == FASTCD_FF_INVALID_BASIS_INDEX ? PyExc_IndexError
                                                              : PyExc_ValueError,
            fastcd_ff_status_string(status));
    return -1;
}

// Translate a complex status using the corresponding real-API exceptions
static inline int pyfastcd_domain_complex_error(FastCDStatus status) {
    if (status == FASTCD_SUCCESS) return 0;
    PyErr_SetString(status == FASTCD_ZERO_NORM             ? PyExc_ZeroDivisionError
                    : status == FASTCD_SIZE_OVERFLOW       ? PyExc_OverflowError
                    : status == FASTCD_INVALID_BASIS_INDEX ? PyExc_IndexError
                                                           : PyExc_ValueError,
            fastcd_status_string(status));
    return -1;
}

#endif
