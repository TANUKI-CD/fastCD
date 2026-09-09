#ifndef FASTCD_INTERNAL_H
#define FASTCD_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "fastcd/fastcd.h"

/*
 * Private kernel interfaces shared by the fastCD translation units.  Every
 * function here assumes that the caller has already validated dimensions,
 * pointers, overlap, and workspace lengths.  Nothing in this header is part
 * of the installed C API.
 */

/* ---------- Quasilinear multiplication ---------- */

// Return the scratch length used by the quasilinear product and cross product
size_t fastcd_quasilinear_workspace(size_t dimension);

// Evaluate one quasilinear product after validation
void fastcd_quasilinear_mul(double* restrict out, const double* x,
        const double* y, size_t dimension, double* restrict workspace);

// Evaluate a validated strided batch of quasilinear products
void fastcd_quasilinear_mul_batch(double* restrict out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, double* restrict workspace);

// Evaluate the imaginary alternating product with the quasilinear recursion
void fastcd_quasilinear_cross(double* restrict out, const double* x,
        const double* y, size_t dimension, double* restrict workspace);

/* ---------- Cariow--Cariowa multiplication ---------- */

/* Largest dimension with a generated straight-line Cariow--Cariowa kernel;
 * larger supported dimensions use the table-driven kernel and 2N scratch.
 * Defined here so that the workspace query stays inline on the validation
 * path of every product. */
#ifndef FASTCD_CC_LEAF
#define FASTCD_CC_LEAF 64
#endif

// Report whether a Cariow--Cariowa kernel exists for a valid dimension
bool fastcd_cc_supported(size_t dimension);

// Return the scratch length used by the Cariow--Cariowa product
static inline size_t fastcd_cc_workspace(size_t dimension) {
    return dimension > FASTCD_CC_LEAF ? 2 * dimension : 0;
}

// Evaluate one Cariow--Cariowa product after validation
void fastcd_cc_mul(double* restrict out, const double* x, const double* y,
        size_t dimension, double* restrict workspace);

// Evaluate a validated strided batch of Cariow--Cariowa products
void fastcd_cc_mul_batch(double* restrict out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, double* restrict workspace);

/* ---------- Direct (defining recursion) multiplication ---------- */

// Return the scratch length used by the direct product
size_t fastcd_direct_workspace(size_t dimension);

// Evaluate one direct product after validation
void fastcd_direct_mul(double* restrict out, const double* x, const double* y,
        size_t dimension, double* restrict workspace);

// Evaluate a validated strided batch of direct products
void fastcd_direct_mul_batch(double* restrict out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, double* restrict workspace);

#endif
