#include "reference_cd.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

// Check a dimension without depending on any library routine
static bool reference_valid_dimension(size_t dimension) {
    return dimension != 0 && (dimension & (dimension - 1)) == 0;
}

// Check whether two nonempty byte ranges overlap
static bool reference_ranges_overlap(const void* left, size_t left_bytes,
        const void* right, size_t right_bytes) {
    uintptr_t left_begin = (uintptr_t)left;
    uintptr_t right_begin = (uintptr_t)right;

    if (left_bytes == 0 || right_bytes == 0) return false;
    if (left_begin <= right_begin) return right_begin - left_begin < left_bytes;
    return left_begin - right_begin < right_bytes;
}

// Evaluate the basis sign of the defining product convention recursively
static int reference_sign_recursive(size_t dimension, size_t left_index,
        size_t right_index) {
    size_t half;

    if (dimension == 1) return 1;
    half = dimension / 2;
    if (left_index < half && right_index < half) {
        return reference_sign_recursive(half, left_index, right_index);
    }
    if (left_index < half) {
        return reference_sign_recursive(half, right_index - half, left_index);
    }
    if (right_index < half) {
        int conjugate_sign = right_index == 0 ? 1 : -1;
        return conjugate_sign
            * reference_sign_recursive(half, left_index - half, right_index);
    }
    {
        size_t left_low = left_index - half;
        size_t right_low = right_index - half;
        int conjugate_sign = right_low == 0 ? 1 : -1;
        return -conjugate_sign
            * reference_sign_recursive(half, right_low, left_low);
    }
}

// Copy the standard involution
static void reference_conjugate(double* out, const double* x,
        size_t dimension) {
    out[0] = x[0];
    for (size_t i = 1; i < dimension; i++) out[i] = -x[i];
}

/*
 * Evaluate (a,b)(c,d) = (ac - conjugate(d)b, da + b conjugate(c)) literally.
 * scratch holds 2N-2 doubles: at a node of size N its first N doubles hold
 * one conjugated operand and one product, and the rest is child scratch.
 */
// Evaluate the defining Cayley--Dickson doubling product recursively
static void reference_multiply_recursive(double* out, const double* x,
        const double* y, size_t dimension, double* scratch) {
    size_t half;
    double* temporary_input;
    double* temporary_product;
    double* child_scratch;

    if (dimension == 1) {
        out[0] = x[0] * y[0];
        return;
    }
    half = dimension / 2;
    temporary_input = scratch;
    temporary_product = scratch + half;
    child_scratch = scratch + dimension;

    reference_multiply_recursive(out, x, y, half, child_scratch);
    reference_conjugate(temporary_input, y + half, half);
    reference_multiply_recursive(temporary_product, temporary_input, x + half,
            half, child_scratch);
    for (size_t i = 0; i < half; i++) out[i] -= temporary_product[i];

    reference_multiply_recursive(out + half, y + half, x, half, child_scratch);
    reference_conjugate(temporary_input, y, half);
    reference_multiply_recursive(temporary_product, x + half, temporary_input,
            half, child_scratch);
    for (size_t i = 0; i < half; i++) out[half + i] += temporary_product[i];
}

// Compute the sign in e_left * e_right = sign * e_(left XOR right)
FastCDStatus reference_cd_basis_sign(size_t dimension, size_t left_index,
        size_t right_index, int* sign) {
    if (sign == NULL) return FASTCD_NULL_POINTER;
    if (!reference_valid_dimension(dimension)) return FASTCD_INVALID_DIMENSION;
    if (left_index >= dimension || right_index >= dimension) {
        return FASTCD_INVALID_BASIS_INDEX;
    }
    *sign = reference_sign_recursive(dimension, left_index, right_index);
    return FASTCD_SUCCESS;
}

// Validate the arguments shared by both oracles
static FastCDStatus reference_validate(const double* out, const double* x,
        const double* y, size_t dimension) {
    size_t element_bytes;

    if (!reference_valid_dimension(dimension)) return FASTCD_INVALID_DIMENSION;
    if (out == NULL || x == NULL || y == NULL) return FASTCD_NULL_POINTER;
    if (dimension > SIZE_MAX / sizeof(double) / 2) return FASTCD_SIZE_OVERFLOW;
    element_bytes = dimension * sizeof(double);
    if (reference_ranges_overlap(out, element_bytes, x, element_bytes)
            || reference_ranges_overlap(out, element_bytes, y, element_bytes)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    return FASTCD_SUCCESS;
}

// Set out = xy as a compensated signed sum over the basis multiplication table
FastCDStatus reference_cd_mul_signed_f64(double* out, const double* x,
        const double* y, size_t dimension) {
    FastCDStatus status = reference_validate(out, x, y, dimension);

    if (status != FASTCD_SUCCESS) return status;
    for (size_t k = 0; k < dimension; k++) {
        long double sum = 0.0L;
        long double correction = 0.0L;

        for (size_t i = 0; i < dimension; i++) {
            size_t j = i ^ k;
            long double product = (long double)x[i] * (long double)y[j];
            long double next;

            if (reference_sign_recursive(dimension, i, j) < 0) product = -product;
            next = sum + product;
            if (fabsl(sum) >= fabsl(product)) {
                correction += (sum - next) + product;
            } else {
                correction += (product - next) + sum;
            }
            sum = next;
        }
        out[k] = (double)(sum + correction);
    }
    return FASTCD_SUCCESS;
}

// Set out = xy by the defining doubling recursion; out must not alias inputs
FastCDStatus reference_cd_mul_f64(double* out, const double* x,
        const double* y, size_t dimension) {
    double* scratch = NULL;
    FastCDStatus status = reference_validate(out, x, y, dimension);

    if (status != FASTCD_SUCCESS) return status;
    if (dimension > 1) {
        scratch = malloc(2 * (dimension - 1) * sizeof(double));
        if (scratch == NULL) return FASTCD_INSUFFICIENT_WORKSPACE;
    }
    reference_multiply_recursive(out, x, y, dimension, scratch);
    free(scratch);
    return FASTCD_SUCCESS;
}
