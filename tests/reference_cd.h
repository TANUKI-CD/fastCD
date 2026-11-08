#ifndef REFERENCE_CD_H
#define REFERENCE_CD_H

#include <stddef.h>

#include "fastcd/fastcd.h"

/*
 * Independent test oracle for Cayley--Dickson products. It evaluates the
 * defining doubling formula literally, recursing to scalars, and shares no
 * code with the library: dimension validation, conjugation and basis signs are
 * derived here. It favors clarity over speed and allocates its own scratch, so
 * only tests and benchmark verification use it.
 */

// Compute the sign in e_left * e_right = sign * e_(left XOR right)
FastCDStatus reference_cd_basis_sign(size_t dimension, size_t left_index,
        size_t right_index, int* sign);

// Set out = xy by the defining doubling recursion; out must not alias inputs
FastCDStatus reference_cd_mul_f64(double* out, const double* x,
        const double* y, size_t dimension);

/*
 * Second oracle with a different structure: out[k] is the signed sum over i
 * of sign(i, i XOR k) x[i] y[i XOR k], accumulated in long double with
 * Neumaier compensation. Only the basis-sign derivation is shared with the
 * recursive oracle, so a transcription error in the doubling formula cannot be
 * common to both. It costs O(N^2 log N) and is meant for dimensions up to 1024.
 */
// Set out = xy as a compensated signed sum over the basis multiplication table
FastCDStatus reference_cd_mul_signed_f64(double* out, const double* x,
        const double* y, size_t dimension);

#endif
