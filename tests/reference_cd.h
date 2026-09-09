#ifndef REFERENCE_CD_H
#define REFERENCE_CD_H

#include <stddef.h>

#include "fastcd/fastcd.h"

/*
 * Independent test oracle.  It evaluates the defining Cayley--Dickson
 * doubling formula literally, recursing to scalars, and shares no code with
 * the library: it has its own dimension validation, conjugation, and basis
 * sign derivation.  It favors clarity over speed and allocates its own
 * scratch storage, so it is used only by tests and benchmark verification.
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
 * Neumaier compensation.  It shares only the basis-sign derivation with the
 * recursive oracle, so a transcription error in the doubling formula cannot
 * be common to both.  O(N^2 log N); intended for dimensions up to 1024.
 */
FastCDStatus reference_cd_mul_signed_f64(double* out, const double* x,
        const double* y, size_t dimension);

#endif
