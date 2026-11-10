#ifndef FASTCD_ARITHMETIC_H
#define FASTCD_ARITHMETIC_H
#include "fastcd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Operations on one algebra record. INNER and QUADRATIC_NORM produce one
 * coefficient, SCALE and DIVIDE take one coefficient in y, and basis
 * operations use basis_index. All other non-null operands are complete
 * records. */
typedef enum {
    FASTCD_OP_MULTIPLY = 0,
    FASTCD_OP_ADD,
    FASTCD_OP_SUBTRACT,
    FASTCD_OP_NEGATE,
    FASTCD_OP_SCALE,
    FASTCD_OP_DIVIDE,
    FASTCD_OP_CONJUGATE,
    FASTCD_OP_IMAG_PART,
    FASTCD_OP_INNER,
    FASTCD_OP_QUADRATIC_NORM,
    FASTCD_OP_RECIPROCAL,
    FASTCD_OP_SQUARE,
    FASTCD_OP_IMAG_CROSS,
    FASTCD_OP_COMMUTATOR,
    FASTCD_OP_ANTICOMMUTATOR,
    FASTCD_OP_ASSOCIATOR,
    FASTCD_OP_BASIS_LEFT,
    FASTCD_OP_BASIS_RIGHT,
    FASTCD_OP_LEFT_OPERATOR_COMMUTATOR,
    FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR
} FastCDOperation;

/* The real entry is an alias of fastcd_norm_squared_f64, with its scaled
 * accumulation and exactly its validation contract. */
// Compute the algebraic quadratic norm Q(x) = sum_i x_i^2
FASTCD_API FastCDStatus fastcd_quadratic_norm_f64(
        double* value, const double* x, size_t dimension);

#ifdef __cplusplus
}
#endif

#endif
