/* Arithmetic shared by the scalar and two-product SIMD instantiations.
 * Intentionally included twice with FASTCD_REAL, FASTCD_ZERO, and FASTCD_NAME
 * supplied by the owning translation unit. */

// Copy the standard involution of a half-dimensional operand
static inline void FASTCD_NAME(direct_conjugate)(FASTCD_REAL* restrict out,
        const FASTCD_REAL* x, size_t dimension) {
    out[0] = x[0];
    for (size_t i = 1; i < dimension; i++) out[i] = -x[i];
}

// Evaluate the defining recursion with straight-line leaves
static void FASTCD_NAME(direct_recursive)(FASTCD_REAL* restrict out,
        const FASTCD_REAL* x, const FASTCD_REAL* y, size_t dimension,
        FASTCD_REAL* restrict scratch) {
    size_t half;
    FASTCD_REAL* temporary_input;
    FASTCD_REAL* temporary_product;
    FASTCD_REAL* child_scratch;

    switch (dimension) {
        case 1: FASTCD_NAME(direct_mul_1)(out, x, y); return;
        case 2: FASTCD_NAME(direct_mul_2)(out, x, y); return;
        case 4: FASTCD_NAME(direct_mul_4)(out, x, y); return;
        case 8: FASTCD_NAME(direct_mul_8)(out, x, y); return;
#if FASTCD_DIRECT_LEAF >= 16
        case 16: FASTCD_NAME(direct_mul_16)(out, x, y); return;
#endif
#if FASTCD_DIRECT_LEAF >= 32
        case 32: FASTCD_NAME(direct_mul_32)(out, x, y); return;
#endif
        default: break;
    }

    half = dimension >> 1;
    temporary_input = scratch;
    temporary_product = scratch + half;
    child_scratch = scratch + dimension;

    FASTCD_NAME(direct_recursive)(out, x, y, half, child_scratch);
    FASTCD_NAME(direct_conjugate)(temporary_input, y + half, half);
    FASTCD_NAME(direct_recursive)(temporary_product, temporary_input, x + half,
            half, child_scratch);
    for (size_t i = 0; i < half; i++) out[i] -= temporary_product[i];

    FASTCD_NAME(direct_recursive)(out + half, y + half, x, half, child_scratch);
    FASTCD_NAME(direct_conjugate)(temporary_input, y, half);
    FASTCD_NAME(direct_recursive)(temporary_product, x + half, temporary_input,
            half, child_scratch);
    for (size_t i = 0; i < half; i++) out[half + i] += temporary_product[i];
}
