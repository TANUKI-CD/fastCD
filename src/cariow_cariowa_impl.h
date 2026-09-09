/* Table-driven Cariow--Cariowa arithmetic shared by the scalar and
 * two-product SIMD instantiations.  Intentionally included twice with
 * FASTCD_REAL, FASTCD_ZERO, and FASTCD_NAME supplied by the owning
 * translation unit, after the generated straight-line kernels.
 *
 * The operations are those the generator unrolls: Walsh--Hadamard
 * transforms with butterflies in increasing length, the pointwise product,
 * a third transform, the exact 1/N scaling, and for every output the sum of
 * the terms selected by its sign-correction row, doubled and subtracted;
 * the scalar coordinate is finally negated.  The correction pattern is read
 * from a table at run time, and the terms of an output are summed as
 * described below rather than in one sequential chain.  The kernel uses 2N
 * scratch coefficients for the two transformed operands.
 *
 * Each output's row is first expanded into the list of its selected indices
 * with byte tables and no data-dependent branches; the terms are then
 * accumulated round-robin into FASTCD_CC_TABLE_CHAINS partial sums, in
 * increasing index order within each partial sum, and the partial sums are
 * combined pairwise.  This is the same set of operations with a different
 * addition order than the unrolled kernels, whose short chains overlap
 * across outputs; single and batched products both use this kernel above
 * the unrolled dimensions and agree bitwise with each other. */

#ifndef FASTCD_CC_TABLE_CHAINS
#define FASTCD_CC_TABLE_CHAINS 4
#endif
#if FASTCD_CC_TABLE_CHAINS != 4 && FASTCD_CC_TABLE_CHAINS != 8
#error "FASTCD_CC_TABLE_CHAINS must be 4 or 8"
#endif

// Apply the unnormalized Walsh--Hadamard transform in place
static void FASTCD_NAME(cc_wht)(FASTCD_REAL* restrict a, size_t dimension) {
    for (size_t length = 1; length < dimension; length <<= 1) {
        for (size_t base = 0; base < dimension; base += 2 * length) {
            for (size_t offset = 0; offset < length; offset++) {
                FASTCD_REAL left = a[base + offset];
                FASTCD_REAL right = a[base + offset + length];

                a[base + offset] = left + right;
                a[base + offset + length] = left - right;
            }
        }
    }
}

// Accumulate one selected term of output k into a partial sum
#define FASTCD_CC_TERM(partial, t) \
    do { \
        size_t index = base + positions[t]; \
        partial = partial + x[index] * y[index ^ k]; \
    } while (0)

// Evaluate one product from its sign-correction table without validation
static void FASTCD_NAME(cc_table_mul)(FASTCD_REAL* restrict out,
        const FASTCD_REAL* x, const FASTCD_REAL* y, size_t dimension,
        const uint64_t* masks, FASTCD_REAL* restrict scratch) {
    FASTCD_REAL* transformed_x = scratch;
    FASTCD_REAL* transformed_y = scratch + dimension;
    const size_t words = dimension > 64 ? dimension / 64 : 1;
    const double inverse = 1.0 / (double)dimension;
    uint8_t positions[64 + 8];

    for (size_t i = 0; i < dimension; i++) {
        transformed_x[i] = x[i];
        transformed_y[i] = y[i];
    }
    FASTCD_NAME(cc_wht)(transformed_x, dimension);
    FASTCD_NAME(cc_wht)(transformed_y, dimension);
    for (size_t i = 0; i < dimension; i++) {
        transformed_x[i] = transformed_x[i] * transformed_y[i];
    }
    FASTCD_NAME(cc_wht)(transformed_x, dimension);
    for (size_t i = 0; i < dimension; i++) {
        transformed_x[i] = transformed_x[i] * inverse;
    }

    for (size_t k = 0; k < dimension; k++) {
        const uint64_t* row = masks + k * words;
        FASTCD_REAL p0 = FASTCD_ZERO;
        FASTCD_REAL p1 = FASTCD_ZERO;
        FASTCD_REAL p2 = FASTCD_ZERO;
        FASTCD_REAL p3 = FASTCD_ZERO;
#if FASTCD_CC_TABLE_CHAINS == 8
        FASTCD_REAL p4 = FASTCD_ZERO;
        FASTCD_REAL p5 = FASTCD_ZERO;
        FASTCD_REAL p6 = FASTCD_ZERO;
        FASTCD_REAL p7 = FASTCD_ZERO;
#endif
        FASTCD_REAL correction;
        FASTCD_REAL result;

        for (size_t w = 0; w < words; w++) {
            size_t base = 64 * w;
            size_t count = cc_expand_word(row[w], positions);
            size_t t = 0;

            for (; t + FASTCD_CC_TABLE_CHAINS <= count;
                    t += FASTCD_CC_TABLE_CHAINS) {
                FASTCD_CC_TERM(p0, t);
                FASTCD_CC_TERM(p1, t + 1);
                FASTCD_CC_TERM(p2, t + 2);
                FASTCD_CC_TERM(p3, t + 3);
#if FASTCD_CC_TABLE_CHAINS == 8
                FASTCD_CC_TERM(p4, t + 4);
                FASTCD_CC_TERM(p5, t + 5);
                FASTCD_CC_TERM(p6, t + 6);
                FASTCD_CC_TERM(p7, t + 7);
#endif
            }
            for (; t < count; t++) FASTCD_CC_TERM(p0, t);
        }
#if FASTCD_CC_TABLE_CHAINS == 8
        correction = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7));
#else
        correction = (p0 + p1) + (p2 + p3);
#endif
        result = transformed_x[k] - correction * 2.0;
        out[k] = k == 0 ? -result : result;
    }
}

#undef FASTCD_CC_TERM
