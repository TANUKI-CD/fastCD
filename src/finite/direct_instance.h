/*
 * Fixed-degree direct CD recursion over F_(p^d), following src/direct_impl.h.
 * The file is included once per degree with FASTCD_FF_DEGREE and
 * FASTCD_FF_NAME, so it has no include guard. Inputs are canonical records, and
 * writable buffers are disjoint from them.
 */
// Evaluate (a,b)(c,d) = (ac - conjugate(d)b, da + b conjugate(c))
static void FASTCD_FF_NAME(direct)(uint32_t* out, const uint32_t* x, const uint32_t* y,
        size_t n, const FastCDField* f, uint32_t* w) {
    const size_t d = FASTCD_FF_DEGREE;
    if (n <= 4) {
        /*
         * Expand the real, complex and quaternion basis signs with exact field
         * products of the fixed degree. Output k pairs coordinate i with
         * j = i ^ k. Each polynomial product is reduced before accumulation, so
         * sums stay canonical for p near 2^31, and the first term initializes
         * each output coefficient.
         */
        uint32_t product[FASTCD_FF_DEGREE];
        for (size_t k = 0; k < n; k++) {
            for (size_t i = 0; i < n; i++) {
                size_t j = i ^ k;
                int positive = i == 0 || j == 0 || (i != j && i % 3 + 1 == j);
                FASTCD_FF_JOIN(fastcd_ff_product, FASTCD_FF_DEGREE)(product, x + i * d, y + j * d, f);
                if (i == 0)
                    memcpy(out + k * d, product, d * sizeof(uint32_t));
                else if (positive)
                    for (size_t l = 0; l < d; l++)
                        out[k * d + l] = fastcd_ff_add(out[k * d + l], product[l], f);
                else
                    for (size_t l = 0; l < d; l++)
                        out[k * d + l] = fastcd_ff_sub(out[k * d + l], product[l], f);
            }
        }
        return;
    }
    size_t h = n / 2, words = h * d;
    /*
     * Scratch holds one conjugated half-record, one partial product and the
     * children's shared scratch; the four recursive calls run in sequence.
     * R(N) = N*d + R(N/2) stays below the 2*N*d words reserved in plan.c.
     */
    uint32_t* temp = w;
    uint32_t* product = w + words;
    uint32_t* child = w + 2 * words;
    // Lower half: ac - conjugate(d)b, negating only imaginary CD coordinates
    // and copying or negating each coefficient as one field scalar
    FASTCD_FF_NAME(direct)(out, x, y, h, f, child);
    memcpy(temp, y + words, d * sizeof(uint32_t));
    for (size_t i = 1; i < h; i++) fastcd_field_neg(temp + i * d, y + words + i * d, f);
    FASTCD_FF_NAME(direct)(product, temp, x + words, h, f, child);
    /*
     * Use the partial product before the upper-half recursion reuses its
     * storage. Neither temporary aliases a public operand.
     */
    for (size_t i = 0; i < words; i++) out[i] = fastcd_ff_sub(out[i], product[i], f);
    // Upper half: da + b conjugate(c), keeping this multiplication order
    FASTCD_FF_NAME(direct)(out + words, y + words, x, h, f, child);
    // The first d words are one scalar coefficient, not d CD coordinates
    memcpy(temp, y, d * sizeof(uint32_t));
    for (size_t i = 1; i < h; i++) fastcd_field_neg(temp + i * d, y + i * d, f);
    FASTCD_FF_NAME(direct)(product, x + words, temp, h, f, child);
    for (size_t i = 0; i < words; i++)
        out[words + i] = fastcd_ff_add(out[words + i], product[i], f);
}
