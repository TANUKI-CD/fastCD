/*
 * Fixed-degree direct CD recursion, mirroring src/direct_impl.h.
 * Included with FASTCD_FF_DEGREE and FASTCD_FF_NAME; intentionally unguarded
 * so every supported polynomial degree gets a compile-time specialization.
 * Inputs are canonical records and writable buffers are already disjoint.
 */
// Evaluate (a,b)(c,d) = (ac - conjugate(d)b, da + b conjugate(c))
static void FASTCD_FF_NAME(direct)(uint32_t* out, const uint32_t* x, const uint32_t* y,
        size_t n, const FastCDField* f, uint32_t* w) {
    const size_t d = FASTCD_FF_DEGREE;
    if (n <= 4) {
        // Expand the real/complex/quaternion basis signs with exact field
        // products. The degree is constant throughout this instantiation.
        // For output k, XOR fixes the partner j=i^k. Each polynomial product
        // reduces before accumulation, so even p near 2^31 stays canonical.
        // Initialize from the first term instead of adding it to zero.
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
    // Reserve one conjugated half-record, one partial product, and then
    // shared child scratch. All four recursive calls run sequentially.
    // R(N)=N*d+R(N/2) is less than 2*N*d words, the bound in plan.c.
    uint32_t* temp = w;
    uint32_t* product = w + words;
    uint32_t* child = w + 2 * words;
    // Lower half: ac - conjugate(d)b. Only imaginary CD coordinates negate;
    // each polynomial coefficient is copied or negated as one field scalar.
    FASTCD_FF_NAME(direct)(out, x, y, h, f, child);
    memcpy(temp, y + words, d * sizeof(uint32_t));
    for (size_t i = 1; i < h; i++) fastcd_field_neg(temp + i * d, y + words + i * d, f);
    FASTCD_FF_NAME(direct)(product, temp, x + words, h, f, child);
    // Consume this partial product before its storage is reused by the
    // upper-half recursion; neither temporary aliases a public operand.
    for (size_t i = 0; i < words; i++) out[i] = fastcd_ff_sub(out[i], product[i], f);
    // Upper half: da + b conjugate(c), preserving this multiplication order.
    FASTCD_FF_NAME(direct)(out + words, y + words, x, h, f, child);
    // The first d words are one scalar coefficient, not d CD coordinates.
    memcpy(temp, y, d * sizeof(uint32_t));
    for (size_t i = 1; i < h; i++) fastcd_field_neg(temp + i * d, y + i * d, f);
    FASTCD_FF_NAME(direct)(product, x + words, temp, h, f, child);
    for (size_t i = 0; i < words; i++)
        out[words + i] = fastcd_ff_add(out[words + i], product[i], f);
}
