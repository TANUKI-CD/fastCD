/* Lagrange reconstruction for linear evaluation points. The numerators are
 * universal integers, and only their reduction, the factorial inverses and the
 * field's monic polynomial depend on the call. */
#include "finite_maps.h"

// Invert a nonzero prime-field residue by Fermat's little theorem
static uint32_t map_inverse(uint32_t a, const FastCDField* f) {
    uint32_t result = 1, e = f->prime - 2;
    while (e) {
        if (e & 1) result = fastcd_ff_prime_mul(result, a, f);
        a = fastcd_ff_prime_mul(a, a, f);
        e >>= 1;
    }
    return result;
}

// Build the Lagrange reconstruction weights reduced into the field basis
void fastcd_ff_prepare_linear(uint32_t output[48][12], const FastCDField* f) {
    size_t d = f->degree, rank = 2 * d - 1, groups = (rank + 15) / 16,
           offset = fastcd_ff_linear_offset[d - 1];
    uint32_t prime = f->prime, r64 = (uint32_t)(UINT64_C(0) - f->reciprocal * prime);
    uint32_t factorial[23], inverse[23], reduced[25][12] = {{0}};
    factorial[0] = 1;
    for (size_t i = 1; i < rank; i++)
        factorial[i] = fastcd_ff_prime_mul(factorial[i - 1], (uint32_t)i, f);
    inverse[rank - 1] = map_inverse(factorial[rank - 1], f);
    for (size_t i = rank - 1; i > 0; i--)
        inverse[i - 1] = fastcd_ff_prime_mul(inverse[i], (uint32_t)i, f);
    for (size_t k = 0; k < d; k++) reduced[k][k] = 1;
    for (size_t k = d; k < rank; k++) {
        uint32_t carry = reduced[k - 1][d - 1];
        for (size_t j = 0; j < d; j++) {
            uint32_t value = fastcd_ff_prime_mul(carry, f->negative[j], f) +
                    (j ? reduced[k - 1][j - 1] : 0);
            reduced[k][j] = value >= prime ? value - prime : value;
        }
    }
    Context c = {_mm512_set1_epi32((int)prime), _mm512_setzero_si512(),
            _mm512_setzero_si512(), _mm512_setzero_si512()};
    Word basis[23][2];
    for (size_t group = 0; group < groups; group++) {
        uint32_t inv[16], q[16];
        for (size_t lane = 0; lane < 16; lane++) {
            size_t point = group * 16 + lane;
            inv[lane] = 0;
            q[lane] = 0;
            if (point < rank) {
                uint32_t v = fastcd_ff_prime_mul(
                        inverse[point], inverse[rank - 1 - point], f);
                if (point & 1) v = prime - v;
                inv[lane] = v;
                q[lane] = (uint32_t)(((uint64_t)v << 32) / prime);
            }
        }
        Word reciprocal = _mm512_loadu_si512((const void*)inv),
             quotient = _mm512_loadu_si512((const void*)q);
        for (size_t k = 0; k < rank; k++) {
            uint32_t values[16];
            for (size_t lane = 0; lane < 16; lane++) {
                size_t point = group * 16 + lane;
                values[lane] = 0;
                if (point < rank) {
                    size_t at = offset + k * rank + point;
                    uint32_t v = fastcd_ff_field_reduce64(fastcd_ff_linear_low[at], f);
                    uint32_t high = fastcd_ff_field_reduce64(
                            (uint64_t)fastcd_ff_linear_high[at] * r64, f);
                    v += high;
                    values[lane] = v >= prime ? v - prime : v;
                }
            }
            Word v = fixed(
                    _mm512_loadu_si512((const void*)values), reciprocal, quotient, &c);
            if (k & 1) {
                v = sub(_mm512_setzero_si512(), v, &c);
            }
            basis[k][group] = v;
        }
    }
    memset(output, 0, 48 * 12 * sizeof(uint32_t));
    Word index = _mm512_setr_epi32(
            0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180);
    for (size_t j = 0; j < d; j++)
        for (size_t group = 0; group < groups; group++) {
            Word value = basis[j][group];
            for (size_t k = d; k < rank; k++) {
                uint32_t factor = reduced[k][j];
                if (factor) {
                    Word a = _mm512_set1_epi32((int)factor),
                         q = _mm512_set1_epi32((int)(((uint64_t)factor << 32) / prime));
                    value = add(value, fixed(basis[k][group], a, q, &c), &c);
                }
            }
            size_t count = rank - group * 16;
            __mmask16 active = count >= 16 ? 0xffff : (__mmask16)((1u << count) - 1);
            _mm512_mask_i32scatter_epi32(
                    &output[group * 16][j], active, index, value, 4);
        }
}
