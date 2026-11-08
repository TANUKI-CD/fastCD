/* Universal small-characteristic CRT maps, reduced for the field's polynomial. */
#include "finite_maps.h"

// Scatter up to 32 packed 16-bit values down one column of a map matrix
static void scatter_map(
        uint32_t matrix[48][12], size_t first, size_t col, size_t count, Word value) {
    Word index = _mm512_setr_epi32(
            0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180);
    __mmask16 lo = count >= 16 ? 0xffff : (__mmask16)((1u << count) - 1);
    _mm512_mask_i32scatter_epi32(&matrix[first][col], lo, index,
            _mm512_cvtepu16_epi32(_mm512_castsi512_si256(value)), 4);
    if (count > 16) {
        __mmask16 hi = count >= 32 ? 0xffff : (__mmask16)((1u << (count - 16)) - 1);
        _mm512_mask_i32scatter_epi32(&matrix[first + 16][col], hi, index,
                _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(value, 1)), 4);
    }
}

// Reduce the universal small-prime CRT maps into the field basis and return the rank
size_t fastcd_ff_prepare_small(
        uint32_t input[48][12], uint32_t output[48][12], const FastCDField* f) {
    const FastCDFFSmallMap* maps = &fastcd_ff_small_maps[f->prime == 3 ? 0
                    : f->prime == 5                               ? 1
                                                                  : 2][f->degree - 1];
    uint32_t reduced[25][12];
    if (f->prime == 3)
        fastcd_ff_small_remainders(reduced, f, maps->total, 3);
    else if (f->prime == 5)
        fastcd_ff_small_remainders(reduced, f, maps->total, 5);
    else
        fastcd_ff_small_remainders(reduced, f, maps->total, 7);
    FastCDPlanFF p = {0};
    p.field = *f;
    p.inverse_two = (f->prime + 1) / 2;
    p.inverse_dimension = 1;
    Context c = context(&p);
    memset(input, 0, 48 * 12 * sizeof(uint32_t));
    memset(output, 0, 48 * 12 * sizeof(uint32_t));
    for (size_t first = 0; first < maps->rank; first += 32) {
        Word basis[25];
        size_t count = maps->rank - first;
        if (count > 32) count = 32;
        for (size_t k = 0; k < maps->total; k++)
            basis[k] = _mm512_cvtepu8_epi16(
                    _mm256_loadu_si256((const void*)(maps->output[k] + first)));
        for (size_t j = 0; j < f->degree; j++) {
            Word a = _mm512_cvtepu8_epi16(
                    _mm256_loadu_si256((const void*)(maps->input[j] + first)));
            scatter_map(input, first, j, count, a);
            Word value = scalar(reduced[0][j], basis[0], &c);
            for (size_t k = 1; k < maps->total; k++)
                value = add(value, scalar(reduced[k][j], basis[k], &c), &c);
            scatter_map(output, first, j, count, value);
        }
    }
    return maps->rank;
}
