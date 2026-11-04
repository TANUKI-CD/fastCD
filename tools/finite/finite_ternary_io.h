/* Source-generation fragment for native ternary conversion and transforms. */
static Word encode_values(
        const uint32_t* values, __m512i one_table, __m512i two_table, __m512i index) {
    __m512i x =
            _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), 0xff, index, values, 4);
    x = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(x));
    return (Word){_mm512_permutexvar_epi64(x, one_table),
            _mm512_permutexvar_epi64(x, two_table)};
}

static __m512i population_vector(__m512i a) {
    static const uint8_t bits[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
    __m512i table = _mm512_broadcast_i32x4(_mm_loadu_si128((const void*)bits)),
            mask = _mm512_set1_epi8(15);
    __m512i lo = _mm512_shuffle_epi8(table, _mm512_and_si512(a, mask));
    __m512i hi =
            _mm512_shuffle_epi8(table, _mm512_and_si512(_mm512_srli_epi64(a, 4), mask));
    return _mm512_sad_epu8(_mm512_add_epi8(lo, hi), _mm512_setzero_si512());
}

static __m512i modulo_three(__m512i a) {
    static const uint8_t values[16] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
    __m512i table = _mm512_broadcast_i32x4(_mm_loadu_si128((const void*)values));
    __m512i lo = _mm512_shuffle_epi8(table, a),
            hi = _mm512_shuffle_epi8(table, _mm512_srli_epi64(a, 4));
    return _mm512_shuffle_epi8(table, _mm512_add_epi64(lo, hi));
}

static void sh(Coef* data, size_t n) {
    for (size_t base = 0; base < n; base += 8) {
        Word a = ld(data + base);
#define TERNARY_BUTTERFLY(index, mask_value)                                           \
    do {                                                                               \
        Word b = {_mm512_permutexvar_epi64(index, a.one),                              \
                _mm512_permutexvar_epi64(index, a.two)};                               \
        Word s = add(a, b), t = sub(b, a);                                             \
        a = (Word){_mm512_mask_blend_epi64(mask_value, s.one, t.one),                  \
                _mm512_mask_blend_epi64(mask_value, s.two, t.two)};                    \
    } while (0)
        TERNARY_BUTTERFLY(_mm512_setr_epi64(1, 0, 3, 2, 5, 4, 7, 6), 0xaa);
        TERNARY_BUTTERFLY(_mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5), 0xcc);
        TERNARY_BUTTERFLY(_mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3), 0xf0);
#undef TERNARY_BUTTERFLY
        st(data + base, a);
    }
    for (size_t width = 8; width < n; width *= 2)
        for (size_t base = 0; base < n; base += 2 * width)
            for (size_t j = 0; j < width; j += 8) {
                Word a = ld(data + base + j), b = ld(data + base + width + j);
                st(data + base + j, add(a, b));
                st(data + base + width + j, sub(a, b));
            }
}
