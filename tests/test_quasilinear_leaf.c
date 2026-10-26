#include <math.h>
#include <stdio.h>

#include "reference_cd.h"

#define FASTCD_REAL double
#define FASTCD_ZERO 0.0
#define FASTCD_NAME(name) test_##name
#include "generated/quasilinear_kernels.h"

// Check the complex leaf directly against four independent real products.
int main(void) {
    size_t checked = 0;
    for (size_t trial = 0; trial < 452; trial++) {
        double xr[8] = {0}, xi[8] = {0}, yr[8] = {0}, yi[8] = {0};
        double real[8], imag[8], ac[8], bd[8], ad[8], bc[8];
        if (trial < 196) {
            size_t left = trial / 14;
            size_t right = trial % 14;
            (left < 7 ? xr : xi)[1 + left % 7] = 1.0;
            (right < 7 ? yr : yi)[1 + right % 7] = 1.0;
        } else {
            // Dyadic dense and nearly parallel operands exercise cancellation.
            for (size_t i = 1; i < 8; i++) {
                xr[i] = (double)((trial * 13 + i * 7) % 101) / 16.0 - 3.0;
                xi[i] = (double)((trial * 17 + i * 3) % 97) / 16.0 - 3.0;
                yr[i] = xr[i] + (double)((trial + i) % 11) / 1024.0;
                yi[i] = xi[i] - (double)((trial + 3 * i) % 13) / 1024.0;
            }
        }
        if (reference_cd_mul_f64(ac, xr, yr, 8) != FASTCD_SUCCESS
                || reference_cd_mul_f64(bd, xi, yi, 8) != FASTCD_SUCCESS
                || reference_cd_mul_f64(ad, xr, yi, 8) != FASTCD_SUCCESS
                || reference_cd_mul_f64(bc, xi, yr, 8) != FASTCD_SUCCESS) return 1;
        // Alternating kernels must ignore the scalar coordinate of each plane.
        xr[0] = xi[0] = yr[0] = yi[0] = NAN;
        test_ql_complex_cross_8(real, imag, xr, xi, yr, yi);
        if (real[0] != 0.0 || imag[0] != 0.0) return 1;
        for (size_t i = 1; i < 8; i++) {
            // All fixtures and intermediates are exactly representable dyadics.
            if (real[i] != ac[i] - bd[i] || imag[i] != ad[i] + bc[i]) {
                fprintf(stderr, "complex octonion leaf mismatch: trial=%zu coordinate=%zu\n",
                        trial, i);
                return 1;
            }
        }
        checked++;
    }
    printf("%zu complex octonion leaf checks passed (196 basis, 256 dense)\n", checked);
    return 0;
}
