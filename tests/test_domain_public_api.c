/* Force out-of-line field symbols as well as the new complex/finite API.
 * In shared builds this catches missing export annotations on inline emitters. */
#define FASTCD_FF_FIELD_EXTERN_ONLY
#include "fastcd/finite.h"
#include "fastcd/complex.h"
#include <stdlib.h>

int main(void) {
    FastCDField f;
    FastCDPlanFF p;
    if (fastcd_field_init(&f, 13, 2, NULL) || fastcd_ff_plan_init(&p, &f, 2))
        return EXIT_FAILURE;
    uint32_t a[2] = {2, 0}, b[2] = {3, 0}, c[2], inverse[2];
    fastcd_ff_field_add(c, a, b, &f);
    if (c[0] != 5) return EXIT_FAILURE;
    fastcd_ff_field_sub(c, c, a, &f);
    if (c[0] != 3) return EXIT_FAILURE;
    fastcd_ff_field_mul(c, a, b, &f);
    if (c[0] != 6 || c[1] != 0) return EXIT_FAILURE;
    fastcd_field_neg(c, a, &f);
    if (c[0] != 11 || fastcd_field_inv(inverse, a, &f)) return EXIT_FAILURE;
    fastcd_ff_field_mul(c, a, inverse, &f);
    if (c[0] != 1 || c[1] != 0) return EXIT_FAILURE;
    if (fastcd_ff_field_reduce64(UINT64_MAX, &f) != UINT64_MAX % 13 ||
            fastcd_ff_prime_mul(2, 3, &f) != 6)
        return EXIT_FAILURE;
    uint32_t x[8] = {1}, y[8] = {0, 0, 1}, out[8], q[2];
    size_t len;
    if (fastcd_ff_cd_workspace_size(&p, &len)) return EXIT_FAILURE;
    uint32_t* scratch = malloc(len * sizeof(*scratch));
    if (!scratch) return EXIT_FAILURE;
    FastCDFFStatus status = fastcd_mul_ff(out, x, y, &p, scratch, len);
    if (!status) status = fastcd_quadratic_norm_ff(q, out, &p);
    free(scratch);
    if (status || q[0] != 1 || q[1] != 0) return EXIT_FAILURE;
    FastCDComplexF64 cx[2] = {{1, 1}, {0, 0}}, cy[2] = {{0, 0}, {1, 0}}, co[2], cq;
    if (fastcd_workspace_size_c128(2, FASTCD_OP_MULTIPLY, &len)) return EXIT_FAILURE;
    double* work = malloc(len * sizeof(*work));
    if (!work) return EXIT_FAILURE;
    FastCDWorkspaceF64 w = {work, len};
    FastCDStatus cs = fastcd_mul_c128(co, cx, cy, 2, &w);
    free(work);
    if (cs || co[1].real != 1 || co[1].imag != 1) return EXIT_FAILURE;
    if (fastcd_quadratic_norm_c128(&cq, co, 2) || cq.real != 0 || cq.imag != 2)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
