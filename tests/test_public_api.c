#include <stddef.h>
#include <stdlib.h>

#include "fastcd/fastcd.h"

/*
 * Link and execute a small public-only call chain.  In shared-library builds
 * this checks that the installed API symbols remain exported.
 */
// Execute a public-only construction and multiplication call chain
int main(void) {
    double x[16];
    double y[16];
    double out[16];
    double* scratch = NULL;
    size_t dimension = 0;
    size_t length = 0;
    FastCDWorkspaceF64 workspace;
    FastCDAlgorithm algorithm;
    FastCDStatus status;

    if (fastcd_version_string() == NULL) return EXIT_FAILURE;
    if (fastcd_dimension(4, &dimension) != FASTCD_SUCCESS || dimension != 16) {
        return EXIT_FAILURE;
    }
    if (fastcd_default_algorithm(dimension, &algorithm) != FASTCD_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (fastcd_algorithm_string(algorithm) == NULL) return EXIT_FAILURE;
    if (fastcd_mul_workspace_size(dimension, &length) != FASTCD_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (length != 0) {
        scratch = malloc(length * sizeof(*scratch));
        if (scratch == NULL) return EXIT_FAILURE;
    }
    workspace.data = scratch;
    workspace.length = length;
    if (fastcd_scalar_f64(x, dimension, 2.0) != FASTCD_SUCCESS
            || fastcd_basis_f64(y, dimension, 1) != FASTCD_SUCCESS) {
        free(scratch);
        return EXIT_FAILURE;
    }
    status = fastcd_mul_f64(out, x, y, dimension,
            length == 0 ? NULL : &workspace);
    free(scratch);
    if (status != FASTCD_SUCCESS) return EXIT_FAILURE;
    for (size_t i = 0; i < dimension; i++) {
        if (out[i] != (i == 1 ? 2.0 : 0.0)) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
