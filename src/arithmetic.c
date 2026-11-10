#include "fastcd/arithmetic.h"

// Return the real quadratic form through the scaled norm_squared operation
FastCDStatus fastcd_quadratic_norm_f64(
        double* value, const double* x, size_t dimension) {
    return fastcd_norm_squared_f64(value, x, dimension);
}
