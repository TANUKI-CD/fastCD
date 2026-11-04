#include "fastcd/arithmetic.h"

// Expose the quadratic-form name using the original scaled real norm path
FastCDStatus fastcd_quadratic_norm_f64(
        double* value, const double* x, size_t dimension) {
    return fastcd_norm_squared_f64(value, x, dimension);
}
