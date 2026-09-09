/* Arithmetic shared by the scalar and two-product SIMD instantiations.
 * Intentionally included twice with FASTCD_REAL, FASTCD_ZERO, and FASTCD_NAME
 * supplied by the owning translation unit. */

typedef struct FASTCD_NAME(FastCDComplex) {
    FASTCD_REAL real;
    FASTCD_REAL imag;
} FASTCD_NAME(FastCDComplex);

// Add two complex coefficients
static inline FASTCD_NAME(FastCDComplex) FASTCD_NAME(complex_add)(
        FASTCD_NAME(FastCDComplex) x, FASTCD_NAME(FastCDComplex) y) {
    FASTCD_NAME(FastCDComplex) result = {x.real + y.real, x.imag + y.imag};
    return result;
}

// Subtract two complex coefficients
static inline FASTCD_NAME(FastCDComplex) FASTCD_NAME(complex_sub)(
        FASTCD_NAME(FastCDComplex) x, FASTCD_NAME(FastCDComplex) y) {
    FASTCD_NAME(FastCDComplex) result = {x.real - y.real, x.imag - y.imag};
    return result;
}

// Multiply two complex coefficients with three real products
static inline FASTCD_NAME(FastCDComplex) FASTCD_NAME(complex_mul)(
        FASTCD_NAME(FastCDComplex) x, FASTCD_NAME(FastCDComplex) y) {
    FASTCD_REAL p = x.real * y.real;
    FASTCD_REAL q = x.imag * y.imag;
    FASTCD_REAL r = (x.real + x.imag) * (y.real + y.imag);
    FASTCD_NAME(FastCDComplex) result = {p - q, (r - p) - q};
    return result;
}

// Halve both planes of a complex coefficient
static inline FASTCD_NAME(FastCDComplex) FASTCD_NAME(complex_half)(
        FASTCD_NAME(FastCDComplex) x) {
    FASTCD_NAME(FastCDComplex) result = {x.real * 0.5, x.imag * 0.5};
    return result;
}

/*
 * Compute the complex-bilinear imaginary alternating product.  Index zero of
 * every input plane is ignored.  Four transformed half-size input planes
 * occupy 2K real scratch slots; the two children are evaluated sequentially
 * into the lower and upper halves of the output planes, reusing those input
 * planes and one child workspace.
 * Paper notation: AltC_n, with dimension = 2^n.  The real/imag planes are
 * coefficients over C; iota is the auxiliary complex unit, not a CD basis
 * unit.  gamma and delta are complex scalars in the half-dimensional algebra.
 */
// Compute the complex alternating product of dimension K without validation
static void FASTCD_NAME(complex_cross)(FASTCD_REAL* restrict out_real,
        FASTCD_REAL* restrict out_imag,
        const FASTCD_REAL* x_real, const FASTCD_REAL* x_imag,
        const FASTCD_REAL* y_real, const FASTCD_REAL* y_imag, size_t dimension,
        FASTCD_REAL* restrict workspace) {
    size_t half;
    FASTCD_REAL* transformed_x_real;
    FASTCD_REAL* transformed_x_imag;
    FASTCD_REAL* transformed_y_real;
    FASTCD_REAL* transformed_y_imag;
    FASTCD_REAL* child_workspace;
    FASTCD_NAME(FastCDComplex) gamma;
    FASTCD_NAME(FastCDComplex) delta;
    FASTCD_NAME(FastCDComplex) scalar = {FASTCD_ZERO, FASTCD_ZERO};

    // AltC_1 = 0 (dimension 2); dimension 1 is also zero.  Larger leaves
    // expand the same AltC recurrence into generated straight-line code.
    switch (dimension) {
        case 1:
            out_real[0] = FASTCD_ZERO;
            out_imag[0] = FASTCD_ZERO;
            return;
        case 2:
            FASTCD_NAME(ql_complex_cross_2)(out_real, out_imag, x_real, x_imag,
                    y_real, y_imag);
            return;
        case 4:
            FASTCD_NAME(ql_complex_cross_4)(out_real, out_imag, x_real, x_imag,
                    y_real, y_imag);
            return;
#if FASTCD_QL_COMPLEX_LEAF >= 8
        case 8:
            FASTCD_NAME(ql_complex_cross_8)(out_real, out_imag, x_real, x_imag,
                    y_real, y_imag);
            return;
#endif
#if FASTCD_QL_COMPLEX_LEAF >= 16
        case 16:
            FASTCD_NAME(ql_complex_cross_16)(out_real, out_imag, x_real, x_imag,
                    y_real, y_imag);
            return;
#endif
        default:
            break;
    }

    // Reserve four half-dimensional input planes and scratch for AltC_{n-1}.
    half = dimension >> 1;
    transformed_x_real = workspace;
    transformed_x_imag = workspace + half;
    transformed_y_real = workspace + dimension;
    transformed_y_imag = workspace + dimension + half;
    child_workspace = workspace + 2 * dimension;
    // Write X = (u, gamma + s), Y = (v, delta + t); u, v, s, t have
    // zero CD scalar coordinate.  Extract gamma and delta from the upper halves.
    gamma.real = x_real[half];
    gamma.imag = x_imag[half];
    delta.real = y_real[half];
    delta.imag = y_imag[half];

    // Keep the transformed operands in V_{n-1}^C: their scalar coordinate is 0.
    transformed_x_real[0] = FASTCD_ZERO;
    transformed_x_imag[0] = FASTCD_ZERO;
    transformed_y_real[0] = FASTCD_ZERO;
    transformed_y_imag[0] = FASTCD_ZERO;
    // Form u + iota s and v + iota t in separate complex coefficient planes.
    for (size_t i = 1; i < half; i++) {
        transformed_x_real[i] = x_real[i] - x_imag[half + i];
        transformed_x_imag[i] = x_imag[i] + x_real[half + i];
        transformed_y_real[i] = y_real[i] - y_imag[half + i];
        transformed_y_imag[i] = y_imag[i] + y_real[half + i];
    }
    // Z_+ = AltC_{n-1}(u + iota s, v + iota t); store in the lower halves.
    FASTCD_NAME(complex_cross)(out_real, out_imag, transformed_x_real,
            transformed_x_imag, transformed_y_real, transformed_y_imag, half,
            child_workspace);

    // Reuse the input planes for u - iota s and v - iota t; index 0 stays zero.
    for (size_t i = 1; i < half; i++) {
        transformed_x_real[i] = x_real[i] + x_imag[half + i];
        transformed_x_imag[i] = x_imag[i] - x_real[half + i];
        transformed_y_real[i] = y_real[i] + y_imag[half + i];
        transformed_y_imag[i] = y_imag[i] - y_real[half + i];
    }
    // Z_- = AltC_{n-1}(u - iota s, v - iota t); store in the upper halves.
    FASTCD_NAME(complex_cross)(out_real + half, out_imag + half, transformed_x_real,
            transformed_x_imag, transformed_y_real, transformed_y_imag, half,
            child_workspace);

    // Accumulate lambda and assemble the non-scalar coordinates of F and G.
    for (size_t i = 1; i < half; i++) {
        // Read coordinate i of u, v, s, t and the two child results Z_+, Z_-.
        FASTCD_NAME(FastCDComplex) u = {x_real[i], x_imag[i]};
        FASTCD_NAME(FastCDComplex) v = {y_real[i], y_imag[i]};
        FASTCD_NAME(FastCDComplex) s = {x_real[half + i], x_imag[half + i]};
        FASTCD_NAME(FastCDComplex) t = {y_real[half + i], y_imag[half + i]};
        FASTCD_NAME(FastCDComplex) plus = {out_real[i], out_imag[i]};
        FASTCD_NAME(FastCDComplex) minus = {out_real[half + i], out_imag[half + i]};
        // Contribution to lambda = <s,v>_bil - <t,u>_bil; no conjugation.
        FASTCD_NAME(FastCDComplex) pair = FASTCD_NAME(complex_sub)(
                FASTCD_NAME(complex_mul)(s, v), FASTCD_NAME(complex_mul)(t, u));
        // Start F with (Z_+ + Z_-)/2.
        FASTCD_NAME(FastCDComplex) first = FASTCD_NAME(complex_half)(
                FASTCD_NAME(complex_add)(plus, minus));
        // mixed = iota (Z_+ - Z_-)/2; iota(a + iota b) = -b + iota a.
        FASTCD_NAME(FastCDComplex) difference = FASTCD_NAME(complex_half)(
                FASTCD_NAME(complex_sub)(plus, minus));
        FASTCD_NAME(FastCDComplex) mixed = {-difference.imag, difference.real};
        // Start the non-scalar part of G with delta u - gamma v.
        FASTCD_NAME(FastCDComplex) second = FASTCD_NAME(complex_sub)(
                FASTCD_NAME(complex_mul)(delta, u),
                FASTCD_NAME(complex_mul)(gamma, v));

        // Sum lambda, then finish F and G with their remaining terms.
        scalar = i == 1 ? pair : FASTCD_NAME(complex_add)(scalar, pair);
        first = FASTCD_NAME(complex_add)(first, FASTCD_NAME(complex_mul)(gamma, t));
        first = FASTCD_NAME(complex_sub)(first, FASTCD_NAME(complex_mul)(delta, s));
        second = FASTCD_NAME(complex_add)(second, mixed);
        // Replace Z_+, Z_- with coordinate i of F, G in the output halves.
        out_real[i] = first.real;
        out_imag[i] = first.imag;
        out_real[half + i] = second.real;
        out_imag[half + i] = second.imag;
    }
    // Return (F,G): F has scalar coordinate 0 and G has scalar coordinate lambda.
    out_real[0] = FASTCD_ZERO;
    out_imag[0] = FASTCD_ZERO;
    out_real[half] = scalar.real;
    out_imag[half] = scalar.imag;
}

/*
 * A real root requires one complex evaluation: the operand halves are its
 * real and imaginary planes, and the two output halves hold the complex
 * result until recombination.
 */
// Compute the quasilinear imaginary cross product without validation
static void FASTCD_NAME(ql_cross_recursive)(FASTCD_REAL* restrict out,
        const FASTCD_REAL* x, const FASTCD_REAL* y, size_t dimension,
        FASTCD_REAL* restrict workspace) {
    size_t half;
    FASTCD_REAL gamma;
    FASTCD_REAL delta;
    FASTCD_REAL scalar = FASTCD_ZERO;

    // AltR_1 = 0 (dimension 2); dimension 1 is also zero.  Larger leaves
    // expand the same AltR recurrence into generated straight-line code.
    switch (dimension) {
        case 1:
            out[0] = FASTCD_ZERO;
            return;
        case 2:
            out[0] = FASTCD_ZERO;
            out[1] = FASTCD_ZERO;
            return;
        case 4:
            FASTCD_NAME(ql_cross_4)(out, x, y);
            return;
        case 8:
            FASTCD_NAME(ql_cross_8)(out, x, y);
            return;
#if FASTCD_QL_FULL_LEAF >= 16
        case 16:
            FASTCD_NAME(ql_cross_16)(out, x, y);
            return;
#endif
#if FASTCD_QL_FULL_LEAF >= 32
        case 32:
            FASTCD_NAME(ql_cross_32)(out, x, y);
            return;
#endif
        default:
            break;
    }

    // AltR_n: write X = (u, gamma + s), Y = (v, delta + t).
    half = dimension >> 1;
    gamma = x[half];
    delta = y[half];
    // Z = AltC_{n-1}(u + iota s, v + iota t).  The input halves supply
    // the complex planes; ignored index 0 omits gamma, delta and x[0], y[0].
    // Store Re_C Z in the lower output half and Im_C Z in the upper half.
    FASTCD_NAME(complex_cross)(out, out + half, x, x + half, y, y + half, half,
            workspace);
    // lambda = <s,v> - <t,u>; F = Re_C Z + gamma t - delta s;
    // the non-scalar part of G is delta u - gamma v - Im_C Z.
    for (size_t i = 1; i < half; i++) {
        FASTCD_REAL pair = x[half + i] * y[i] - y[half + i] * x[i];
        FASTCD_REAL first = out[i] + gamma * y[half + i];
        FASTCD_REAL second = delta * x[i] - gamma * y[i];

        scalar = i == 1 ? pair : scalar + pair;
        first = first - delta * x[half + i];
        second = second - out[half + i];
        out[i] = first;
        out[half + i] = second;
    }
    // Return (F,G), placing lambda in the scalar coordinate of G.
    out[0] = FASTCD_ZERO;
    out[half] = scalar;
}

// Add the scalar--imaginary terms to a completed imaginary cross product
static void FASTCD_NAME(reconstruct)(FASTCD_REAL* restrict out,
        const FASTCD_REAL* x, const FASTCD_REAL* y, size_t dimension) {
    // Mul_n: alpha = RePart(x), beta = RePart(y); coordinates 1..N-1
    // hold X = ImPart(x), Y = ImPart(y).  out already contains Z = AltR_n(X,Y).
    FASTCD_REAL alpha = x[0];
    FASTCD_REAL beta = y[0];
    FASTCD_REAL dot = x[1] * y[1];

    // Compute <X,Y> and write the scalar part alpha beta - <X,Y>.
    for (size_t i = 2; i < dimension; i++) dot = dot + x[i] * y[i];
    out[0] = alpha * beta - dot;
    // Complete the imaginary part Z + alpha Y + beta X.
    for (size_t i = 1; i < dimension; i++) {
        out[i] = out[i] + (alpha * y[i] + beta * x[i]);
    }
}

// Evaluate one complete quasilinear product without validation
static void FASTCD_NAME(ql_mul_recursive)(FASTCD_REAL* restrict out,
        const FASTCD_REAL* x, const FASTCD_REAL* y, size_t dimension,
        FASTCD_REAL* restrict workspace) {
    // Mul_0 is ordinary real multiplication (dimension 1); larger leaves
    // evaluate the complete Mul_n formula in generated straight-line code.
    switch (dimension) {
        case 1:
            FASTCD_NAME(ql_mul_1)(out, x, y);
            return;
        case 2:
            FASTCD_NAME(ql_mul_2)(out, x, y);
            return;
        case 4:
            FASTCD_NAME(ql_mul_4)(out, x, y);
            return;
        case 8:
            FASTCD_NAME(ql_mul_8)(out, x, y);
            return;
#if FASTCD_QL_FULL_LEAF >= 16
        case 16:
            FASTCD_NAME(ql_mul_16)(out, x, y);
            return;
#endif
#if FASTCD_QL_FULL_LEAF >= 32
        case 32:
            FASTCD_NAME(ql_mul_32)(out, x, y);
            return;
#endif
        default:
            break;
    }
    // Z = AltR_n(ImPart(x), ImPart(y)); the cross product ignores x[0], y[0].
    FASTCD_NAME(ql_cross_recursive)(out, x, y, dimension, workspace);
    // Return (alpha beta - <X,Y>) + alpha Y + beta X + Z.
    FASTCD_NAME(reconstruct)(out, x, y, dimension);
}
