/*
 * Single-product quasilinear kernels for builds with the two-lane SIMD type.
 * This file is included once from quasilinear.c, after the scalar and paired
 * instantiations of quasilinear_impl.h, and only when FASTCD_HAVE_PAIR is set.
 *
 * The top complex call evaluates its two half-dimensional recursive calls as
 * the two lanes of the paired kernel, and does its own O(K) work two
 * coordinates per instruction. Every output coordinate follows the arithmetic
 * schedule of the scalar recursion, with the blocked balanced sums receiving
 * terms in the same order, so single products, batches and non-SIMD builds
 * agree bitwise. The top call needs 6K doubles of pair storage plus the
 * scratch of the paired recursive calls, as fastcd_quasilinear_workspace
 * reports.
 *
 * Paper notation: dimension = 2^n for AltC_n, AltR_n and Mul_n. The real and
 * imag planes hold coefficients over C, and iota is the auxiliary complex
 * unit. SIMD lanes are separate calculations, not real and imaginary parts.
 */

/* Two adjacent doubles of a scalar plane, read and written unaligned. */
typedef double FastCDLanes __attribute__((vector_size(16), aligned(8), may_alias));

typedef struct FastCDComplexLanes {
    FastCDPair real;
    FastCDPair imag;
} FastCDComplexLanes;

// Load two adjacent coordinates of a plane
static inline FastCDPair ql_load2(const double* p) {
    return *(const FastCDLanes*)p;
}

// Store two adjacent coordinates of a plane
static inline void ql_store2(double* p, FastCDPair v) {
    *(FastCDLanes*)p = v;
}

// Pack the first lanes of two pairs
static inline FastCDPair ql_low(FastCDPair a, FastCDPair b) {
    return (FastCDPair){a[0], b[0]};
}

// Pack the second lanes of two pairs
static inline FastCDPair ql_high(FastCDPair a, FastCDPair b) {
    return (FastCDPair){a[1], b[1]};
}

// Broadcast one complex coefficient to both lanes
static inline FastCDComplexLanes ql_broadcast(fastcd_FastCDComplex z) {
    FastCDComplexLanes result = {{z.real, z.real}, {z.imag, z.imag}};
    return result;
}

// Add two complex coefficients in each lane (mirrors complex_add)
static inline FastCDComplexLanes ql_lanes_add(FastCDComplexLanes x,
        FastCDComplexLanes y) {
    FastCDComplexLanes result = {x.real + y.real, x.imag + y.imag};
    return result;
}

// Subtract two complex coefficients in each lane (mirrors complex_sub)
static inline FastCDComplexLanes ql_lanes_sub(FastCDComplexLanes x,
        FastCDComplexLanes y) {
    FastCDComplexLanes result = {x.real - y.real, x.imag - y.imag};
    return result;
}

// Recover the complex vector outputs of three real nonrecursive-map evaluations
static inline FastCDComplexLanes ql_lanes_recover(FastCDPair p,
        FastCDPair q, FastCDPair w) {
    FastCDComplexLanes result = {p - q, (w - p) - q};
    return result;
}

// Halve both planes in each lane (mirrors complex_half)
static inline FastCDComplexLanes ql_lanes_half(FastCDComplexLanes x) {
    FastCDComplexLanes result = {x.real * 0.5, x.imag * 0.5};
    return result;
}

// Return the first pair-aligned address inside a double workspace
static inline FastCDPair* ql_aligned_pairs(double* workspace) {
    if ((uintptr_t)workspace % sizeof(FastCDPair) != 0) workspace++;
    return (FastCDPair*)(void*)workspace;
}

/*
 * Lane 0 holds the input of the plus recursive call and lane 1 that of the
 * minus call, formed with the same operations as the two scalar transform
 * loops.
 */
// Pack the transformed planes of one operand for both recursive calls
static void ql_simd_transform(FastCDPair* restrict t_real,
        FastCDPair* restrict t_imag, const double* real, const double* imag,
        size_t half) {
    size_t i;

    // AltC_n: both u +/- iota s (or v +/- iota t) belong to V_{n-1}^C, so
    // their CD scalar coordinates are zero in both lanes
    t_real[0] = (FastCDPair){0.0, 0.0};
    t_imag[0] = (FastCDPair){0.0, 0.0};
    // Form Re_C(u +/- iota s) = Re_C u -/+ Im_C s, two coordinates at a time
    for (i = 1; i + 1 < half; i += 2) {
        FastCDPair a = ql_load2(real + i);
        FastCDPair b = ql_load2(imag + half + i);
        FastCDPair plus_child = a - b;
        FastCDPair minus_child = a + b;

        // Transpose coordinate lanes into call lanes [plus_i, minus_i]
        t_real[i] = ql_low(plus_child, minus_child);
        t_real[i + 1] = ql_high(plus_child, minus_child);
        // Im_C(u +/- iota s) = Im_C u +/- Re_C s, in the same layout
        a = ql_load2(imag + i);
        b = ql_load2(real + half + i);
        plus_child = a + b;
        minus_child = a - b;
        t_imag[i] = ql_low(plus_child, minus_child);
        t_imag[i + 1] = ql_high(plus_child, minus_child);
    }
    // Pack any remaining coordinate with the same formulas
    for (; i < half; i++) {
        t_real[i] = (FastCDPair){real[i] - imag[half + i], real[i] + imag[half + i]};
        t_imag[i] = (FastCDPair){imag[i] + real[half + i], imag[i] - real[half + i]};
    }
}

// Compute one complex call with its two recursive calls in the SIMD lanes
static void ql_simd_complex_cross(double* restrict out_real,
        double* restrict out_imag, const double* x_real, const double* x_imag,
        const double* y_real, const double* y_imag, size_t dimension,
        double* restrict workspace) {
    size_t half;
    size_t i;
    FastCDPair* pairs;
    FastCDPair* tx_real;
    FastCDPair* tx_imag;
    FastCDPair* ty_real;
    FastCDPair* ty_imag;
    FastCDPair* child_real;
    FastCDPair* child_imag;
    FastCDPair* child_workspace;
    fastcd_FastCDComplex gamma;
    fastcd_FastCDComplex delta;
    double gamma_sum;
    double delta_sum;
    FastCDPair gamma_sum_lanes;
    FastCDPair delta_sum_lanes;
    FastCDComplexLanes gamma_lanes;
    FastCDComplexLanes delta_lanes;

    // Small calls use the scalar leaves, including AltC_1 = 0
    if (dimension <= 8 || dimension <= FASTCD_QL_COMPLEX_LEAF) {
        fastcd_complex_cross(out_real, out_imag, x_real, x_imag, y_real,
                y_imag, dimension, workspace);
        return;
    }
    // Reserve four transformed input planes, the two result planes of the
    // recursive calls and their scratch, each plane half pairs long
    half = dimension >> 1;
    pairs = ql_aligned_pairs(workspace);
    tx_real = pairs;
    tx_imag = pairs + half;
    ty_real = pairs + 2 * half;
    ty_imag = pairs + 3 * half;
    child_real = pairs + 4 * half;
    child_imag = pairs + 5 * half;
    child_workspace = pairs + 6 * half;
    // Write X = (u, gamma + s) and Y = (v, delta + t), and broadcast the
    // complex scalars gamma and delta for two-coordinate recombination
    gamma.real = x_real[half];
    gamma.imag = x_imag[half];
    delta.real = y_real[half];
    delta.imag = y_imag[half];
    gamma_lanes = ql_broadcast(gamma);
    delta_lanes = ql_broadcast(delta);
    gamma_sum = gamma.real + gamma.imag;
    delta_sum = delta.real + delta.imag;
    gamma_sum_lanes = (FastCDPair){gamma_sum, gamma_sum};
    delta_sum_lanes = (FastCDPair){delta_sum, delta_sum};

    // Pack u +/- iota s and v +/- iota t, with plus in lane 0 and minus in lane 1
    ql_simd_transform(tx_real, tx_imag, x_real, x_imag, half);
    ql_simd_transform(ty_real, ty_imag, y_real, y_imag, half);
    // Evaluate Z_+ = AltC_{n-1}(u + iota s, v + iota t) and Z_- together,
    // with the paired template continuing the recursion
    fastcd_pair_complex_cross(child_real, child_imag, tx_real, tx_imag,
            ty_real, ty_imag, half, child_workspace);

    double* scalar_p = (double*)(void*)tx_real;
    double* scalar_q = (double*)(void*)tx_imag;
    double* scalar_w = (double*)(void*)ty_real;
    // Assemble lambda, F and G, with the lanes holding coordinates i and i+1
    for (i = 1; i + 1 < half; i += 2) {
        FastCDComplexLanes u = {ql_load2(x_real + i), ql_load2(x_imag + i)};
        FastCDComplexLanes v = {ql_load2(y_real + i), ql_load2(y_imag + i)};
        FastCDComplexLanes s = {ql_load2(x_real + half + i),
            ql_load2(x_imag + half + i)};
        FastCDComplexLanes t = {ql_load2(y_real + half + i),
            ql_load2(y_imag + half + i)};
        // Transpose [Z_+, Z_-] pairs into adjacent coordinates of each Z
        FastCDComplexLanes plus = {ql_low(child_real[i], child_real[i + 1]),
            ql_low(child_imag[i], child_imag[i + 1])};
        FastCDComplexLanes minus = {ql_high(child_real[i], child_real[i + 1]),
            ql_high(child_imag[i], child_imag[i + 1])};
        // Evaluate the whole nonrecursive map on the real, imaginary and summed inputs
        fastcd_pair_FastCDNonrecursive p = fastcd_pair_nonrecursive_bilinear(
                u.real, v.real, s.real, t.real, gamma_lanes.real, delta_lanes.real);
        fastcd_pair_FastCDNonrecursive q = fastcd_pair_nonrecursive_bilinear(
                u.imag, v.imag, s.imag, t.imag, gamma_lanes.imag, delta_lanes.imag);
        fastcd_pair_FastCDNonrecursive w = fastcd_pair_nonrecursive_bilinear(
                u.real + u.imag, v.real + v.imag, s.real + s.imag,
                t.real + t.imag, gamma_sum_lanes, delta_sum_lanes);
        FastCDComplexLanes first = ql_lanes_recover(p.first, q.first, w.first);
        FastCDComplexLanes second = ql_lanes_recover(p.second, q.second, w.second);
        FastCDComplexLanes difference = ql_lanes_half(ql_lanes_sub(plus, minus));
        FastCDComplexLanes mixed = {-difference.imag, difference.real};

        // Keep each reduction in scalar coordinate order, i then i+1
        ql_store2(scalar_p + i, p.scalar);
        ql_store2(scalar_q + i, q.scalar);
        ql_store2(scalar_w + i, w.scalar);
        first = ql_lanes_add(ql_lanes_half(ql_lanes_add(plus, minus)), first);
        second = ql_lanes_add(second, mixed);
        // Store the non-scalar coordinates of (F,G) in the two output halves
        ql_store2(out_real + i, first.real);
        ql_store2(out_imag + i, first.imag);
        ql_store2(out_real + half + i, second.real);
        ql_store2(out_imag + half + i, second.imag);
    }
    // Finish any remaining coordinate with the same lambda, F and G formulas,
    // reading Z_+ and Z_- from the two lanes
    for (; i < half; i++) {
        fastcd_FastCDComplex u = {x_real[i], x_imag[i]};
        fastcd_FastCDComplex v = {y_real[i], y_imag[i]};
        fastcd_FastCDComplex s = {x_real[half + i], x_imag[half + i]};
        fastcd_FastCDComplex t = {y_real[half + i], y_imag[half + i]};
        fastcd_FastCDComplex plus = {child_real[i][0], child_imag[i][0]};
        fastcd_FastCDComplex minus = {child_real[i][1], child_imag[i][1]};
        fastcd_FastCDNonrecursive p = fastcd_nonrecursive_bilinear(
                u.real, v.real, s.real, t.real, gamma.real, delta.real);
        fastcd_FastCDNonrecursive q = fastcd_nonrecursive_bilinear(
                u.imag, v.imag, s.imag, t.imag, gamma.imag, delta.imag);
        fastcd_FastCDNonrecursive w = fastcd_nonrecursive_bilinear(
                u.real + u.imag, v.real + v.imag, s.real + s.imag,
                t.real + t.imag, gamma_sum, delta_sum);
        fastcd_FastCDComplex first = fastcd_complex_recover(p.first, q.first, w.first);
        fastcd_FastCDComplex second = fastcd_complex_recover(p.second, q.second, w.second);
        fastcd_FastCDComplex difference = fastcd_complex_half(
                fastcd_complex_sub(plus, minus));
        fastcd_FastCDComplex mixed = {-difference.imag, difference.real};
        scalar_p[i] = p.scalar;
        scalar_q[i] = q.scalar;
        scalar_w[i] = w.scalar;
        first = fastcd_complex_add(fastcd_complex_half(
                    fastcd_complex_add(plus, minus)), first);
        second = fastcd_complex_add(second, mixed);
        out_real[i] = first.real;
        out_imag[i] = first.imag;
        out_real[half + i] = second.real;
        out_imag[half + i] = second.imag;
    }
    fastcd_Sum3 scalars = fastcd_sum3_arrays(
            scalar_p + 1, scalar_q + 1, scalar_w + 1, half - 1);
    // Return (F,G), where F has scalar coordinate 0 and G has scalar coordinate lambda
    out_real[0] = 0.0;
    out_imag[0] = 0.0;
    fastcd_FastCDComplex scalar = fastcd_complex_recover(
            scalars.first, scalars.second, scalars.third);
    out_real[half] = scalar.real;
    out_imag[half] = scalar.imag;
}

// Compute the alternating product of one real product above the leaves
static void ql_simd_cross_recursive(double* restrict out, const double* x,
        const double* y, size_t dimension, double* restrict workspace) {
    // AltR_n above the leaves, with X = (u, gamma + s) and Y = (v, delta + t)
    size_t half = dimension >> 1;
    size_t i;
    double gamma = x[half];
    double delta = y[half];
    fastcd_BalancedSum scalar;
    FastCDPair gamma_lanes = {gamma, gamma};
    FastCDPair delta_lanes = {delta, delta};

    /* Z = AltC_{n-1}(u + iota s, v + iota t), whose ignored coordinate 0 keeps
     * x[0], y[0], gamma and delta out of the complex inputs. Re_C Z goes to the
     * lower output half and Im_C Z to the upper half. */
    ql_simd_complex_cross(out, out + half, x, x + half, y, y + half, half,
            workspace);
    fastcd_sum_init(&scalar);
    // Compute lambda, F = Re_C Z + gamma t - delta s and the non-scalar part of
    // G = delta u - gamma v - Im_C Z, two coordinates at a time
    for (size_t base = 1; base < half; base += FASTCD_SUM_BLOCK) {
        size_t count = half - base < FASTCD_SUM_BLOCK ? half - base : FASTCD_SUM_BLOCK;
        size_t end = base + count;
        double total = 0.0;
        for (i = base; i + 1 < end; i += 2) {
            FastCDPair x_low = ql_load2(x + i);
            FastCDPair y_low = ql_load2(y + i);
            FastCDPair x_high = ql_load2(x + half + i);
            FastCDPair y_high = ql_load2(y + half + i);
            FastCDPair out_low = ql_load2(out + i);
            FastCDPair out_high = ql_load2(out + half + i);
            FastCDPair pair = x_high * y_low - y_high * x_low;
            FastCDPair first = out_low + gamma_lanes * y_high;
            FastCDPair second = delta_lanes * x_low - gamma_lanes * y_low;

            // Sum lambda = <s,v> - <t,u> in scalar coordinate order
            total = i == base ? pair[0] : total + pair[0];
            total = total + pair[1];
            // Complete and store the current coordinates of F and G
            first = first - delta_lanes * x_high;
            second = second - out_high;
            ql_store2(out + i, first);
            ql_store2(out + half + i, second);
        }
        // Finish any remaining coordinate with the same one-call root formulas
        for (; i < end; i++) {
            double pair = x[half + i] * y[i] - y[half + i] * x[i];
            double first = out[i] + gamma * y[half + i];
            double second = delta * x[i] - gamma * y[i];

            total = i == base ? pair : total + pair;
            first = first - delta * x[half + i];
            second = second - out[half + i];
            out[i] = first;
            out[half + i] = second;
        }
        if (count == FASTCD_SUM_BLOCK) fastcd_sum_block(&scalar, total);
        else {
            scalar.block = total;
            scalar.used = (unsigned)count;
        }
    }
    // Return (F,G), with lambda in the scalar coordinate of G
    out[0] = 0.0;
    out[half] = fastcd_sum_finish(&scalar);
}

// Add the scalar and imaginary terms to a completed alternating product
static void ql_simd_reconstruct(double* restrict out, const double* x,
        const double* y, size_t dimension) {
    // Mul_n: alpha = RePart(x), beta = RePart(y), coordinates 1..N-1 hold
    // X = ImPart(x) and Y = ImPart(y), and out already holds Z = AltR_n(X,Y)
    double alpha = x[0];
    double beta = y[0];

    FastCDPair alpha_lanes = {alpha, alpha};
    FastCDPair beta_lanes = {beta, beta};
    size_t i;

    // Match the scalar blocked tree, feeding adjacent coordinates in order
    out[0] = alpha * beta - fastcd_imaginary_dot(x, y, dimension);
    // Complete Z + alpha Y + beta X, with adjacent coordinates in the lanes
    for (i = 1; i + 1 < dimension; i += 2) {
        FastCDPair term = alpha_lanes * ql_load2(y + i)
            + beta_lanes * ql_load2(x + i);

        ql_store2(out + i, ql_load2(out + i) + term);
    }
    // Finish any remaining coordinate of the same reconstruction
    for (; i < dimension; i++) {
        out[i] = out[i] + (alpha * y[i] + beta * x[i]);
    }
}
