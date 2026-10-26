/*
 * Single-product quasilinear kernels for builds with the two-lane SIMD type.
 * Included once from quasilinear.c, after the scalar and paired
 * instantiations of quasilinear_impl.h, and only when FASTCD_HAVE_PAIR.
 *
 * The top complex alternating node evaluates its half-dimensional children as
 * the two lanes of the paired kernel, and performs its own O(K) work two
 * coordinates per instruction.  Every output coordinate follows exactly the
 * arithmetic schedule of the scalar recursion (the sequential sums keep
 * their order), so single products, batches, and non-SIMD builds agree
 * bitwise.  The node needs 6K doubles of pair storage plus the paired
 * child's scratch; fastcd_quasilinear_workspace reports that requirement.
 *
 * Paper notation: dimension = 2^n for AltC_n, AltR_n and Mul_n.  The
 * real/imag planes hold coefficients over C; iota is the auxiliary complex
 * unit.  SIMD lanes are separate calculations, not real/imag components.
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

// Recover the complex vector outputs of three real local-map evaluations
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
 * Write the transformed input planes of both children into pair arrays,
 * lane 0 holding the first (plus) child and lane 1 the second (minus) child,
 * with the same operations as the two scalar transform loops.
 */
// Pack both children's transformed planes of one operand
static void ql_simd_transform(FastCDPair* restrict t_real,
        FastCDPair* restrict t_imag, const double* real, const double* imag,
        size_t half) {
    size_t i;

    // AltC_n: both u +/- iota s (or v +/- iota t) belong to V_{n-1}^C.
    // Set their CD scalar coordinates to zero in both child lanes.
    t_real[0] = (FastCDPair){0.0, 0.0};
    t_imag[0] = (FastCDPair){0.0, 0.0};
    // Form Re_C(u +/- iota s) = Re_C u -/+ Im_C s, two coordinates at a time.
    for (i = 1; i + 1 < half; i += 2) {
        FastCDPair a = ql_load2(real + i);
        FastCDPair b = ql_load2(imag + half + i);
        FastCDPair plus_child = a - b;
        FastCDPair minus_child = a + b;

        // Transpose coordinate lanes into child lanes: [plus_i, minus_i].
        t_real[i] = ql_low(plus_child, minus_child);
        t_real[i + 1] = ql_high(plus_child, minus_child);
        // Im_C(u +/- iota s) = Im_C u +/- Re_C s, in the same child layout.
        a = ql_load2(imag + i);
        b = ql_load2(real + half + i);
        plus_child = a + b;
        minus_child = a - b;
        t_imag[i] = ql_low(plus_child, minus_child);
        t_imag[i + 1] = ql_high(plus_child, minus_child);
    }
    // Pack any remaining coordinate with the same formulas; no recursive call.
    for (; i < half; i++) {
        t_real[i] = (FastCDPair){real[i] - imag[half + i], real[i] + imag[half + i]};
        t_imag[i] = (FastCDPair){imag[i] + real[half + i], imag[i] - real[half + i]};
    }
}

// Compute one complex alternating node with its children in SIMD lanes
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
    double scalar_p = 0.0;
    double scalar_q = 0.0;
    double scalar_w = 0.0;
    double gamma_sum;
    double delta_sum;
    FastCDPair gamma_sum_lanes;
    FastCDPair delta_sum_lanes;
    FastCDComplexLanes gamma_lanes;
    FastCDComplexLanes delta_lanes;

    // Small AltC_n nodes use the scalar leaves, including AltC_1 = 0.
    if (dimension <= 8 || dimension <= FASTCD_QL_COMPLEX_LEAF) {
        fastcd_complex_cross(out_real, out_imag, x_real, x_imag, y_real,
                y_imag, dimension, workspace);
        return;
    }
    // Reserve four transformed input planes, two child-result planes and
    // deeper scratch.  Each plane has half pairs, one lane per child.
    half = dimension >> 1;
    pairs = ql_aligned_pairs(workspace);
    tx_real = pairs;
    tx_imag = pairs + half;
    ty_real = pairs + 2 * half;
    ty_imag = pairs + 3 * half;
    child_real = pairs + 4 * half;
    child_imag = pairs + 5 * half;
    child_workspace = pairs + 6 * half;
    // Write X = (u, gamma + s), Y = (v, delta + t).  gamma and delta
    // are complex scalars; broadcast them for two-coordinate recombination.
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

    // Pack u +/- iota s and v +/- iota t, with plus in lane 0, minus in lane 1.
    ql_simd_transform(tx_real, tx_imag, x_real, x_imag, half);
    ql_simd_transform(ty_real, ty_imag, y_real, y_imag, half);
    // Evaluate Z_+ = AltC_{n-1}(u + iota s, v + iota t) and Z_- with
    // minus signs together.  Recursion continues inside the paired template.
    fastcd_pair_complex_cross(child_real, child_imag, tx_real, tx_imag,
            ty_real, ty_imag, half, child_workspace);

    // Assemble lambda, F and G; arithmetic lanes now represent coordinates i,i+1.
    for (i = 1; i + 1 < half; i += 2) {
        FastCDComplexLanes u = {ql_load2(x_real + i), ql_load2(x_imag + i)};
        FastCDComplexLanes v = {ql_load2(y_real + i), ql_load2(y_imag + i)};
        FastCDComplexLanes s = {ql_load2(x_real + half + i),
            ql_load2(x_imag + half + i)};
        FastCDComplexLanes t = {ql_load2(y_real + half + i),
            ql_load2(y_imag + half + i)};
        // Transpose [Z_+, Z_-] child pairs into adjacent coordinates of each Z.
        FastCDComplexLanes plus = {ql_low(child_real[i], child_real[i + 1]),
            ql_low(child_imag[i], child_imag[i + 1])};
        FastCDComplexLanes minus = {ql_high(child_real[i], child_real[i + 1]),
            ql_high(child_imag[i], child_imag[i + 1])};
        // Evaluate the whole local map on real, imaginary and summed inputs.
        fastcd_pair_FastCDLocal p = fastcd_pair_local_bilinear(
                u.real, v.real, s.real, t.real, gamma_lanes.real, delta_lanes.real);
        fastcd_pair_FastCDLocal q = fastcd_pair_local_bilinear(
                u.imag, v.imag, s.imag, t.imag, gamma_lanes.imag, delta_lanes.imag);
        fastcd_pair_FastCDLocal w = fastcd_pair_local_bilinear(
                u.real + u.imag, v.real + v.imag, s.real + s.imag,
                t.real + t.imag, gamma_sum_lanes, delta_sum_lanes);
        FastCDComplexLanes first = ql_lanes_recover(p.first, q.first, w.first);
        FastCDComplexLanes second = ql_lanes_recover(p.second, q.second, w.second);
        FastCDComplexLanes difference = ql_lanes_half(ql_lanes_sub(plus, minus));
        FastCDComplexLanes mixed = {-difference.imag, difference.real};

        // Keep each reduction in scalar coordinate order, i then i+1.
        scalar_p = i == 1 ? p.scalar[0] : scalar_p + p.scalar[0];
        scalar_p = scalar_p + p.scalar[1];
        scalar_q = i == 1 ? q.scalar[0] : scalar_q + q.scalar[0];
        scalar_q = scalar_q + q.scalar[1];
        scalar_w = i == 1 ? w.scalar[0] : scalar_w + w.scalar[0];
        scalar_w = scalar_w + w.scalar[1];
        first = ql_lanes_add(ql_lanes_half(ql_lanes_add(plus, minus)), first);
        second = ql_lanes_add(second, mixed);
        // Store the non-scalar coordinates of (F,G) in the two output halves.
        ql_store2(out_real + i, first.real);
        ql_store2(out_imag + i, first.imag);
        ql_store2(out_real + half + i, second.real);
        ql_store2(out_imag + half + i, second.imag);
    }
    // Finish any remaining coordinate with the same lambda, F, G formulas.
    // Child lanes still hold Z_+, Z_-; this is scalar recombination, not recursion.
    for (; i < half; i++) {
        fastcd_FastCDComplex u = {x_real[i], x_imag[i]};
        fastcd_FastCDComplex v = {y_real[i], y_imag[i]};
        fastcd_FastCDComplex s = {x_real[half + i], x_imag[half + i]};
        fastcd_FastCDComplex t = {y_real[half + i], y_imag[half + i]};
        fastcd_FastCDComplex plus = {child_real[i][0], child_imag[i][0]};
        fastcd_FastCDComplex minus = {child_real[i][1], child_imag[i][1]};
        fastcd_FastCDLocal p = fastcd_local_bilinear(
                u.real, v.real, s.real, t.real, gamma.real, delta.real);
        fastcd_FastCDLocal q = fastcd_local_bilinear(
                u.imag, v.imag, s.imag, t.imag, gamma.imag, delta.imag);
        fastcd_FastCDLocal w = fastcd_local_bilinear(
                u.real + u.imag, v.real + v.imag, s.real + s.imag,
                t.real + t.imag, gamma_sum, delta_sum);
        fastcd_FastCDComplex first = fastcd_complex_recover(p.first, q.first, w.first);
        fastcd_FastCDComplex second = fastcd_complex_recover(p.second, q.second, w.second);
        fastcd_FastCDComplex difference = fastcd_complex_half(
                fastcd_complex_sub(plus, minus));
        fastcd_FastCDComplex mixed = {-difference.imag, difference.real};
        scalar_p = i == 1 ? p.scalar : scalar_p + p.scalar;
        scalar_q = i == 1 ? q.scalar : scalar_q + q.scalar;
        scalar_w = i == 1 ? w.scalar : scalar_w + w.scalar;
        first = fastcd_complex_add(fastcd_complex_half(
                    fastcd_complex_add(plus, minus)), first);
        second = fastcd_complex_add(second, mixed);
        out_real[i] = first.real;
        out_imag[i] = first.imag;
        out_real[half + i] = second.real;
        out_imag[half + i] = second.imag;
    }
    // Return (F,G): F has scalar coordinate 0 and G has scalar coordinate lambda.
    out_real[0] = 0.0;
    out_imag[0] = 0.0;
    fastcd_FastCDComplex scalar = fastcd_complex_recover(scalar_p, scalar_q, scalar_w);
    out_real[half] = scalar.real;
    out_imag[half] = scalar.imag;
}

// Compute the imaginary cross product of one real product above the leaves
static void ql_simd_cross_recursive(double* restrict out, const double* x,
        const double* y, size_t dimension, double* restrict workspace) {
    // AltR_n above the leaves: write X = (u, gamma + s), Y = (v, delta + t).
    size_t half = dimension >> 1;
    size_t i;
    double gamma = x[half];
    double delta = y[half];
    double scalar = 0.0;
    FastCDPair gamma_lanes = {gamma, gamma};
    FastCDPair delta_lanes = {delta, delta};

    // Z = AltC_{n-1}(u + iota s, v + iota t); ignored coordinate 0
    // omits x[0], y[0], gamma and delta from the complex child inputs.
    // Store Re_C Z in the lower output half and Im_C Z in the upper half.
    ql_simd_complex_cross(out, out + half, x, x + half, y, y + half, half,
            workspace);
    // Compute lambda, F = Re_C Z + gamma t - delta s, and the non-scalar
    // part of G = delta u - gamma v - Im_C Z, two coordinates at a time.
    for (i = 1; i + 1 < half; i += 2) {
        FastCDPair x_low = ql_load2(x + i);
        FastCDPair y_low = ql_load2(y + i);
        FastCDPair x_high = ql_load2(x + half + i);
        FastCDPair y_high = ql_load2(y + half + i);
        FastCDPair out_low = ql_load2(out + i);
        FastCDPair out_high = ql_load2(out + half + i);
        FastCDPair pair = x_high * y_low - y_high * x_low;
        FastCDPair first = out_low + gamma_lanes * y_high;
        FastCDPair second = delta_lanes * x_low - gamma_lanes * y_low;

        // Sum lambda = <s,v> - <t,u> in scalar coordinate order.
        scalar = i == 1 ? pair[0] : scalar + pair[0];
        scalar = scalar + pair[1];
        // Complete and store the current coordinates of F and G.
        first = first - delta_lanes * x_high;
        second = second - out_high;
        ql_store2(out + i, first);
        ql_store2(out + half + i, second);
    }
    // Finish any remaining coordinate with the same real-root formulas.
    for (; i < half; i++) {
        double pair = x[half + i] * y[i] - y[half + i] * x[i];
        double first = out[i] + gamma * y[half + i];
        double second = delta * x[i] - gamma * y[i];

        scalar = i == 1 ? pair : scalar + pair;
        first = first - delta * x[half + i];
        second = second - out[half + i];
        out[i] = first;
        out[half + i] = second;
    }
    // Return (F,G), with lambda in the scalar coordinate of G.
    out[0] = 0.0;
    out[half] = scalar;
}

// Add the scalar--imaginary terms to a completed imaginary cross product
static void ql_simd_reconstruct(double* restrict out, const double* x,
        const double* y, size_t dimension) {
    // Mul_n: alpha = RePart(x), beta = RePart(y); coordinates 1..N-1
    // hold X = ImPart(x), Y = ImPart(y).  out already holds Z = AltR_n(X,Y).
    double alpha = x[0];
    double beta = y[0];
    double dot = x[1] * y[1];
    FastCDPair alpha_lanes = {alpha, alpha};
    FastCDPair beta_lanes = {beta, beta};
    size_t i;

    // Compute <X,Y> in scalar order, then write alpha beta - <X,Y>.
    for (i = 2; i < dimension; i++) dot = dot + x[i] * y[i];
    out[0] = alpha * beta - dot;
    // Complete Z + alpha Y + beta X, with adjacent coordinates in the lanes.
    for (i = 1; i + 1 < dimension; i += 2) {
        FastCDPair term = alpha_lanes * ql_load2(y + i)
            + beta_lanes * ql_load2(x + i);

        ql_store2(out + i, ql_load2(out + i) + term);
    }
    // Finish any remaining coordinate of the same Mul_n reconstruction.
    for (; i < dimension; i++) {
        out[i] = out[i] + (alpha * y[i] + beta * x[i]);
    }
}
