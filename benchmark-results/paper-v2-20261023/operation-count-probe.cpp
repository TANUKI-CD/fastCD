
#include <cstdio>
#include <vector>

struct Number {
    double value;
    bool fixed;
    Number(double v = 0.0, bool f = true) : value(v), fixed(f) {}
};
static unsigned long long products, additions, scalings;
static Number operator-(Number a) { return Number(-a.value, a.fixed); }
static Number operator+(Number a, Number b) {
    if (a.fixed && b.fixed) return Number(a.value + b.value);
    if (a.fixed && a.value == 0.0) return b;
    if (b.fixed && b.value == 0.0) return a;
    ++additions;
    return Number(a.value + b.value, false);
}
static Number operator-(Number a, Number b) {
    if (a.fixed && b.fixed) return Number(a.value - b.value);
    if (b.fixed && b.value == 0.0) return a;
    if (a.fixed && a.value == 0.0) return -b;
    ++additions;
    return Number(a.value - b.value, false);
}
static Number operator*(Number a, Number b) {
    if (a.fixed && b.fixed) return Number(a.value * b.value);
    if (a.fixed || b.fixed) {
        double constant = a.fixed ? a.value : b.value;
        if (constant == 0.0) return Number(0.0);
        if (constant != 1.0 && constant != -1.0) ++scalings;
    } else {
        ++products;
    }
    return Number(a.value * b.value, false);
}

#define restrict __restrict
#define FASTCD_REAL Number
#define FASTCD_ZERO Number(0.0)
#define FASTCD_NAME(name) measured_##name
#define FASTCD_QL_FULL_LEAF 16
#define FASTCD_QL_COMPLEX_LEAF 8
#include "generated/quasilinear_kernels.h"
#include "quasilinear_impl.h"

static void report(const char* kind, size_t n) {
    std::printf("{\"kind\":\"%s\",\"dimension\":%zu,"
        "\"M\":%llu,\"A\":%llu,\"S\":%llu,\"T\":%llu}\n",
        kind, n, products, additions, scalings, products + additions + scalings);
}
int main() {
    for (size_t n = 1; n <= 1024; n *= 2) {
        std::vector<Number> x(n), y(n), xi(n), yi(n), out(n), oi(n), scratch(8*n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = Number(0.25, false); y[i] = Number(0.5, false);
            xi[i] = Number(0.125, false); yi[i] = Number(-0.25, false);
        }
        products = additions = scalings = 0;
        measured_ql_mul_recursive(out.data(), x.data(), y.data(), n, scratch.data());
        report("full", n);
        products = additions = scalings = 0;
        measured_complex_cross(out.data(), oi.data(), x.data(), xi.data(),
            y.data(), yi.data(), n, scratch.data());
        report("complex", n);
    }
}
