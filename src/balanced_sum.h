/* Internal binary64 reduction schedule, instantiated with FASTCD_REAL,
 * FASTCD_ZERO and FASTCD_NAME. Independent SIMD lanes never interact. There is
 * no include guard, because the scalar and paired types share this code. */
#include <limits.h>

#ifndef FASTCD_SUM_BLOCK
#define FASTCD_SUM_BLOCK 8
#endif
#if FASTCD_SUM_BLOCK != 1 && FASTCD_SUM_BLOCK != 4 && FASTCD_SUM_BLOCK != 8 && FASTCD_SUM_BLOCK != 16
#error "FASTCD_SUM_BLOCK must be 1, 4, 8, or 16"
#endif

typedef struct FASTCD_NAME(BalancedSum) {
    FASTCD_REAL levels[sizeof(size_t) * CHAR_BIT];
    FASTCD_REAL block;
    size_t blocks;
    unsigned used;
} FASTCD_NAME(BalancedSum);

// Initialize the bookkeeping only, since unused arithmetic slots are never read
static inline void FASTCD_NAME(sum_init)(FASTCD_NAME(BalancedSum)* sum) {
    sum->blocks = 0;
    sum->used = 0;
}

// Insert one completed block, merging adjacent equal-sized earlier blocks
static inline void FASTCD_NAME(sum_block)(FASTCD_NAME(BalancedSum)* sum,
        FASTCD_REAL value) {
    size_t occupied = sum->blocks;
    size_t level = 0;

    while ((occupied & 1) != 0) {
        value = sum->levels[level] + value;
        occupied >>= 1;
        level++;
    }
    sum->levels[level] = value;
    sum->blocks++;
}

// Accumulate a fixed small block, then carry its total through a balanced tree
static inline void FASTCD_NAME(sum_push)(FASTCD_NAME(BalancedSum)* sum,
        FASTCD_REAL value) {
    if (sum->used == 0) sum->block = value;
    else sum->block = sum->block + value;
    sum->used++;
    if (sum->used == FASTCD_SUM_BLOCK) {
        FASTCD_NAME(sum_block)(sum, sum->block);
        sum->used = 0;
    }
}

/* Rightmost blocks fold first, with m-1 additions for m > 0 terms, a depth of
 * at most B-1 + ceil(log2(ceil(m/B))) + 1, and no arithmetic zero inserted.
 * The level array covers every addressable array, since a carry past it would
 * need SIZE_MAX+1 terms even when B = 1, and no shift by a type width occurs.
 * Recursion-local bounded arrays add O(log N) machine words per reduction and
 * at most O(log^2 N) live words, within O(N) total storage. */
// Finish a reduction, folding the partial block and then the occupied levels
static inline FASTCD_REAL FASTCD_NAME(sum_finish)(
        const FASTCD_NAME(BalancedSum)* sum) {
    FASTCD_REAL result = FASTCD_ZERO;
    size_t occupied = sum->blocks;
    size_t level = 0;
    int have_result = sum->used != 0;

    if (have_result) result = sum->block;
    while (occupied != 0) {
        if ((occupied & 1) != 0) {
            result = have_result ? sum->levels[level] + result : sum->levels[level];
            have_result = 1;
        }
        occupied >>= 1;
        level++;
    }
    return result;
}

/* The whole-block helpers below share the binary counter across synchronized
 * streams. Hot loops form each block directly and call them once per block,
 * and the streaming helper above serves irregular internal callers. */
// Keep a function out of line on every supported compiler
#ifndef FASTCD_NOINLINE
#if defined(_MSC_VER)
#define FASTCD_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define FASTCD_NOINLINE __attribute__((noinline))
#else
#define FASTCD_NOINLINE
#endif
#endif

typedef struct FASTCD_NAME(Sum2) {
    FASTCD_REAL first, second;
} FASTCD_NAME(Sum2);

typedef struct FASTCD_NAME(Sum3) {
    FASTCD_REAL first, second, third;
} FASTCD_NAME(Sum3);

typedef struct FASTCD_NAME(BalancedTree2) {
    FASTCD_NAME(Sum2) levels[sizeof(size_t) * CHAR_BIT];
    size_t blocks;
} FASTCD_NAME(BalancedTree2);

typedef struct FASTCD_NAME(BalancedTree3) {
    FASTCD_NAME(Sum3) levels[sizeof(size_t) * CHAR_BIT];
    size_t blocks;
} FASTCD_NAME(BalancedTree3);

// Merge one completed block of two independent streams with shared bookkeeping
static inline void FASTCD_NAME(sum2_block)(FASTCD_NAME(BalancedTree2)* tree,
        FASTCD_NAME(Sum2) value) {
    size_t occupied = tree->blocks;
    size_t level = 0;
    while ((occupied & 1) != 0) {
        value.first = tree->levels[level].first + value.first;
        value.second = tree->levels[level].second + value.second;
        occupied >>= 1;
        level++;
    }
    tree->levels[level] = value;
    tree->blocks++;
}

// Merge one completed block of three independent streams with one binary carry
static inline void FASTCD_NAME(sum3_block)(FASTCD_NAME(BalancedTree3)* tree,
        FASTCD_NAME(Sum3) value) {
    size_t occupied = tree->blocks;
    size_t level = 0;
    while ((occupied & 1) != 0) {
        value.first = tree->levels[level].first + value.first;
        value.second = tree->levels[level].second + value.second;
        value.third = tree->levels[level].third + value.third;
        occupied >>= 1;
        level++;
    }
    tree->levels[level] = value;
    tree->blocks++;
}

// Complete two trees, folding a partial rightmost block before the occupied levels
static inline FASTCD_NAME(Sum2) FASTCD_NAME(sum2_finish)(
        const FASTCD_NAME(BalancedTree2)* tree, FASTCD_NAME(Sum2) tail,
        int have_tail) {
    size_t occupied = tree->blocks;
    size_t level = 0;
    while (occupied != 0) {
        if ((occupied & 1) != 0) {
            if (have_tail) {
                tail.first = tree->levels[level].first + tail.first;
                tail.second = tree->levels[level].second + tail.second;
            } else tail = tree->levels[level];
            have_tail = 1;
        }
        occupied >>= 1;
        level++;
    }
    return tail;
}

// Complete three trees with the same rightmost-first schedule as sum_finish
static inline FASTCD_NAME(Sum3) FASTCD_NAME(sum3_finish)(
        const FASTCD_NAME(BalancedTree3)* tree, FASTCD_NAME(Sum3) tail,
        int have_tail) {
    size_t occupied = tree->blocks;
    size_t level = 0;
    while (occupied != 0) {
        if ((occupied & 1) != 0) {
            if (have_tail) {
                tail.first = tree->levels[level].first + tail.first;
                tail.second = tree->levels[level].second + tail.second;
                tail.third = tree->levels[level].third + tail.third;
            } else tail = tree->levels[level];
            have_tail = 1;
        }
        occupied >>= 1;
        level++;
    }
    return tail;
}

// Reduce long arrays in fixed blocks, out of line to keep callers' frames small
static FASTCD_NOINLINE FASTCD_NAME(Sum3) FASTCD_NAME(sum3_arrays_long)(
        const FASTCD_REAL* first, const FASTCD_REAL* second,
        const FASTCD_REAL* third, size_t count) {
    FASTCD_NAME(BalancedTree3) tree;
    FASTCD_NAME(Sum3) total = {FASTCD_ZERO, FASTCD_ZERO, FASTCD_ZERO};
    tree.blocks = 0;
    // Fold each block left to right and carry every full block into the tree
    for (size_t base = 0; base < count; base += FASTCD_SUM_BLOCK) {
        size_t length = count - base < FASTCD_SUM_BLOCK ? count - base : FASTCD_SUM_BLOCK;
        total.first = first[base];
        total.second = second[base];
        total.third = third[base];
        for (size_t j = 1; j < length; j++) {
            total.first = total.first + first[base + j];
            total.second = total.second + second[base + j];
            total.third = total.third + third[base + j];
        }
        if (length == FASTCD_SUM_BLOCK) FASTCD_NAME(sum3_block)(&tree, total);
    }
    return FASTCD_NAME(sum3_finish)(&tree, total, count % FASTCD_SUM_BLOCK != 0);
}

// Reduce three arrays, folding a short reduction as one block without a level array
static inline FASTCD_NAME(Sum3) FASTCD_NAME(sum3_arrays)(
        const FASTCD_REAL* first, const FASTCD_REAL* second,
        const FASTCD_REAL* third, size_t count) {
    FASTCD_NAME(Sum3) total = {FASTCD_ZERO, FASTCD_ZERO, FASTCD_ZERO};
    if (count > FASTCD_SUM_BLOCK) {
        return FASTCD_NAME(sum3_arrays_long)(first, second, third, count);
    }
    if (count == 0) return total;
    total.first = first[0];
    total.second = second[0];
    total.third = third[0];
    for (size_t j = 1; j < count; j++) {
        total.first = total.first + first[j];
        total.second = total.second + second[j];
        total.third = total.third + third[j];
    }
    return total;
}
