#ifndef FASTCD_BENCH_STACK_OFFSET_H
#define FASTCD_BENCH_STACK_OFFSET_H

#include <stddef.h>
#if defined(_WIN32)
#include <malloc.h>
#define FASTCD_BENCH_ALLOCA _alloca
#else
#include <alloca.h>
#define FASTCD_BENCH_ALLOCA alloca
#endif
#if defined(_MSC_VER)
#define FASTCD_BENCH_NOINLINE __declspec(noinline)
#else
#define FASTCD_BENCH_NOINLINE __attribute__((noinline))
#endif

/* Stack placement for timed blocks. A process keeps one randomized stack
 * position for its whole life, and some kernels run markedly slower at a few
 * 16-byte positions within a 4 KB page. A runner that times every block in
 * one process therefore runs each block at its own stack offset. The offsets
 * are spread evenly over the page, so that the observations of a cell sample
 * many placements instead of sharing one. */

// Return a block's stack offset in bytes: evenly spaced 16-byte steps over one page
static inline size_t fastcd_block_stack_offset(size_t block, size_t blocks) {
    return 16 * (block % blocks * 256 / blocks);
}

// Run a timed callback with the stack moved down by the given offset
static FASTCD_BENCH_NOINLINE int fastcd_at_stack_offset(
        size_t offset, int (*timed)(void*), void* context) {
    volatile unsigned char* pad =
            (volatile unsigned char*)FASTCD_BENCH_ALLOCA(offset + 1);
    pad[0] = 0;
    int result = timed(context);
    pad[offset] = 0;
    return result;
}

#endif
