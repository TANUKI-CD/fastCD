#!/usr/bin/env python3
"""Generate separate, explicitly targeted SIMD Montgomery experiments."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".private/finite-optimization/representations"


def generate(lanes, bits, prefix, target, fast=False, values=False, build=True):
    p = prefix
    v = f"__m{bits}i"
    name = f"{'value' if values else 'fast' if fast else 'packed'}{lanes}"
    z = f"{p}_setzero_si{bits}()"

    # Intrinsic names for register-wide bitwise operations follow register bits.
    def And(a, b):
        return f"{p}_and_si{bits}({a},{b})"

    def Or(a, b):
        return f"{p}_or_si{bits}({a},{b})"

    source = f"""
#include <immintrin.h>
#include <string.h>
#include "fastcd/finite.h"
typedef {v} Word;
typedef struct {{ Word prime,negative_inverse,r2,normalization; }} Context;
static inline Word add(Word a,Word b,const Context* c) {{
    Word t={p}_sub_epi32({p}_add_epi32(a,b),c->prime);
    return {p}_add_epi32(t,{And(f"{p}_srai_epi32(t,31)", "c->prime")});
}}
static inline Word sub(Word a,Word b,const Context* c) {{
    Word t={p}_sub_epi32(a,b);
    return {p}_add_epi32(t,{And(f"{p}_srai_epi32(t,31)", "c->prime")});
}}
static inline Word half(Word a,const Context* c) {{
    Word odd={p}_sub_epi32({z},{And("a", f"{p}_set1_epi32(1)")});
    return {p}_srli_epi32({p}_add_epi32(a,{And("odd", "c->prime")}),1);
}}
static inline Word multiply(Word a,Word b,const Context* c) {{
    Word even={p}_mul_epu32(a,b);
    Word odd={p}_mul_epu32({p}_srli_epi64(a,32),{p}_srli_epi64(b,32));
    Word me={p}_mullo_epi32(even,c->negative_inverse);
    Word mo={p}_mullo_epi32(odd,c->negative_inverse);
    even={p}_srli_epi64({p}_add_epi64(even,{p}_mul_epu32(me,c->prime)),32);
    odd={p}_slli_epi64({p}_srli_epi64({p}_add_epi64(odd,{p}_mul_epu32(mo,c->prime)),32),32);
    Word t={p}_sub_epi32({Or("even", "odd")},c->prime);
    return {p}_add_epi32(t,{And(f"{p}_srai_epi32(t,31)", "c->prime")});
}}
static inline Word scale(Word a,int kind,const Context* c) {{
    return kind==0?half(a,c):multiply(a,c->normalization,c);
}}
#define FASTCD_FF_WORD Word
#define FASTCD_FF_STEP 1
#define FASTCD_FF_CONTEXT const Context
#define FASTCD_FF_NAME(n) {name}_##n
#define FASTCD_FF_ADD(o,a,b,c) (*(o)=add(*(a),*(b),c))
#define FASTCD_FF_SUB(o,a,b,c) (*(o)=sub(*(a),*(b),c))
#define FASTCD_FF_MUL(o,a,b,c) (*(o)=multiply(*(a),*(b),c))
#define FASTCD_FF_SCALE(o,a,s,c) (*(o)=scale(*(a),s,c))
#define FASTCD_FF_COPY(o,a,c) (*(o)=*(a))
#define FASTCD_FF_ZERO(o,c) (*(o)={z})
#define FASTCD_FF_INVERSE_TWO(c) 0
#define FASTCD_FF_INVERSE_N(c) 1
#define FASTCD_FF_STATS_DECL
#define FASTCD_FF_STATS_ARG
#define FASTCD_FF_RECORD_NODE(n) ((void)0)
#include "schedule_impl.h"

void {name}_run(uint32_t* out,uint32_t* x,uint32_t* y,const FastCDPlanFF* p,uint32_t* work,int loop) {{
    uint32_t prime=p->field.prime,inv=1;
    for(int i=0;i<5;i++) inv*=2-prime*inv;
    uint32_t r=(uint32_t)((UINT64_C(1)<<32)%prime);
    uint32_t r2=fastcd_ff_prime_mul(r,r,&p->field);
    Context c={{ {p}_set1_epi32((int)prime),{p}_set1_epi32((int)(0u-inv)),
        {p}_set1_epi32((int)r2),{p}_set1_epi32((int)fastcd_ff_prime_mul(p->inverse_dimension,r,&p->field)) }};
    size_t size=p->dimension*(loop?2:1);
    /* Public buffers need only uint32_t alignment. Unaligned intrinsic loads
     * also support arbitrary offsets within a queried workspace. */
    Word* ax=(Word*)x; Word* by=(Word*)y; Word* result=(Word*)out;
    /* Recursive word arrays are accessed through an unaligned vector typedef. */
    for(size_t i=0;i<size;i++) {{
        Word a={p}_loadu_si{bits}((const void*)(x+i*{lanes}));
        Word b={p}_loadu_si{bits}((const void*)(y+i*{lanes}));
        {p}_storeu_si{bits}((void*)(x+i*{lanes}),multiply(a,c.r2,&c));
        {p}_storeu_si{bits}((void*)(y+i*{lanes}),multiply(b,c.r2,&c));
    }}
    if(loop) {name}_loop(result,ax,by,p->dimension,(Word*)work,&c);
    else {name}_cd(result,ax,by,p->dimension,(Word*)work,&c);
    Word one={p}_set1_epi32(1);
    for(size_t i=0;i<size;i++) {{
        Word a={p}_loadu_si{bits}((const void*)(out+i*{lanes}));
        {p}_storeu_si{bits}((void*)(out+i*{lanes}),multiply(a,one,&c));
    }}
}}
"""
    # GNU/Clang permit a vector typedef with alignment one, avoiding accidental
    # aligned load assumptions in the scalar schedule's pointer dereferences.
    source = source.replace(
        f"typedef {v} Word;", f"typedef {v} Word __attribute__((aligned(1),may_alias));"
    )
    if fast:
        # A complete bilinear coordinate a*b-c*d can share one reduction.
        # t=a*b+(p-c)*d < 2*p^2 < 2^63; t+m*p < 2^64 and REDC < 2p.
        extra = f"""
static inline Word difference(Word a,Word b,Word x,Word y,const Context* c) {{
    Word pc={p}_sub_epi32(c->prime,x);
    Word even={p}_add_epi64({p}_mul_epu32(a,b),{p}_mul_epu32(pc,y));
    Word odd={p}_add_epi64({p}_mul_epu32({p}_srli_epi64(a,32),{p}_srli_epi64(b,32)),
        {p}_mul_epu32({p}_srli_epi64(pc,32),{p}_srli_epi64(y,32)));
    Word me={p}_mullo_epi32(even,c->negative_inverse);
    Word mo={p}_mullo_epi32(odd,c->negative_inverse);
    even={p}_srli_epi64({p}_add_epi64(even,{p}_mul_epu32(me,c->prime)),32);
    odd={p}_slli_epi64({p}_srli_epi64({p}_add_epi64(odd,{p}_mul_epu32(mo,c->prime)),32),32);
    Word t={p}_sub_epi32({Or("even", "odd")},c->prime);
    return {p}_add_epi32(t,{And(f"{p}_srai_epi32(t,31)", "c->prime")});
}}
static inline Word fixed(Word a,Word b,Word quotient,const Context* c) {{
    Word even={p}_srli_epi64({p}_mul_epu32(a,quotient),32);
    Word odd={p}_mul_epu32({p}_srli_epi64(a,32),{p}_srli_epi64(quotient,32));
    odd={And("odd", f"{p}_set1_epi64x(-4294967296LL)") if bits < 512 else And("odd", f"{p}_set1_epi64(-4294967296LL)")};
    Word q={Or("even", "odd")};
    Word t={p}_sub_epi32({p}_sub_epi32({p}_mullo_epi32(a,b),{p}_mullo_epi32(q,c->prime)),c->prime);
    return {p}_add_epi32(t,{And(f"{p}_srai_epi32(t,31)", "c->prime")});
}}
"""
        source = source.replace(
            "#define FASTCD_FF_WORD Word", extra + "\n#define FASTCD_FF_WORD Word"
        )
        schedule = (
            (ROOT / "src/finite/schedule_impl.h")
            .read_text()
            .replace("FASTCD_FF_CONTEXT* ctx", "FASTCD_FF_CONTEXT* restrict ctx")
        )
        schedule = schedule.replace(
            "static void FASTCD_FF_NAME(recover)",
            "static inline __attribute__((always_inline)) void FASTCD_FF_NAME(recover)",
        )
        first = schedule.index("static void FASTCD_FF_NAME(local)(")
        last = schedule.index("\n/*", first)
        replacement = """static inline __attribute__((always_inline)) void FASTCD_FF_NAME(local)(
    Word* first,Word* scalar,Word* second,const Word* u,const Word* v,
    const Word* s,const Word* t,const Word* gamma,const Word* delta,Word* tmp,
    FASTCD_FF_CONTEXT* restrict ctx) {
    (void)tmp;
    Word uu=*u,vv=*v,ss=*s,tt=*t,gg=*gamma,dd=*delta;
    *first=difference(gg,tt,dd,ss,ctx);
    *scalar=difference(ss,vv,tt,uu,ctx);
    *second=difference(dd,uu,gg,vv,ctx);
}
"""
        schedule = schedule[:first] + replacement + schedule[last:]
        (WORK / "fast_schedule.h").write_text(schedule)
        source = source.replace(
            '#include "schedule_impl.h"', '#include "fast_schedule.h"'
        )
        if bits == 128:
            horizontal = f"""a=add(a,{p}_shuffle_epi32(a,0xb1),c);
    a=add(a,{p}_shuffle_epi32(a,0x4e),c);
    return (uint32_t)_mm_cvtsi128_si32(a);"""
        elif bits == 256:
            horizontal = f"""a=add(a,{p}_shuffle_epi32(a,0xb1),c);
    a=add(a,{p}_shuffle_epi32(a,0x4e),c);
    a=add(a,{p}_permute2x128_si256(a,a,1),c);
    return (uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(a));"""
        else:
            horizontal = f"""a=add(a,{p}_shuffle_epi32(a,0xb1),c);
    a=add(a,{p}_shuffle_epi32(a,0x4e),c);
    a=add(a,{p}_shuffle_i32x4(a,a,0xb1),c);
    a=add(a,{p}_shuffle_i32x4(a,a,0x4e),c);
    return (uint32_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(a));"""
        source += f"""
static uint32_t horizontal(Word a,const Context* c) {{
    {horizontal}
}}
void {name}_tensor(uint32_t* restrict out,const uint32_t* x,const uint32_t* y,
    const FastCDPlanFF* p,uint32_t* restrict work,int loop,size_t rank,
    const uint32_t input[48][12],const uint32_t output[48][12]) {{
    uint32_t prime=p->field.prime,inv=1;
    for(int i=0;i<5;i++) inv*=2-prime*inv;
    uint32_t r=(uint32_t)((UINT64_C(1)<<32)%prime),ri=1,b=r,e=prime-2;
    while(e) {{if(e&1)ri=fastcd_ff_prime_mul(ri,b,&p->field);b=fastcd_ff_prime_mul(b,b,&p->field);e>>=1;}}
    Context c={{ {p}_set1_epi32((int)prime),{p}_set1_epi32((int)(0u-inv)),
        {z},{p}_set1_epi32((int)fastcd_ff_prime_mul(p->inverse_dimension,r,&p->field)) }};
    size_t size=p->dimension*(loop?2:1),d=p->field.degree;
    Word* ex=(Word*)work;Word* ey=ex+size;Word* eo=ey+size;Word* scratch=eo+size;
    memset(out,0,size*d*sizeof(uint32_t));
    for(size_t first=0;first<rank;first+={lanes}) {{
        Word a[12],ah[12],w[12],wh[12];
        for(size_t j=0;j<d;j++) {{
            uint32_t av[{lanes}],qv[{lanes}],wv[{lanes}],qh[{lanes}];
            for(size_t lane=0;lane<{lanes};lane++) {{
                size_t row=first+lane;
                av[lane]=row<rank?fastcd_ff_prime_mul(input[row][j],r,&p->field):0;
                wv[lane]=row<rank?fastcd_ff_prime_mul(output[row][j],ri,&p->field):0;
                qv[lane]=(uint32_t)(((uint64_t)av[lane]<<32)/prime);
                qh[lane]=(uint32_t)(((uint64_t)wv[lane]<<32)/prime);
            }}
            a[j]={p}_loadu_si{bits}((const void*)av);ah[j]={p}_loadu_si{bits}((const void*)qv);
            w[j]={p}_loadu_si{bits}((const void*)wv);wh[j]={p}_loadu_si{bits}((const void*)qh);
        }}
        for(size_t i=0;i<size;i++) {{
            Word vx={z},vy={z};
            for(size_t j=0;j<d;j++) {{
                vx=add(vx,fixed({p}_set1_epi32((int)x[i*d+j]),a[j],ah[j],&c),&c);
                vy=add(vy,fixed({p}_set1_epi32((int)y[i*d+j]),a[j],ah[j],&c),&c);
            }}
            ex[i]=vx;ey[i]=vy;
        }}
        if(loop) {name}_loop(eo,ex,ey,p->dimension,scratch,&c);
        else {name}_cd(eo,ex,ey,p->dimension,scratch,&c);
        for(size_t i=0;i<size;i++) for(size_t j=0;j<d;j++) {{
            uint32_t v=horizontal(fixed(eo[i],w[j],wh[j],&c),&c);
            uint32_t sum=out[i*d+j]+v;
            out[i*d+j]=sum>=prime?sum-prime:sum;
        }}
    }}
}}
"""
        if values:
            definitions = f"""#define VADD(a,b) add(a,b,ctx)
#define VSUB(a,b) sub(a,b,ctx)
#define VMUL(a,b) multiply(a,b,ctx)
#define VDIFF(a,b,x,y) difference(a,b,x,y,ctx)
#define VHALF(a) half(a,ctx)
#define VNORMALIZE(a) multiply(a,ctx->normalization,ctx)
#define VZERO {z}
"""
            source = source.replace(
                '#include "fast_schedule.h"',
                definitions + '#include "finite_value_schedule.h"',
            )
    path = WORK / f"{name}.c"
    path.write_text(source)
    if build:
        subprocess.run(
            [
                "cc",
                "-std=c11",
                "-O3",
                "-DNDEBUG",
                *target,
                "-fno-fast-math",
                "-ffp-contract=off",
                "-I" + str(ROOT / "tools/finite/finite"),
                "-I" + str(ROOT / "include"),
                "-I" + str(ROOT / "src/finite"),
                "-c",
                str(path),
                "-o",
                str(WORK / f"{name}.o"),
            ],
            check=True,
        )


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    for args in (
        (4, 128, "_mm", ["-msse4.1"]),
        (8, 256, "_mm256", ["-mavx2"]),
        (16, 512, "_mm512", ["-mavx512f", "-mavx512dq"]),
    ):
        generate(*args)
        generate(*args, fast=True)
        generate(*args, fast=True, values=True)
