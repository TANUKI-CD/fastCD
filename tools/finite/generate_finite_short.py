#!/usr/bin/env python3
"""Packed uint16 arithmetic: canonical Barrett for p<256, Montgomery p<2^15."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".private/finite-optimization/representations"


def generate(lanes, bits, prefix, flags, mont, build=True):
    p = prefix
    v = f"__m{bits}i"
    name = f"short{'mont' if mont else 'bar'}{lanes}"

    def And(a, b):
        return f"{p}_and_si{bits}({a},{b})"

    z = f"{p}_setzero_si{bits}()"
    if mont:
        is_zero = (
            f"{p}_maskz_set1_epi16({p}_cmpeq_epi16_mask(lo,{z}),1)"
            if bits == 512
            else And(f"{p}_cmpeq_epi16(lo,{z})", f"{p}_set1_epi16(1)")
        )
        multiply = f"""
    Word lo={p}_mullo_epi16(a,b),hi={p}_mulhi_epu16(a,b);
    Word m={p}_mullo_epi16(lo,c->inverse);
    Word carry={is_zero};
    Word u={p}_add_epi16(hi,{p}_mulhi_epu16(m,c->prime));
    u={p}_add_epi16(u,{p}_sub_epi16({p}_set1_epi16(1),carry));
    Word t={p}_sub_epi16(u,c->prime);"""
    else:
        multiply = f"""
    Word t={p}_mullo_epi16(a,b);
    Word q={p}_mulhi_epu16(t,c->inverse);
    t={p}_sub_epi16({p}_sub_epi16(t,{p}_mullo_epi16(q,c->prime)),c->prime);"""
    if bits == 128:
        horizontal = """__m128i q=_mm_madd_epi16(a,_mm_set1_epi16(1));
    q=_mm_hadd_epi32(q,q);q=_mm_hadd_epi32(q,q);
    return (uint32_t)_mm_cvtsi128_si32(q);"""
    elif bits == 256:
        horizontal = """__m256i q=_mm256_madd_epi16(a,_mm256_set1_epi16(1));
    q=_mm256_hadd_epi32(q,q);q=_mm256_hadd_epi32(q,q);
    __m128i t=_mm_add_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
    return (uint32_t)_mm_cvtsi128_si32(t);"""
    else:
        horizontal = """__m512i q=_mm512_madd_epi16(a,_mm512_set1_epi16(1));
    return (uint32_t)_mm512_reduce_add_epi32(q);"""
    source = f"""
#include <immintrin.h>
#include <string.h>
#include "fastcd/finite.h"
typedef {v} Word __attribute__((aligned(1),may_alias));
typedef struct {{Word prime,inverse,normalization;}} Context;
static inline Word add(Word a,Word b,const Context* c) {{
    Word t={p}_sub_epi16({p}_add_epi16(a,b),c->prime);
    return {p}_add_epi16(t,{And(f"{p}_srai_epi16(t,15)", "c->prime")});
}}
static inline Word sub(Word a,Word b,const Context* c) {{
    Word t={p}_sub_epi16(a,b);
    return {p}_add_epi16(t,{And(f"{p}_srai_epi16(t,15)", "c->prime")});
}}
static inline Word half(Word a,const Context* c) {{
    Word odd={p}_sub_epi16({z},{And("a", f"{p}_set1_epi16(1)")});
    return {p}_srli_epi16({p}_add_epi16(a,{And("odd", "c->prime")}),1);
}}
static inline Word multiply(Word a,Word b,const Context* c) {{
    {multiply}
    return {p}_add_epi16(t,{And(f"{p}_srai_epi16(t,15)", "c->prime")});
}}
static inline Word fixed(Word a,Word b,Word quotient,const Context* c) {{
    Word q={p}_mulhi_epu16(a,quotient);
    Word t={p}_sub_epi16({p}_sub_epi16({p}_mullo_epi16(a,b),{p}_mullo_epi16(q,c->prime)),c->prime);
    return {p}_add_epi16(t,{And(f"{p}_srai_epi16(t,15)", "c->prime")});
}}
#define FASTCD_FF_NAME(n) {name}_##n
#define VADD(a,b) add(a,b,ctx)
#define VSUB(a,b) sub(a,b,ctx)
#define VMUL(a,b) multiply(a,b,ctx)
#define VDIFF(a,b,x,y) sub(multiply(a,b,ctx),multiply(x,y,ctx),ctx)
#define VHALF(a) half(a,ctx)
#define VNORMALIZE(a) multiply(a,ctx->normalization,ctx)
#define VZERO {z}
#include "finite_value_schedule.h"
static uint32_t horizontal(Word a) {{
    {horizontal}
}}
void {name}_tensor(uint32_t* restrict out,const uint32_t* x,const uint32_t* y,
    const FastCDPlanFF* p,uint32_t* restrict work,int loop,size_t rank,
    const uint32_t input[48][12],const uint32_t output[48][12]) {{
    uint32_t prime=p->field.prime,inv=1;
    for(int i=0;i<4;i++) inv*=2-prime*inv;
    uint32_t r={"65536%prime" if mont else "1"},ri=1,b=r,e=prime-2;
    while(e) {{if(e&1)ri=fastcd_ff_prime_mul(ri,b,&p->field);b=fastcd_ff_prime_mul(b,b,&p->field);e>>=1;}}
    Context c={{ {p}_set1_epi16((short)prime),{p}_set1_epi16((short){"(0u-inv)" if mont else "(65536/prime)"}),
        {p}_set1_epi16((short)fastcd_ff_prime_mul(p->inverse_dimension,r,&p->field)) }};
    size_t size=p->dimension*(loop?2:1),d=p->field.degree;
    Word* ex=(Word*)work;Word* ey=ex+size;Word* eo=ey+size;Word* scratch=eo+size;
    memset(out,0,size*d*sizeof(uint32_t));
    for(size_t first=0;first<rank;first+={lanes}) {{
        Word a[12],ah[12],w[12],wh[12];
        for(size_t j=0;j<d;j++) {{
            uint16_t av[{lanes}],qv[{lanes}],wv[{lanes}],qh[{lanes}];
            for(size_t lane=0;lane<{lanes};lane++) {{
                size_t row=first+lane;
                av[lane]=(uint16_t)(row<rank?fastcd_ff_prime_mul(input[row][j],r,&p->field):0);
                wv[lane]=(uint16_t)(row<rank?fastcd_ff_prime_mul(output[row][j],ri,&p->field):0);
                qv[lane]=(uint16_t)(((uint32_t)av[lane]<<16)/prime);
                qh[lane]=(uint16_t)(((uint32_t)wv[lane]<<16)/prime);
            }}
            a[j]={p}_loadu_si{bits}((const void*)av);ah[j]={p}_loadu_si{bits}((const void*)qv);
            w[j]={p}_loadu_si{bits}((const void*)wv);wh[j]={p}_loadu_si{bits}((const void*)qh);
        }}
        for(size_t i=0;i<size;i++) {{
            Word vx={z},vy={z};
            for(size_t j=0;j<d;j++) {{
                vx=add(vx,fixed({p}_set1_epi16((short)x[i*d+j]),a[j],ah[j],&c),&c);
                vy=add(vy,fixed({p}_set1_epi16((short)y[i*d+j]),a[j],ah[j],&c),&c);
            }}
            ex[i]=vx;ey[i]=vy;
        }}
        if(loop) {name}_loop(eo,ex,ey,p->dimension,scratch,&c);
        else {name}_cd(eo,ex,ey,p->dimension,scratch,&c);
        for(size_t i=0;i<size;i++) for(size_t j=0;j<d;j++) {{
            uint32_t v=fastcd_ff_field_reduce64(horizontal(fixed(eo[i],w[j],wh[j],&c)),&p->field);
            uint32_t sum=out[i*d+j]+v;out[i*d+j]=sum>=prime?sum-prime:sum;
        }}
    }}
}}
"""
    path = WORK / f"{name}.c"
    path.write_text(source)
    if not build:
        return
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-O3",
            "-DNDEBUG",
            *flags,
            "-I" + str(ROOT / "tools/finite/finite"),
            "-I" + str(ROOT / "include"),
            "-c",
            str(path),
            "-o",
            str(WORK / f"{name}.o"),
        ],
        check=True,
    )


if __name__ == "__main__":
    for args in (
        (8, 128, "_mm", ["-msse4.1"]),
        (16, 256, "_mm256", ["-mavx2"]),
        (32, 512, "_mm512", ["-mavx512f", "-mavx512bw"]),
    ):
        for mont in (False, True):
            generate(*args, mont)
