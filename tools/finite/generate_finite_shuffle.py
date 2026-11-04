#!/usr/bin/env python3
"""Packed exact arithmetic for p<8 using byte-shuffle modular tables.

Canonical sums are <16, products <64 and ab+p^2-cd<128. Two 16-byte
nibble tables therefore reduce all products exactly. These bounds define
the eligibility class, independently of modulus, dimension or benchmark.
"""

import re
import subprocess

from finite_vector_packing import optimize_nodes
from generate_finite_prime_simd import append as prime_source
from generate_finite_simd import ROOT, WORK
from generate_finite_simd import generate as simd_source


def common(lanes, bits, p):
    name = f"shuffle{lanes}"

    def broadcast(a):
        return (
            a
            if bits == 128
            else f"{p}_broadcastsi128_si256({a})"
            if bits == 256
            else f"{p}_broadcast_i32x4({a})"
        )

    tables = "static const uint8_t small_products[8][8][16]={\n"
    for prime in range(8):
        tables += "    {\n"
        for a in range(8):
            tables += (
                "        {"
                + ",".join(str((a * b) % prime if prime else 0) for b in range(16))
                + "},\n"
            )
        tables += "    },\n"
    tables += "};\nstatic const uint8_t high_factor[8]={0,0,0,1,0,1,4,2};\n"
    code = f"""
#include <immintrin.h>
#include <string.h>
#include "fastcd/finite.h"
typedef __m{bits}i Word __attribute__((aligned(1),may_alias));
typedef struct {{Word prime,square,lo,hi,half,norm;const uint8_t (*rows)[16];}} Context;
{tables}
static inline Word add(Word a,Word b,const Context* c) {{return {p}_shuffle_epi8(c->lo,{p}_add_epi16(a,b));}}
static inline Word sub(Word a,Word b,const Context* c) {{return {p}_shuffle_epi8(c->lo,{p}_sub_epi16({p}_add_epi16(a,c->prime),b));}}
static inline Word reduce(Word a,const Context* c) {{
    return add({p}_shuffle_epi8(c->lo,a),{p}_shuffle_epi8(c->hi,{p}_srli_epi16(a,4)),c);
}}
static inline Word multiply(Word a,Word b,const Context* c) {{return reduce({p}_mullo_epi16(a,b),c);}}
static inline Word difference(Word a,Word b,Word x,Word y,const Context* c) {{
    return reduce({p}_sub_epi16({p}_add_epi16({p}_mullo_epi16(a,b),c->square),{p}_mullo_epi16(x,y)),c);
}}
static inline Word half(Word a,const Context* c) {{return {p}_shuffle_epi8(c->half,a);}}
static inline Word normalize(Word a,const Context* c) {{return {p}_shuffle_epi8(c->norm,a);}}
static inline Word scalar(uint32_t a,Word b,const Context* c) {{
    /* Canonical scalars are below eight. Bound the address even if a caller
     * violates that precondition or mutates shared storage after validation;
     * this does not define arithmetic for noncanonical operands. */
    Word row={broadcast("_mm_loadu_si128((const void*)c->rows[a & 7u])")};return {p}_shuffle_epi8(row,b);
}}
static Context context(const FastCDPlanFF* p) {{
    uint32_t prime=p->field.prime;const uint8_t (*rows)[16]=small_products[prime];
    Context c={{{p}_set1_epi16((short)prime),{p}_set1_epi16((short)(prime*prime)),
        {broadcast("_mm_loadu_si128((const void*)rows[1])")},{broadcast("_mm_loadu_si128((const void*)rows[high_factor[prime]])")},
        {broadcast("_mm_loadu_si128((const void*)rows[p->inverse_two])")},{broadcast("_mm_loadu_si128((const void*)rows[p->inverse_dimension])")},rows}};
    return c;
}}
#define FASTCD_FF_NAME(n) {name}_##n
#define VADD(a,b) add(a,b,ctx)
#define VSUB(a,b) sub(a,b,ctx)
#define VMUL(a,b) multiply(a,b,ctx)
#define VDIFF(a,b,x,y) difference(a,b,x,y,ctx)
#define VHALF(a) half(a,ctx)
#define VNORMALIZE(a) normalize(a,ctx)
#define VZERO {p}_setzero_si{bits}()
#include "finite_value_schedule.h"
"""
    return code


def gather_function(lanes, bits, p):
    if bits == 512:
        indices = f"{p}_setr_epi32(" + ",".join(str(j * lanes) for j in range(16)) + ")"
        return f"""
static Word gather_coordinates(const uint16_t* data,size_t point) {{
    const uint16_t* pair=data+(point&~(size_t)1);Word index={indices};
    Word lo={p}_i32gather_epi32(index,(const void*)pair,2);
    Word hi={p}_i32gather_epi32(index,(const void*)(pair+16*{lanes}),2);
    if(point&1){{lo={p}_srli_epi32(lo,16);hi={p}_srli_epi32(hi,16);}}
    __m256i a={p}_cvtepi32_epi16(lo),b={p}_cvtepi32_epi16(hi);
    return {p}_inserti64x4({p}_castsi256_si512(a),b,1);
}}
"""
    if bits == 256:
        indices = f"{p}_setr_epi32(" + ",".join(str(j * lanes) for j in range(8)) + ")"
        return f"""
static Word gather_coordinates(const uint16_t* data,size_t point) {{
    const uint16_t* pair=data+(point&~(size_t)1);Word index={indices};
    Word lo={p}_i32gather_epi32((const void*)pair,index,2),hi={p}_i32gather_epi32((const void*)(pair+8*{lanes}),index,2);
    if(point&1){{lo={p}_srli_epi32(lo,16);hi={p}_srli_epi32(hi,16);}}
    lo={p}_and_si256(lo,{p}_set1_epi32(65535));hi={p}_and_si256(hi,{p}_set1_epi32(65535));
    return {p}_permute4x64_epi64({p}_packus_epi32(lo,hi),0xd8);
}}
"""
    return """
static Word gather_coordinates(const uint16_t* data,size_t point) {
    uint16_t values[8];for(size_t i=0;i<8;i++)values[i]=data[i*8+point];
    return _mm_loadu_si128((const void*)values);
}
"""


def projection(lanes, bits, p):
    if bits == 512:
        idx = f"{p}_setr_epi32(" + ",".join(f"{j}*(int)d" for j in range(16)) + ")"
        old = f"""Word index={idx};
                    Word lo={p}_i32gather_epi32(index,out+i*d+column,4),hi={p}_i32gather_epi32(index,out+(i+16)*d+column,4);
                    __m256i a={p}_cvtepi32_epi16(lo),b={p}_cvtepi32_epi16(hi);
                    Word previous={p}_inserti64x4({p}_castsi256_si512(a),b,1);value=add(previous,value,c);"""
        save = f"""Word index={idx};
                Word lo={p}_cvtepu16_epi32({p}_castsi512_si256(value)),hi={p}_cvtepu16_epi32({p}_extracti64x4_epi64(value,1));
                {p}_i32scatter_epi32(out+i*d+column,index,lo,4);{p}_i32scatter_epi32(out+(i+16)*d+column,index,hi,4);"""
    else:
        old = f"""uint16_t previous[{lanes}];for(size_t lane=0;lane<{lanes};lane++)previous[lane]=(uint16_t)out[(i+lane)*d+column];
                    value=add(value,{p}_loadu_si{bits}((const void*)previous),c);"""
        save = f"""uint16_t canonical[{lanes}];{p}_storeu_si{bits}((void*)canonical,value);
                for(size_t lane=0;lane<{lanes};lane++)out[(i+lane)*d+column]=canonical[lane];"""
    return f"""
static void project(uint32_t* out,const Word* result,size_t size,size_t d,size_t first,size_t rank,
    const uint32_t weights[48][12],const Context* restrict c) {{
    size_t count=rank-first<{lanes}?rank-first:{lanes};
    for(size_t i=0;i<size;i+={lanes}) {{
        Word values[{lanes}];const uint16_t* data=(const uint16_t*)(result+i);
        for(size_t r=0;r<count;r++)values[r]=gather_coordinates(data,r);
        for(size_t col=0;col<d;col+=4) {{
            Word acc[4];
#pragma GCC unroll 4
            for(size_t j=0;j<4;j++)if(col+j<d)acc[j]=scalar(weights[first][col+j],values[0],c);
            for(size_t r=1;r<count;r++){{
#pragma GCC unroll 4
                for(size_t j=0;j<4;j++)if(col+j<d)acc[j]=add(acc[j],scalar(weights[first+r][col+j],values[r],c),c);
            }}
#pragma GCC unroll 4
            for(size_t j=0;j<4;j++)if(col+j<d) {{
                size_t column=col+j;Word value=acc[j];
                if(first) {{{old}}}
                {save}
            }}
        }}
    }}
}}
"""


def prime32():
    # Emit the same validated packing pattern for uint16 scalar planes. The
    # temporary 32-bit spelling is never compiled and is removed afterwards.
    args = (32, 512, "_mm512", ["-mavx512f", "-mavx512bw"])
    simd_source(*args, fast=True, values=True, build=False)
    prime_source(*args, build=False)
    text = (WORK / "value32.c").read_text()
    (WORK / "value32.c").unlink()
    first = text.index("static void value32_node32(")
    last = text.index("void value32_prime_tensor(", first)
    nodes = (
        text[first:last]
        .replace("value32", "shuffle32")
        .replace("uint32_t", "uint16_t")
        .replace("UINT32_MAX", "UINT16_MAX")
    )
    nodes = nodes.replace("_epi32", "_epi16")
    # Gather both descendant signs from two contiguous vectors of child data.
    for batch in (1, 2, 4, 8, 16):
        begin = nodes.index(f"static void shuffle32_node{batch}(")
        end = nodes.find("\nstatic void ", begin + 1)
        if end < 0:
            end = len(nodes)
        part = nodes[begin:end]
        first_gather = part.index("        Word pr=")
        last_gather = part.index("        Word fr=", first_gather)
        idx = [r * 2 * batch + b for r in range(32 // batch) for b in range(batch)]
        plus = "_mm512_setr_epi16(" + ",".join(map(str, idx)) + ")"
        minus = "_mm512_setr_epi16(" + ",".join(str(q + batch) for q in idx) + ")"
        replacement = f"""        Word ar=_mm512_loadu_si512((const void*)(cr+i*{2 * batch})),br=_mm512_loadu_si512((const void*)(cr+i*{2 * batch}+32));
        Word ai=_mm512_loadu_si512((const void*)(ci+i*{2 * batch})),bi=_mm512_loadu_si512((const void*)(ci+i*{2 * batch}+32));
        Word pr=_mm512_permutex2var_epi16(ar,{plus},br),pi=_mm512_permutex2var_epi16(ai,{plus},bi);
        Word mr=_mm512_permutex2var_epi16(ar,{minus},br),mi=_mm512_permutex2var_epi16(ai,{minus},bi);
"""
        part = part[:first_gather] + replacement + part[last_gather:]
        nodes = nodes[:begin] + part + nodes[end:]
    assert "gather_" not in nodes
    nodes = re.sub(
        r"_mm512_setr_epi16\(([^()]*)\)",
        lambda m: "_mm512_set_epi16(" + ",".join(reversed(m.group(1).split(","))) + ")",
        nodes,
    )
    nodes = optimize_nodes(nodes, "shuffle32", 32, 512, "_mm512", 16)
    return (
        """
static uint32_t horizontal(Word a,const Context* c) {
    uint32_t prime=(uint16_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(c->prime));
    return (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(a,_mm512_set1_epi16(1)))%prime;
}
"""
        + nodes
        + """
void shuffle32_prime(uint32_t* out,const uint32_t* x,const uint32_t* y,const FastCDPlanFF* p,uint32_t* work,int loop) {
    Context c=context(p);size_t n=p->dimension,size=n*(loop?2:1);
    uint16_t* ex=(uint16_t*)work;uint16_t* ey=ex+size;uint16_t* result=ey+size;uint16_t* scratch=result+size;
    for(size_t i=0;i<size;i++){ex[i]=(uint16_t)x[i];ey[i]=(uint16_t)y[i];}
    if(loop) {
        uint16_t* f=scratch;uint16_t* u=f+n;uint16_t* fp=u+n;uint16_t* up=fp+n;uint16_t* g=up+n;
        for(size_t i=0;i<n;i+=32){Word a=_mm512_loadu_si512((const void*)(ex+i)),b=_mm512_loadu_si512((const void*)(ex+n+i));
            Word d=_mm512_loadu_si512((const void*)(ey+i)),e=_mm512_loadu_si512((const void*)(ey+n+i));
            _mm512_storeu_si512((void*)(f+i),add(a,b,&c));_mm512_storeu_si512((void*)(u+i),sub(a,b,&c));
            _mm512_storeu_si512((void*)(fp+i),add(d,e,&c));_mm512_storeu_si512((void*)(up+i),sub(d,e,&c));}
        shuffle32_prime_hadamard(f,n,&c);shuffle32_prime_hadamard(fp,n,&c);
        for(size_t i=0;i<n;i+=32){Word a=multiply(_mm512_loadu_si512((const void*)(f+i)),_mm512_loadu_si512((const void*)(fp+i)),&c);_mm512_storeu_si512((void*)(f+i),a);}
        shuffle32_prime_hadamard(f,n,&c);shuffle32_prime_cd(g,u,up,n,g+n,&c);
        for(size_t i=0;i<n;i+=32){Word a=normalize(_mm512_loadu_si512((const void*)(f+i)),&c),b=_mm512_loadu_si512((const void*)(g+i));
            _mm512_storeu_si512((void*)(result+i),half(add(a,b,&c),&c));_mm512_storeu_si512((void*)(result+n+i),half(sub(a,b,&c),&c));}
    }else shuffle32_prime_cd(result,ex,ey,n,scratch,&c);
    for(size_t i=0;i<size;i++)out[i]=result[i];
}
"""
    )


def generate(lanes, bits, p, flags, build=True):
    name = f"shuffle{lanes}"
    source = common(lanes, bits, p)
    source += gather_function(lanes, bits, p) + projection(lanes, bits, p)
    source += f"""
void {name}_tensor(uint32_t* out,const uint32_t* x,const uint32_t* y,const FastCDPlanFF* p,
    uint32_t* work,int loop,size_t rank,const uint32_t input[48][12],const uint32_t output[48][12]) {{
    Context c=context(p);
    size_t n=p->dimension,size=n*(loop?2:1),d=p->field.degree;
    size_t cd_scratch=n>=8?2*n-16:0;
    Word* data=(Word*)work;Word* result=data+(loop?5*n:2*n)+cd_scratch;
    for(size_t first=0;first<rank;first+={lanes}) {{
        Word a[12];
        for(size_t j=0;j<d;j++){{uint16_t values[{lanes}];for(size_t lane=0;lane<{lanes};lane++)values[lane]=(uint16_t)(first+lane<rank?input[first+lane][j]:0);
            a[j]={p}_loadu_si{bits}((const void*)values);}}
        if(loop) {{
            Word* f=data;Word* u=f+n;Word* fp=u+n;Word* up=fp+n;Word* g=up+n;Word* scratch=g+n;
            for(size_t i=0;i<n;i++){{
                Word xl=scalar(x[i*d],a[0],&c),xu=scalar(x[(n+i)*d],a[0],&c);
                Word yl=scalar(y[i*d],a[0],&c),yu=scalar(y[(n+i)*d],a[0],&c);
                for(size_t j=1;j<d;j++){{xl=add(xl,scalar(x[i*d+j],a[j],&c),&c);xu=add(xu,scalar(x[(n+i)*d+j],a[j],&c),&c);
                    yl=add(yl,scalar(y[i*d+j],a[j],&c),&c);yu=add(yu,scalar(y[(n+i)*d+j],a[j],&c),&c);}}
                f[i]=add(xl,xu,&c);u[i]=sub(xl,xu,&c);fp[i]=add(yl,yu,&c);up[i]=sub(yl,yu,&c);
            }}
            {name}_hadamard(f,n,&c);{name}_hadamard(fp,n,&c);
            for(size_t i=0;i<n;i++)f[i]=multiply(f[i],fp[i],&c);
            {name}_hadamard(f,n,&c);
            if(p->inverse_dimension!=1)for(size_t i=0;i<n;i++)f[i]=normalize(f[i],&c);
            {name}_cd(g,u,up,n,scratch,&c);
            for(size_t i=0;i<n;i++){{result[i]=half(add(f[i],g[i],&c),&c);result[n+i]=half(sub(f[i],g[i],&c),&c);}}
        }}else{{
            Word* ex=data;Word* ey=ex+n;Word* scratch=ey+n;
            for(size_t i=0;i<n;i++){{Word vx=scalar(x[i*d],a[0],&c),vy=scalar(y[i*d],a[0],&c);
                for(size_t j=1;j<d;j++){{vx=add(vx,scalar(x[i*d+j],a[j],&c),&c);vy=add(vy,scalar(y[i*d+j],a[j],&c),&c);}}
                ex[i]=vx;ey[i]=vy;
            }}
            {name}_cd(result,ex,ey,n,scratch,&c);
        }}
        project(out,result,size,d,first,rank,output,&c);
    }}
}}
"""
    if lanes == 32:
        source += prime32()
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
            "-I" + str(ROOT / "src/finite"),
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
        generate(*args)
