#!/usr/bin/env python3
"""Structured extension transforms and descendant packing in 16-bit lanes.

Barrett fused differences require 2*p*p<2^16. Montgomery fused differences
require 2*p*p<2^15*2^16, giving p<2^15. Both include every conversion.
"""

import re
import subprocess

from finite_transpose import calls as transpose_calls
from finite_transpose import code as transpose_code
from generate_finite_multipoint import point_width
from generate_finite_short_prime import append as prime_entry
from generate_finite_shuffle import prime32
from generate_finite_simd import ROOT, WORK


def generate(mont, build=True, vector_packing=False):
    name = (
        ("tilemont" if mont else "tilebar")
        if vector_packing
        else ("medium_mont" if mont else "medium_bar")
    )
    base = (WORK / ("shortmont32.c" if mont else "shortbar32.c")).read_text()
    source = base[: base.index("#define FASTCD_FF_NAME")]
    if mont:
        difference = """
static inline Word difference(Word a,Word b,Word x,Word y,const Context* c){
 Word pc=_mm512_sub_epi16(c->prime,x),lo0=_mm512_mullo_epi16(a,b),lo1=_mm512_mullo_epi16(pc,y);
 Word lo=_mm512_add_epi16(lo0,lo1),carry=_mm512_maskz_set1_epi16(_mm512_cmplt_epu16_mask(lo,lo0),1);
 Word hi=_mm512_add_epi16(_mm512_add_epi16(_mm512_mulhi_epu16(a,b),_mm512_mulhi_epu16(pc,y)),carry);
 Word m=_mm512_mullo_epi16(lo,c->inverse);
 carry=_mm512_maskz_set1_epi16(_mm512_cmpneq_epi16_mask(lo,_mm512_setzero_si512()),1);
 Word u=_mm512_add_epi16(_mm512_add_epi16(hi,_mm512_mulhi_epu16(m,c->prime)),carry);
 Word t=_mm512_sub_epi16(u,c->prime);return _mm512_add_epi16(t,_mm512_and_si512(_mm512_srai_epi16(t,15),c->prime));
}
"""
    else:
        difference = """
static inline Word difference(Word a,Word b,Word x,Word y,const Context* c){
 Word t=_mm512_sub_epi16(_mm512_add_epi16(_mm512_mullo_epi16(a,b),_mm512_mullo_epi16(c->prime,c->prime)),_mm512_mullo_epi16(x,y));
 return fixed(t,_mm512_set1_epi16(1),c->inverse,c);
}
"""
    source += (
        difference
        + f"""
#define FASTCD_FF_NAME(n) {name}_##n
#define VADD(a,b) add(a,b,ctx)
#define VSUB(a,b) sub(a,b,ctx)
#define VMUL(a,b) multiply(a,b,ctx)
#define VDIFF(a,b,c,d) difference(a,b,c,d,ctx)
#define VHALF(a) half(a,ctx)
#define VNORMALIZE(a) multiply(a,ctx->normalization,ctx)
#define VZERO _mm512_setzero_si512()
#include "finite_value_schedule.h"
"""
    )
    nodes = prime32()
    nodes = nodes[
        nodes.index("static void shuffle32_node32(") : nodes.index(
            "static void shuffle32_prime_cd("
        )
    ]
    source += nodes.replace("shuffle32", name)
    small = (WORK / "sm32.c").read_text()
    first = small.index("static void sm32_points_cd8(")
    last = small.index("static Word gather_coordinates", first)
    source += small[first:last].replace("sm32", name)
    source += f"""
static void core(uint16_t* out,const uint16_t* x,const uint16_t* y,size_t n,uint16_t* w,size_t points,const Context* c){{
 switch(points){{case 8:{name}_points_cd8(out,x,y,n,w,c);break;case 16:{name}_points_cd16(out,x,y,n,w,c);break;
 case 32:{name}_cd((Word*)out,(const Word*)x,(const Word*)y,n,(Word*)w,c);break;}}
}}
static void transform(uint16_t* a,size_t n,size_t points,const Context* c){{switch(points){{case 8:{name}_points_hadamard8(a,n,c);break;
 case 16:{name}_points_hadamard16(a,n,c);break;case 32:{name}_hadamard((Word*)a,n,c);break;}}}}
static inline void save(uint16_t* out,Word a,size_t points){{uint16_t values[32];_mm512_storeu_si512((void*)values,a);memcpy(out,values,points*sizeof(uint16_t));}}
static Word gather(const uint16_t* data,size_t point,size_t points){{
 Word index=_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);index=_mm512_mullo_epi32(index,_mm512_set1_epi32((int)points));
 const uint16_t* pair=data+(point&~(size_t)1);
 Word lo=_mm512_i32gather_epi32(index,(const void*)pair,2),hi=_mm512_i32gather_epi32(index,(const void*)(pair+16*points),2);
 if(point&1){{lo=_mm512_srli_epi32(lo,16);hi=_mm512_srli_epi32(hi,16);}}
 return _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(lo)),_mm512_cvtepi32_epi16(hi),1);
}}
"""
    for d in range(2, 13):
        rank = 2 * d - 1
        P = point_width(rank, 32)
        G = (rank + P - 1) // P
        idx = "_mm512_setr_epi32(" + ",".join(str(i * d) for i in range(16)) + ")"
        if vector_packing:
            source += transpose_code(f"pack{d}", 32, 16, rank, P)
        source += f"""
static inline __attribute__((always_inline)) void rows{d}(uint16_t out[{rank}][32],const uint32_t* x,
 const uint32_t a[12][12],const uint32_t q[12][12],uint32_t residue,const Context* c){{
 Word xx[{d}],v[{d}],index={idx};
 for(size_t j=0;j<{d};j++){{Word lo=_mm512_i32gather_epi32(index,x+j,4),hi=_mm512_i32gather_epi32(index,x+j+16*{d},4);
  xx[j]=_mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(lo)),_mm512_cvtepi32_epi16(hi),1);
"""
        if mont:
            source += "  if(residue!=1)xx[j]=fixed(xx[j],_mm512_set1_epi16((short)residue),_mm512_set1_epi16((short)((residue<<16)/(uint16_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(c->prime)))),c);\n"
        else:
            source += "  (void)residue;\n"
        source += f""" }}
 v[0]=xx[0];v[1]=xx[1];for(size_t j=2;j<{d};j++)v[1]=add(v[1],xx[j],c);
 for(size_t k=2;k<{d};k++){{v[k]=fixed(xx[k],_mm512_set1_epi16((short)a[k][k]),_mm512_set1_epi16((short)q[k][k]),c);
  for(size_t j=k+1;j<{d};j++)v[k]=add(v[k],fixed(xx[j],_mm512_set1_epi16((short)a[k][j]),_mm512_set1_epi16((short)q[k][j]),c),c);}}
 for(size_t r=0;r<{rank};r++){{_mm512_storeu_si512((void*)out[r],v[0]);for(size_t k=0;k<{d - 1};k++)v[k]=add(v[k],v[k+1],c);}}
}}
static void degree{d}(uint32_t* restrict out,const uint32_t* x,const uint32_t* y,const FastCDPlanFF* p,uint32_t* restrict work,int loop,
 const uint32_t output[48][12]){{
 uint32_t prime=p->field.prime,inv=1;for(int k=0;k<4;k++)inv*=2-prime*inv;
 uint32_t residue={"65536%prime" if mont else "1"},ri=1,power=residue,exponent=prime-2;
 while(exponent){{if(exponent&1)ri=fastcd_ff_prime_mul(ri,power,&p->field);power=fastcd_ff_prime_mul(power,power,&p->field);exponent>>=1;}}
 Context c={{_mm512_set1_epi16((short)prime),_mm512_set1_epi16((short){"(0u-inv)" if mont else "(65536/prime)"}),_mm512_set1_epi16((short)fastcd_ff_prime_mul(p->inverse_dimension,residue,&p->field))}};
 uint32_t stirling[12][12]={{{{1}}}},a[12][12]={{{{0}}}},q[12][12]={{{{0}}}},factorial=1;
 for(size_t j=1;j<{d};j++)for(size_t k=1;k<=j;k++){{uint32_t v=fastcd_ff_prime_mul((uint32_t)k,stirling[j-1][k],&p->field)+stirling[j-1][k-1];stirling[j][k]=v>=prime?v-prime:v;}}
 for(size_t k=0;k<{d};k++){{if(k)factorial=fastcd_ff_prime_mul(factorial,(uint32_t)k,&p->field);for(size_t j=k;j<{d};j++){{a[k][j]=fastcd_ff_prime_mul(factorial,stirling[j][k],&p->field);q[k][j]=(a[k][j]<<16)/prime;}}}}
 size_t n=p->dimension,size=n*(loop?2:1),words=n*{P};uint16_t* state=(uint16_t*)work;uint16_t* tail=state+(loop?4:2)*words*{G};
 for(size_t i=0;i<n;i+=32){{uint16_t xl[{rank}][32],yl[{rank}][32],xu[{rank}][32],yu[{rank}][32];
  rows{d}(xl,x+i*{d},(const uint32_t (*)[12])a,(const uint32_t (*)[12])q,residue,&c);rows{d}(yl,y+i*{d},(const uint32_t (*)[12])a,(const uint32_t (*)[12])q,residue,&c);
  if(loop){{rows{d}(xu,x+(n+i)*{d},(const uint32_t (*)[12])a,(const uint32_t (*)[12])q,residue,&c);rows{d}(yu,y+(n+i)*{d},(const uint32_t (*)[12])a,(const uint32_t (*)[12])q,residue,&c);}}
  for(size_t coord=0;coord<32;coord++)for(size_t group=0;group<{G};group++){{uint16_t av[32],bv[32],ap[32],bp[32];
   for(size_t k=0;k<32;k++){{size_t r=group*{P}+k;av[k]=k<{P}&&r<{rank}?xl[r][coord]:0;bv[k]=k<{P}&&r<{rank}?yl[r][coord]:0;
    if(loop){{ap[k]=k<{P}&&r<{rank}?xu[r][coord]:0;bp[k]=k<{P}&&r<{rank}?yu[r][coord]:0;}}}}
   Word ax=_mm512_loadu_si512((const void*)av),by=_mm512_loadu_si512((const void*)bv);uint16_t* s=state+group*(loop?4:2)*words;
   if(loop){{Word au=_mm512_loadu_si512((const void*)ap),bu=_mm512_loadu_si512((const void*)bp);
    save(s+(i+coord)*{P},add(ax,au,&c),{P});save(s+words+(i+coord)*{P},sub(ax,au,&c),{P});
    save(s+2*words+(i+coord)*{P},add(by,bu,&c),{P});save(s+3*words+(i+coord)*{P},sub(by,bu,&c),{P});
   }}else{{save(s+(i+coord)*{P},ax,{P});save(s+words+(i+coord)*{P},by,{P});}}
  }}
 }}
 for(size_t group=0;group<{G};group++){{uint16_t* s=state+group*(loop?4:2)*words;
  if(loop){{uint16_t* f=s;uint16_t* u=f+words;uint16_t* fp=u+words;uint16_t* up=fp+words;uint16_t* g=tail;
   transform(f,n,{P},&c);transform(fp,n,{P},&c);for(size_t i=0;i<words;i+=32){{Word v=multiply(_mm512_loadu_si512((const void*)(f+i)),_mm512_loadu_si512((const void*)(fp+i)),&c);_mm512_storeu_si512((void*)(f+i),v);}}
   transform(f,n,{P},&c);if(p->inverse_dimension!=1)for(size_t i=0;i<words;i+=32){{Word v=multiply(_mm512_loadu_si512((const void*)(f+i)),c.normalization,&c);_mm512_storeu_si512((void*)(f+i),v);}}
   core(g,u,up,n,g+words,{P},&c);
   for(size_t i=0;i<words;i+=32){{Word left=_mm512_loadu_si512((const void*)(f+i)),right=_mm512_loadu_si512((const void*)(g+i));_mm512_storeu_si512((void*)(f+i),half(add(left,right,&c),&c));_mm512_storeu_si512((void*)(u+i),half(sub(left,right,&c),&c));}}
  }}else{{core(tail,s,s+words,n,tail+words,{P},&c);memcpy(s,tail,words*sizeof(uint16_t));}}
 }}
 uint32_t weight[{rank}][{d}],quotient[{rank}][{d}];for(size_t r=0;r<{rank};r++)for(size_t j=0;j<{d};j++){{weight[r][j]=fastcd_ff_prime_mul(output[r][j],ri,&p->field);quotient[r][j]=(weight[r][j]<<16)/prime;}}
 for(size_t i=0;i<size;i+=32){{Word values[{rank}];for(size_t r=0;r<{rank};r++){{const uint16_t* base=state+(r/{P})*(loop?4:2)*words+i*{P};values[r]=gather(base,r%{P},{P});}}
  for(size_t first=0;first<{d};first+=4){{Word acc[4][4];for(size_t col=0;col<4;col++)for(size_t k=0;k<4;k++)acc[col][k]=_mm512_setzero_si512();
   for(size_t r=0;r<{rank};r+=4){{
#pragma GCC unroll 4
    for(size_t k=0;k<4;k++)if(r+k<{rank}){{
#pragma GCC unroll 4
     for(size_t col=0;col<4;col++)if(first+col<{d})acc[col][k]=add(acc[col][k],fixed(values[r+k],_mm512_set1_epi16((short)weight[r+k][first+col]),_mm512_set1_epi16((short)quotient[r+k][first+col]),&c),&c);
    }}
   }}
   for(size_t col=0;col<4;col++)if(first+col<{d}){{Word v=add(add(acc[col][0],acc[col][1],&c),add(acc[col][2],acc[col][3],&c),&c);Word index={idx};
    _mm512_i32scatter_epi32(out+i*{d}+first+col,index,_mm512_cvtepu16_epi32(_mm512_castsi512_si256(v)),4);
    _mm512_i32scatter_epi32(out+(i+16)*{d}+first+col,index,_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v,1)),4);
   }}
  }}
 }}
}}
"""
    source += f"""void {name}_tensor(uint32_t* out,const uint32_t* x,const uint32_t* y,const FastCDPlanFF* p,uint32_t* work,int loop,size_t rank,
 const uint32_t input[48][12],const uint32_t output[48][12]){{(void)rank;(void)input;switch(p->field.degree){{\n"""
    for d in range(2, 13):
        source += f"case {d}:degree{d}(out,x,y,p,work,loop,output);break;\n"
    source += "}}\n"
    if vector_packing:
        for d in range(2, 13):
            rank = 2 * d - 1
            P = point_width(rank, 32)
            G = (rank + P - 1) // P
            first = source.index(f"static void degree{d}(")
            start = source.index("  for(size_t coord=0;coord<32;coord++)", first)
            end = source.index(
                f" for(size_t group=0;group<{G};group++){{uint16_t* s=state", start
            )
            source = (
                source[:start]
                + transpose_calls(f"pack{d}", "uint16_t", 32, P, G, "words")
                + " }\n"
                + source[end:]
            )
    prime = prime_entry("", name, mont)
    helper = prime[
        prime.index("static inline Word normalize(") : prime.index(
            "static uint32_t prime_horizontal("
        )
    ]
    horizontal = prime[
        prime.index("static uint32_t prime_horizontal(") : prime.index(
            f"static void {name}_node32("
        )
    ]
    tail = prime[prime.index(f"static void {name}_prime_cd(") :]
    source += helper + horizontal + tail
    for text in ("for(size_t col=0;col<4;col++)", "for(size_t k=0;k<4;k++)"):
        source = source.replace(text, "\n#pragma GCC unroll 4\n" + text)
    for d in range(2, 13):
        for text in (
            f"for(size_t j=0;j<{d};j++)",
            f"for(size_t k=2;k<{d};k++)",
            f"for(size_t j=k+1;j<{d};j++)",
            f"for(size_t k=0;k<{d - 1};k++)",
            f"for(size_t first=0;first<{d};first+=4)",
        ):
            source = source.replace(text, "\n#pragma GCC unroll 12\n" + text)
    source = re.sub(
        r"(?:\s*#pragma GCC unroll [0-9]+\s*\n){2,}",
        lambda m: (
            "\n#pragma GCC unroll "
            + str(max(map(int, re.findall(r"unroll ([0-9]+)", m.group(0)))))
            + "\n"
        ),
        source,
    )
    path = WORK / f"{name}.c"
    path.write_text(source)
    if build:
        subprocess.run(
            [
                "cc",
                "-O3",
                "-std=c11",
                "-mavx512f",
                "-mavx512bw",
                "-Iinclude",
                "-Itools",
                "-c",
                str(path),
                "-o",
                str(WORK / f"{name}.o"),
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    for mode in (False, True):
        generate(mode)
