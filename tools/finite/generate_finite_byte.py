#!/usr/bin/env python3
"""Sixty-four canonical F_3 byte lanes, using AVX-512BW (no VBMI)."""

import subprocess

from generate_finite_simd import ROOT, WORK


def generate(build=True):
    old = (WORK / "slicevec4.c").read_text()
    source = """#include <immintrin.h>
#include <string.h>
#include "fastcd/finite.h"
typedef uint8_t Coef;
typedef __m512i Word __attribute__((aligned(1),may_alias));
typedef struct {int inverse;} Context;
static inline Coef sa(Coef a,Coef b){return (Coef)((a+b)%3);}
static inline Coef sn(Coef a){return (Coef)((3-a)%3);}
static inline Coef ss(Coef a,Coef b){return (Coef)((a+3-b)%3);}
static inline Coef sm(Coef a,Coef b){return (Coef)(a*b%3);}
static inline Word table(const uint8_t* t){return _mm512_broadcast_i32x4(_mm_loadu_si128((const void*)t));}
static inline Word add(Word a,Word b){static const uint8_t t[16]={0,1,2,0,1,2};return _mm512_shuffle_epi8(table(t),_mm512_add_epi8(a,b));}
static inline Word neg(Word a){static const uint8_t t[16]={0,2,1};return _mm512_shuffle_epi8(table(t),a);}
static inline Word sub(Word a,Word b){return add(a,neg(b));}
static inline Word multiply(Word a,Word b){static const uint8_t t[16]={0,0,0,0,0,1,2,0,0,2,1};return _mm512_shuffle_epi8(table(t),_mm512_add_epi8(a,_mm512_slli_epi16(b,2)));}
#define FASTCD_FF_NAME(n) byte64_##n
#define VADD(a,b) add(a,b)
#define VSUB(a,b) sub(a,b)
#define VMUL(a,b) multiply(a,b)
#define VDIFF(a,b,c,d) sub(multiply(a,b),multiply(c,d))
#define VHALF(a) neg(a)
#define VNORMALIZE(a) (ctx->inverse==1?(a):neg(a))
#define VZERO _mm512_setzero_si512()
#include "finite_value_schedule.h"
static inline Word ld(const Coef* a){return _mm512_loadu_si512((const void*)a);}
static inline void st(Coef* a,Word v){_mm512_storeu_si512((void*)a,v);}
static inline Word broadcast(Coef a){return _mm512_set1_epi8((char)a);}
static inline Word mask(Word a,Word m){return _mm512_and_si512(a,m);}
static inline Coef horizontal(Word a){return (Coef)(_mm512_reduce_add_epi64(_mm512_sad_epu8(a,_mm512_setzero_si512()))%3);}
static void byte64_node64(Coef* restrict r,Coef* restrict im,const Coef* xr,const Coef* xi,const Coef* yr,const Coef* yi,size_t n,Coef* restrict work,const Context* restrict ctx){
 byte64_pair_cross((Word*)r,(Word*)im,(const Word*)xr,(const Word*)xi,(const Word*)yr,(const Word*)yi,n,(Word*)work,ctx);}
"""

    def idx16(v):
        return "_mm512_set_epi16(" + ",".join(map(str, reversed(v))) + ")"

    def keep(B):
        return (
            "_mm512_set_epi8("
            + ",".join("0" if k < B else "-1" for k in reversed(range(64)))
            + ")"
        )

    def zipword(a, b, B, part):
        if B < 16:
            lo = f"_mm512_unpacklo_epi{8 * B}({a},{b})"
            hi = f"_mm512_unpackhi_epi{8 * B}({a},{b})"
        else:
            lo, hi = a, b
        ix = (
            ([0, 1, 8, 9, 2, 3, 10, 11] if part == 0 else [4, 5, 12, 13, 6, 7, 14, 15])
            if B <= 16
            else (
                [0, 1, 2, 3, 8, 9, 10, 11]
                if part == 0
                else [4, 5, 6, 7, 12, 13, 14, 15]
            )
        )
        return (
            f"_mm512_permutex2var_epi64({lo},_mm512_setr_epi64("
            + ",".join(map(str, ix))
            + f"),{hi})"
        )

    original = old[
        old.index("static void sv4_node2(") : old.index("static void sv4_node1(")
    ]
    tail = original[original.index(" Coef gg[4]") :]
    for B in (32, 16, 8, 4, 2, 1):
        C = 64 // B
        source += f"""static void byte64_node{B}(Coef* restrict real,Coef* restrict imag,const Coef* xr,const Coef* xi,const Coef* yr,const Coef* yi,size_t n,Coef* restrict work,const Context* restrict ctx){{
 size_t h=n/2,size=n*{B};Coef* cr=work+4*size;Coef* ci=cr+size;
 if(n>4){{Coef* ar=work;Coef* ai=ar+size;Coef* br=ai+size;Coef* bi=br+size;
  for(size_t i=0;i<h;i+={C}){{
   Word a=ld(xr+i*{B}),b=ld(xi+(h+i)*{B}),c=ld(xi+i*{B}),d=ld(xr+(h+i)*{B});
   Word e=ld(yr+i*{B}),f=ld(yi+(h+i)*{B}),g=ld(yi+i*{B}),t=ld(yr+(h+i)*{B});
   Word ap=sub(a,b),am=add(a,b),ip=add(c,d),im=sub(c,d),bp=sub(e,f),bm=add(e,f),jp=add(g,t),jm=sub(g,t);
   if(i==0){{Word keep={keep(B)};ap=mask(ap,keep);am=mask(am,keep);ip=mask(ip,keep);im=mask(im,keep);bp=mask(bp,keep);bm=mask(bm,keep);jp=mask(jp,keep);jm=mask(jm,keep);}}
"""
        for arr, a, b in [
            ("ar", "ap", "am"),
            ("ai", "ip", "im"),
            ("br", "bp", "bm"),
            ("bi", "jp", "jm"),
        ]:
            for part in (0, 1):
                source += (
                    f"   st({arr}+i*{2 * B}+{64 * part},{zipword(a, b, B, part)});\n"
                )
        source += f"  }}byte64_node{2 * B}(cr,ci,ar,ai,br,bi,h,ci+size,ctx);\n }}else memset(cr,0,2*size*sizeof(Coef));\n"
        t = (
            tail.replace("256", "512")
            .replace("sv4", "byte64")
            .replace("[4]", "[64]")
            .replace("k<4", "k<64")
            .replace("k%2", f"k%{B}")
            .replace("k+=2", f"k+={B}")
            .replace("i+=2", f"i+={C}")
            .replace("b<2", f"b<{B}")
        )
        t = (
            t.replace("h*2", f"h*{B}")
            .replace("i*2", f"i*{B}")
            .replace("(h+i)*2", f"(h+i)*{B}")
        )
        t = (
            t.replace("_mm512_setr_epi64x(0,0,-1,-1)", keep(B))
            .replace("(Coef){0,0}", "(Coef)0")
            .replace("{0,0}", "0")
        )
        start = t.index("  Word cr0,cr1,ci0,ci1;")
        end = t.index("  Word fr=", start)
        replacement = "  Word cr0,cr1,ci0,ci1;\n"
        for arr in ("cr", "ci"):
            for sign in (0, 1):
                lo = f"ld({arr}+i*{2 * B})"
                hi = f"ld({arr}+i*{2 * B}+64)"
                if B >= 2:
                    ix = [
                        r * B + b + sign * (B // 2)
                        for r in range(C)
                        for b in range(B // 2)
                    ]
                    expr = f"_mm512_permutex2var_epi16({lo},{idx16(ix)},{hi})"
                else:

                    def convert(v, sign=sign):
                        return (
                            f"_mm512_cvtepi16_epi8({v})"
                            if sign == 0
                            else f"_mm512_cvtepi16_epi8(_mm512_srli_epi16({v},8))"
                        )

                    expr = f"_mm512_inserti64x4(_mm512_castsi256_si512({convert(lo)}),{convert(hi)},1)"
                replacement += f"  {arr}{sign}={expr};\n"
        t = t[:start] + replacement + t[end:]
        pivot = t.index(" Coef aa[64]")
        t = t[:pivot]
        for width in (1, 2, 4, 8, 16, 32):
            if width < B:
                continue
            for arr in ("ap", "aq", "aw"):
                perm = (
                    f"_mm512_or_si512(_mm512_slli_epi16({arr},8),_mm512_srli_epi16({arr},8))"
                    if width == 1
                    else f"_mm512_permutexvar_epi16({idx16([i ^ (width // 2) for i in range(32)])},{arr})"
                )
                t += f" {arr}=add({arr},{perm});\n"
        m = (1 << B) - 1
        t += f" _mm512_mask_storeu_epi8(real,UINT64_C({m}),VZERO);_mm512_mask_storeu_epi8(imag,UINT64_C({m}),VZERO);\n"
        t += f" _mm512_mask_storeu_epi8(real+h*{B},UINT64_C({m}),sub(ap,aq));_mm512_mask_storeu_epi8(imag+h*{B},UINT64_C({m}),sub(sub(aw,ap),aq));\n}}\n"
        source += t
    root = old[old.index("static void sv4_root(") : old.index("static void sh(")]
    root = (
        root.replace("256", "512")
        .replace("sv4", "byte64")
        .replace("[4]", "[64]")
        .replace("k<4", "k<64")
        .replace("i+=4", "i+=64")
    )
    root = (
        root.replace("_mm512_setr_epi64x(0,-1,-1,-1)", keep(1))
        .replace("(Coef){0,0}", "(Coef)0")
        .replace("{0,0}", "0")
    )
    root = root.replace(
        "Word a=ld(x+i),b=ld(y+i);dot=add(dot,multiply(a,b));",
        "Word left=ld(x+i),right=ld(y+i);dot=add(dot,multiply(left,right));",
    ).replace(
        "multiply(alpha,b),multiply(beta,a)",
        "multiply(alpha,right),multiply(beta,left)",
    )
    root = root.replace(
        "Coef partial[64],a=0;st(partial,sum);for(int k=0;k<64;k++)a=sa(a,partial[k]);out[h]=a;",
        "out[h]=horizontal(sum);",
    )
    root = root.replace(
        "st(partial,dot);a=(Coef)0;for(int k=0;k<64;k++)a=sa(a,partial[k]);out[0]=ss(sn(sm(x[0],y[0])),a);",
        "out[0]=ss(sn(sm(x[0],y[0])),horizontal(dot));",
    )
    source += root
    source += "static void sh(Coef* data,size_t n){for(size_t base=0;base<n;base+=64){Word a=ld(data+base);\n"
    for width in (1, 2, 4, 8, 16, 32):
        perm = (
            "_mm512_or_si512(_mm512_slli_epi16(a,8),_mm512_srli_epi16(a,8))"
            if width == 1
            else f"_mm512_permutexvar_epi16({idx16([i ^ (width // 2) for i in range(32)])},a)"
        )
        m = sum(1 << i for i in range(64) if i & width)
        source += f" {{Word b={perm},s=add(a,b),t=sub(b,a);a=_mm512_mask_blend_epi8(UINT64_C({m}),s,t);}}\n"
    source += """ st(data+base,a);}for(size_t width=64;width<n;width*=2)for(size_t base=0;base<n;base+=2*width)for(size_t j=0;j<width;j+=64){Word a=ld(data+base+j),b=ld(data+base+width+j);st(data+base+j,add(a,b));st(data+base+width+j,sub(a,b));}}
static inline Word load_public(const uint32_t* x,size_t d,size_t j){
 if(d==1){Word v=_mm512_castsi128_si512(_mm512_cvtepi32_epi8(_mm512_loadu_si512((const void*)x)));v=_mm512_inserti32x4(v,_mm512_cvtepi32_epi8(_mm512_loadu_si512((const void*)(x+16))),1);v=_mm512_inserti32x4(v,_mm512_cvtepi32_epi8(_mm512_loadu_si512((const void*)(x+32))),2);return _mm512_inserti32x4(v,_mm512_cvtepi32_epi8(_mm512_loadu_si512((const void*)(x+48))),3);}
 Word ix=_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);ix=_mm512_mullo_epi32(ix,_mm512_set1_epi32((int)d));
 Word a=_mm512_i32gather_epi32(ix,x+j,4),b=_mm512_i32gather_epi32(ix,x+16*d+j,4),c=_mm512_i32gather_epi32(ix,x+32*d+j,4),e=_mm512_i32gather_epi32(ix,x+48*d+j,4);
 Word v=_mm512_castsi128_si512(_mm512_cvtepi32_epi8(a));v=_mm512_inserti32x4(v,_mm512_cvtepi32_epi8(b),1);v=_mm512_inserti32x4(v,_mm512_cvtepi32_epi8(c),2);return _mm512_inserti32x4(v,_mm512_cvtepi32_epi8(e),3);}
static inline void store_public(uint32_t* out,Word value,size_t d,size_t j){
 if(d==1){_mm512_storeu_si512((void*)out,_mm512_cvtepu8_epi32(_mm512_castsi512_si128(value)));_mm512_storeu_si512((void*)(out+16),_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(value,1)));_mm512_storeu_si512((void*)(out+32),_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(value,2)));_mm512_storeu_si512((void*)(out+48),_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(value,3)));return;}
 Word ix=_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);ix=_mm512_mullo_epi32(ix,_mm512_set1_epi32((int)d));
 _mm512_i32scatter_epi32(out+j,ix,_mm512_cvtepu8_epi32(_mm512_castsi512_si128(value)),4);
 _mm512_i32scatter_epi32(out+16*d+j,ix,_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(value,1)),4);
 _mm512_i32scatter_epi32(out+32*d+j,ix,_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(value,2)),4);
 _mm512_i32scatter_epi32(out+48*d+j,ix,_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(value,3)),4);}
static void finish(Coef* state,size_t n,Coef* tail,const Context* c){
 Coef* f=state;Coef* u=f+n;Coef* fp=u+n;Coef* up=fp+n;Coef* g=tail;
 sh(f,n);sh(fp,n);for(size_t i=0;i<n;i+=64)st(f+i,multiply(ld(f+i),ld(fp+i)));sh(f,n);
 if(c->inverse==2)for(size_t i=0;i<n;i+=64)st(f+i,neg(ld(f+i)));
 byte64_root(g,u,up,n,g+n,c);for(size_t i=0;i<n;i+=64){Word a=ld(f+i),b=ld(g+i);st(f+i,neg(add(a,b)));st(u+i,neg(sub(a,b)));}}
void byte64_prime(uint32_t* restrict out,const uint32_t* x,const uint32_t* y,const FastCDPlanFF* p,uint32_t* restrict work,int loop){
 size_t n=p->dimension;Coef* state=(Coef*)work;Context c={(int)p->inverse_dimension};
 if(loop){for(size_t i=0;i<n;i+=64){Word a=load_public(x+i,1,0),b=load_public(x+n+i,1,0),d=load_public(y+i,1,0),e=load_public(y+n+i,1,0);st(state+i,add(a,b));st(state+n+i,sub(a,b));st(state+2*n+i,add(d,e));st(state+3*n+i,sub(d,e));}finish(state,n,state+4*n,&c);}
 else{Coef* result=state+2*n;for(size_t i=0;i<n;i+=64){st(state+i,load_public(x+i,1,0));st(state+n+i,load_public(y+i,1,0));}byte64_root(result,state,state+n,n,result+n,&c);state=result;}
 for(size_t i=0;i<n*(loop?2:1);i+=64)store_public(out+i,ld(state+i),1,0);}
void byte64_quadratic(uint32_t* restrict out,const uint32_t* x,const uint32_t* y,const FastCDPlanFF* p,uint32_t* restrict work,int loop){
 size_t n=p->dimension,stride=n*(loop?4:3);Coef* state=(Coef*)work;Coef* tail=state+3*stride;Context c={(int)p->inverse_dimension};
 for(size_t i=0;i<n;i+=64){Word a=load_public(x+2*i,2,0),b=load_public(x+2*i,2,1),d=load_public(y+2*i,2,0),e=load_public(y+2*i,2,1);
  Word ax[3]={a,add(a,b),sub(a,b)},by[3]={d,add(d,e),sub(d,e)};
  if(loop){Word u=load_public(x+2*(n+i),2,0),v=load_public(x+2*(n+i),2,1),s=load_public(y+2*(n+i),2,0),t=load_public(y+2*(n+i),2,1);Word au[3]={u,add(u,v),sub(u,v)},bu[3]={s,add(s,t),sub(s,t)};
   for(size_t r=0;r<3;r++){Coef* q=state+r*stride;st(q+i,add(ax[r],au[r]));st(q+n+i,sub(ax[r],au[r]));st(q+2*n+i,add(by[r],bu[r]));st(q+3*n+i,sub(by[r],bu[r]));}}
  else for(size_t r=0;r<3;r++){Coef* q=state+r*stride;st(q+i,ax[r]);st(q+n+i,by[r]);}}
 for(size_t r=0;r<3;r++){Coef* q=state+r*stride;if(loop)finish(q,n,tail,&c);else byte64_root(q+2*n,q,q+n,n,tail,&c);}
 Word n0=broadcast((Coef)p->field.negative[0]),n1=broadcast((Coef)p->field.negative[1]);
 for(size_t i=0;i<n*(loop?2:1);i+=64){size_t offset=loop?i:2*n+i;Word a=ld(state+offset),b=ld(state+stride+offset),e=ld(state+2*stride+offset);Word c1=sub(e,b),c2=neg(add(add(a,b),e));store_public(out+2*i,add(a,multiply(c2,n0)),2,0);store_public(out+2*i,add(c1,multiply(c2,n1)),2,1);}}
"""
    helpers = "static inline void load_quadratic(const uint32_t* x,Word* a,Word* b){\n"
    for packet in range(4):
        even = "_mm512_setr_epi32(" + ",".join(map(str, range(0, 32, 2))) + ")"
        odd = "_mm512_setr_epi32(" + ",".join(map(str, range(1, 32, 2))) + ")"
        helpers += f" {{Word lo=_mm512_loadu_si512((const void*)(x+{32 * packet})),hi=_mm512_loadu_si512((const void*)(x+{32 * packet + 16}));\n"
        for name, ix in [("a", even), ("b", odd)]:
            packed = f"_mm512_cvtepi32_epi8(_mm512_permutex2var_epi32(lo,{ix},hi))"
            helpers += (
                f" *{name}=_mm512_castsi128_si512({packed});\n"
                if packet == 0
                else f" *{name}=_mm512_inserti32x4(*{name},{packed},{packet});\n"
            )
        helpers += " }\n"
    helpers += "}\nstatic inline void store_quadratic(uint32_t* out,Word a,Word b){\n"
    for packet in range(4):

        def extract(x, packet=packet):
            return (
                f"_mm512_castsi512_si128({x})"
                if packet == 0
                else f"_mm512_extracti32x4_epi32({x},{packet})"
            )

        helpers += f" {{Word aa=_mm512_cvtepu8_epi32({extract('a')}),bb=_mm512_cvtepu8_epi32({extract('b')});\n"
        for part in (0, 1):
            ix = (
                "_mm512_setr_epi32("
                + ",".join(
                    str(k) for i in range(part * 8, part * 8 + 8) for k in (i, i + 16)
                )
                + ")"
            )
            helpers += f" _mm512_storeu_si512((void*)(out+{32 * packet + 16 * part}),_mm512_permutex2var_epi32(aa,{ix},bb));\n"
        helpers += " }\n"
    helpers += "}\n"
    source = source.replace("static void finish(", helpers + "static void finish(")
    source = source.replace(
        "Word a=load_public(x+2*i,2,0),b=load_public(x+2*i,2,1),d=load_public(y+2*i,2,0),e=load_public(y+2*i,2,1);",
        "Word a,b,d,e;load_quadratic(x+2*i,&a,&b);load_quadratic(y+2*i,&d,&e);",
    )
    source = source.replace(
        "Word u=load_public(x+2*(n+i),2,0),v=load_public(x+2*(n+i),2,1),s=load_public(y+2*(n+i),2,0),t=load_public(y+2*(n+i),2,1);",
        "Word u,v,s,t;load_quadratic(x+2*(n+i),&u,&v);load_quadratic(y+2*(n+i),&s,&t);",
    )
    source = source.replace(
        "store_public(out+2*i,add(a,multiply(c2,n0)),2,0);store_public(out+2*i,add(c1,multiply(c2,n1)),2,1);",
        "store_quadratic(out+2*i,add(a,multiply(c2,n0)),add(c1,multiply(c2,n1)));",
    )
    path = WORK / "byte64.c"
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
                "-Isrc",
                "-Itools",
                "-c",
                str(path),
                "-o",
                str(WORK / "byte64.o"),
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    generate()
