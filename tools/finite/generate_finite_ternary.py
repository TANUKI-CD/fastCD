#!/usr/bin/env python3
"""Ternary bit planes with the smallest usable coefficient mask width."""

import subprocess

from generate_finite_simd import ROOT, WORK


def generate(bits, build=True):
    lanes = 512 // bits
    name = f"tri{bits}"
    ctype = f"uint{bits}_t"
    old = (WORK / "slicevec4.c").read_text()
    source = old[: old.index("static inline __attribute__((always_inline)) Word ld(")]
    source = (
        source.replace("256", "512").replace("sv4", name).replace("uint64_t", ctype)
    )
    source = '#include "finite_maps.h"\n' + source

    def index(values):
        return (
            "_mm512_set_epi16(" + ",".join(map(str, reversed(values))) + ")"
            if bits == 16
            else f"_mm512_setr_epi{bits}(" + ",".join(map(str, values)) + ")"
        )

    def perm(a, ix, b):
        return f"_mm512_permutex2var_epi{bits}({a},{index(ix)},{b})"

    source += f"""
static inline __attribute__((always_inline)) Word ld(const Coef* a){{
 __m512i x=_mm512_loadu_si512((const void*)a),y=_mm512_loadu_si512((const void*)(a+{lanes // 2}));
 return (Word){{{perm("x", list(range(0, 2 * lanes, 2)), "y")},{perm("x", list(range(1, 2 * lanes, 2)), "y")}}};}}
static inline __attribute__((always_inline)) void st(Coef* a,Word w){{
 _mm512_storeu_si512((void*)a,{perm("w.one", [i // 2 + lanes * (i % 2) for i in range(lanes)], "w.two")});
 _mm512_storeu_si512((void*)(a+{lanes // 2}),{perm("w.one", [i // 2 + lanes * (i % 2) for i in range(lanes, 2 * lanes)], "w.two")});}}
static inline Word broadcast(Coef a){{return (Word){{_mm512_set1_epi{bits}(({"short" if bits == 16 else "int" if bits == 32 else "long long"})a.one),_mm512_set1_epi{bits}(({"short" if bits == 16 else "int" if bits == 32 else "long long"})a.two)}};}}
static inline Word mask(Word a,__m512i m){{return (Word){{_mm512_and_si512(a.one,m),_mm512_and_si512(a.two,m)}};}}
static void {name}_node{lanes}(Coef* restrict r,Coef* restrict im,const Coef* xr,const Coef* xi,const Coef* yr,const Coef* yi,size_t n,Coef* restrict work,const Context* restrict ctx){{
 {name}_pair_cross((Word*)r,(Word*)im,(const Word*)xr,(const Word*)xi,(const Word*)yr,(const Word*)yi,n,(Word*)work,ctx);}}
"""
    original = old[
        old.index("static void sv4_node2(") : old.index("static void sv4_node1(")
    ]
    tail = original[original.index(" Coef gg[4]") :]
    for B in reversed([1 << i for i in range(lanes.bit_length() - 1)]):
        C = lanes // B
        source += f"""
static void {name}_node{B}(Coef* restrict real,Coef* restrict imag,const Coef* xr,const Coef* xi,const Coef* yr,const Coef* yi,
 size_t n,Coef* restrict work,const Context* restrict ctx){{
 size_t h=n/2,size=n*{B};Coef* cr=work+4*size;Coef* ci=cr+size;
 if(n>4){{Coef* ar=work;Coef* ai=ar+size;Coef* br=ai+size;Coef* bi=br+size;
  for(size_t i=0;i<h;i+={C}){{
   Word a=ld(xr+i*{B}),b=ld(xi+(h+i)*{B}),c=ld(xi+i*{B}),d=ld(xr+(h+i)*{B});
   Word e=ld(yr+i*{B}),f=ld(yi+(h+i)*{B}),g=ld(yi+i*{B}),t=ld(yr+(h+i)*{B});
   Word ap=sub(a,b),am=add(a,b),ip=add(c,d),im=sub(c,d),bp=sub(e,f),bm=add(e,f),jp=add(g,t),jm=sub(g,t);
   if(i==0){{__m512i keep={index([0 if k < B else -1 for k in range(lanes)])};
    ap=mask(ap,keep);am=mask(am,keep);ip=mask(ip,keep);im=mask(im,keep);bp=mask(bp,keep);bm=mask(bm,keep);jp=mask(jp,keep);jm=mask(jm,keep);}}
"""
        for arr, left, right in [
            ("ar", "ap", "am"),
            ("ai", "ip", "im"),
            ("br", "bp", "bm"),
            ("bi", "jp", "jm"),
        ]:
            for part in (0, 1):
                ix = [
                    (k // (2 * B)) * B + k % B + lanes * ((k // B) % 2)
                    for k in range(part * lanes, part * lanes + lanes)
                ]
                v = (
                    "(Word){"
                    + perm(left + ".one", ix, right + ".one")
                    + ","
                    + perm(left + ".two", ix, right + ".two")
                    + "}"
                )
                at = f"i*{2 * B}+{lanes * part}"
                source += (
                    f"   ((Word*){arr})[({at})/{lanes}]={v};\n"
                    if B == lanes // 2
                    else f"   st({arr}+{at},{v});\n"
                )
        source += f"  }}\n  {name}_node{B * 2}(cr,ci,ar,ai,br,bi,h,ci+size,ctx);\n }}else memset(cr,0,2*size*sizeof(Coef));\n"
        t = (
            tail.replace("256", "512")
            .replace("sv4", name)
            .replace("[4]", f"[{lanes}]")
            .replace("k<4", f"k<{lanes}")
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
        t = t.replace(
            "_mm512_setr_epi64x(0,0,-1,-1)",
            index([0 if k < B else -1 for k in range(lanes)]),
        )
        start = t.index("  Word cr0,cr1,ci0,ci1;")
        end = t.index("  Word fr=", start)
        replacement = "  Word cr0,cr1,ci0,ci1;\n"
        for arr in ("cr", "ci"):
            for sign in (0, 1):
                lo = (
                    f"((const Word*){arr})[i*{2 * B}/{lanes}]"
                    if B == lanes // 2
                    else f"ld({arr}+i*{2 * B})"
                )
                hi = (
                    f"((const Word*){arr})[i*{2 * B}/{lanes}+1]"
                    if B == lanes // 2
                    else f"ld({arr}+i*{2 * B}+{lanes})"
                )
                ix = [r * 2 * B + b + sign * B for r in range(C) for b in range(B)]
                replacement += f"  {{Word lo={lo},hi={hi};{arr}{sign}=(Word){{{perm('lo.one', ix, 'hi.one')},{perm('lo.two', ix, 'hi.two')}}};}}\n"
        source += t[:start] + replacement + t[end:]
    root = old[old.index("static void sv4_root(") : old.index("static void sh(")]
    root = (
        root.replace("256", "512")
        .replace("sv4", name)
        .replace("[4]", f"[{lanes}]")
        .replace("k<4", f"k<{lanes}")
        .replace("i+=4", f"i+={lanes}")
    )
    root = root.replace(
        "_mm512_setr_epi64x(0,-1,-1,-1)", index([0] + [-1] * (lanes - 1))
    )
    root = root.replace(
        "Word a=ld(x+i),b=ld(y+i);dot=add(dot,multiply(a,b));",
        "Word left=ld(x+i),right=ld(y+i);dot=add(dot,multiply(left,right));",
    ).replace(
        "multiply(alpha,b),multiply(beta,a)",
        "multiply(alpha,right),multiply(beta,left)",
    )
    source += root
    if bits == 64:
        load = "__m512i x=_mm512_mask_i32gather_epi32(_mm512_setzero_si512(),0xff,index,values,4);x=_mm512_cvtepu32_epi64(_mm512_castsi512_si256(x));"
    elif bits == 32:
        load = "__m512i x=_mm512_i32gather_epi32(index,values,4);"
    else:
        load = "__m512i lo=_mm512_i32gather_epi32(index,values,4),hi=_mm512_i32gather_epi32(index,values+16*d,4);__m512i x=_mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(lo)),_mm512_cvtepi32_epi16(hi),1);"
    source += f"""static inline __attribute__((always_inline)) Word encode_values(const uint32_t* values,__m512i a,__m512i b,__m512i index,size_t d){{(void)d;{load}
 return (Word){{_mm512_permutexvar_epi{bits}(x,a),_mm512_permutexvar_epi{bits}(x,b)}};}}
static inline __attribute__((always_inline)) __m512i population_vector(__m512i a){{
 static const uint8_t lut[16]={{0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4}};
 __m512i table=_mm512_broadcast_i32x4(_mm_loadu_si128((const void*)lut)),m=_mm512_set1_epi8(15);
 __m512i v=_mm512_add_epi8(_mm512_shuffle_epi8(table,_mm512_and_si512(a,m)),_mm512_shuffle_epi8(table,_mm512_and_si512(_mm512_srli_epi64(a,4),m)));
"""
    source += (
        " return _mm512_sad_epu8(v,_mm512_setzero_si512());"
        if bits == 64
        else " return _mm512_madd_epi16(_mm512_maddubs_epi16(v,_mm512_set1_epi8(1)),_mm512_set1_epi16(1));"
        if bits == 32
        else " return _mm512_maddubs_epi16(v,_mm512_set1_epi8(1));"
    ) + "}\n"
    source += """static inline __attribute__((always_inline)) __m512i modulo_three(__m512i a){
 static const uint8_t lut[16]={0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0};__m512i t=_mm512_broadcast_i32x4(_mm_loadu_si128((const void*)lut));
 return _mm512_shuffle_epi8(t,_mm512_add_epi64(_mm512_shuffle_epi8(t,a),_mm512_shuffle_epi8(t,_mm512_srli_epi64(a,4))));}
"""
    source += f"static void sh(Coef* data,size_t n){{for(size_t base=0;base<n;base+={lanes}){{Word a=ld(data+base);\n"
    for width in [1 << i for i in range(lanes.bit_length() - 1)]:
        ix = index([i ^ width for i in range(lanes)])
        mask = sum(1 << i for i in range(lanes) if i & width)
        source += f" {{Word b={{_mm512_permutexvar_epi{bits}({ix},a.one),_mm512_permutexvar_epi{bits}({ix},a.two)}};Word s=add(a,b),t=sub(b,a);a=(Word){{_mm512_mask_blend_epi{bits}(UINT64_C({mask}),s.one,t.one),_mm512_mask_blend_epi{bits}(UINT64_C({mask}),s.two,t.two)}};}}\n"
    source += f""" st(data+base,a);}}for(size_t width={lanes};width<n;width*=2)for(size_t base=0;base<n;base+=2*width)for(size_t j=0;j<width;j+={lanes}){{Word a=ld(data+base+j),b=ld(data+base+width+j);st(data+base+j,add(a,b));st(data+base+width+j,sub(a,b));}}}}
void {name}_tensor(uint32_t* restrict out,const uint32_t* x,const uint32_t* y,const FastCDPlanFF* p,uint32_t* restrict work,int loop,size_t rank,const uint32_t input[48][12],const uint32_t output[48][12]){{
 (void)rank;(void)input;(void)output;size_t d=p->field.degree,n=p->dimension;const CdlSmallMap* maps=&fastcd_ff_small_maps[0][d-1];
 Coef a[12][3]={{{{{{0,0}}}}}},w[12]={{{{0,0}}}},basis[25][3]={{{{{{0,0}}}}}};uint32_t reduced[25][12];fastcd_ff_small_remainders(reduced,&p->field,maps->total,3);
 for(size_t k=0;k<maps->total;k++){{__m512i v=_mm512_loadu_si512((const void*)maps->output[k]);basis[k][1]=(Coef){{({ctype})_mm512_cmpeq_epi8_mask(v,_mm512_set1_epi8(1)),({ctype})_mm512_cmpeq_epi8_mask(v,_mm512_set1_epi8(2))}};basis[k][2]=sn(basis[k][1]);}}
 Word weights[12];__m512i table_one[12],table_two[12];
 for(size_t j=0;j<d;j++){{__m512i v=_mm512_loadu_si512((const void*)maps->input[j]);a[j][1]=(Coef){{({ctype})_mm512_cmpeq_epi8_mask(v,_mm512_set1_epi8(1)),({ctype})_mm512_cmpeq_epi8_mask(v,_mm512_set1_epi8(2))}};a[j][2]=sn(a[j][1]);
  for(size_t k=0;k<maps->total;k++){{w[j]=sa(w[j],basis[k][reduced[k][j]]);}}weights[j]=broadcast(w[j]);
  table_one[j]={index(["0", "a[j][1].one", "a[j][2].one"] + ["0"] * (lanes - 3))};table_two[j]={index(["0", "a[j][1].two", "a[j][2].two"] + ["0"] * (lanes - 3))};
 }}
 __m512i index=_mm512_setr_epi32({",".join(str(i) + "*(int)d" if i < min(lanes, 16) else "0" for i in range(16))});
 Coef* ex=(Coef*)work;Coef* ey=ex+n;Coef* eo=ey+n;Coef* tail=eo+n;Context c={{(int)p->inverse_dimension}};
 if(loop){{Coef* f=ex;Coef* u=ey;Coef* fp=eo;Coef* up=tail;Coef* g=up+n;
  for(size_t i=0;i<n;i+={lanes}){{Word ax=encode_values(x+i*d,table_one[0],table_two[0],index,d),au=encode_values(x+(n+i)*d,table_one[0],table_two[0],index,d);
   Word by=encode_values(y+i*d,table_one[0],table_two[0],index,d),bu=encode_values(y+(n+i)*d,table_one[0],table_two[0],index,d);
   for(size_t j=1;j<d;j++){{ax=add(ax,encode_values(x+i*d+j,table_one[j],table_two[j],index,d));au=add(au,encode_values(x+(n+i)*d+j,table_one[j],table_two[j],index,d));by=add(by,encode_values(y+i*d+j,table_one[j],table_two[j],index,d));bu=add(bu,encode_values(y+(n+i)*d+j,table_one[j],table_two[j],index,d));}}
   st(f+i,add(ax,au));st(u+i,sub(ax,au));st(fp+i,add(by,bu));st(up+i,sub(by,bu));}}
  sh(f,n);sh(fp,n);for(size_t i=0;i<n;i+={lanes})st(f+i,multiply(ld(f+i),ld(fp+i)));sh(f,n);
  if(c.inverse==2)for(size_t i=0;i<n;i+={lanes})st(f+i,neg(ld(f+i)));
  {name}_root(g,u,up,n,g+n,&c);for(size_t i=0;i<n;i+={lanes}){{Word a0=ld(f+i),b0=ld(g+i);st(f+i,neg(add(a0,b0)));st(u+i,neg(sub(a0,b0)));}}
 }}else{{for(size_t i=0;i<n;i+={lanes}){{Word a0=encode_values(x+i*d,table_one[0],table_two[0],index,d),b0=encode_values(y+i*d,table_one[0],table_two[0],index,d);for(size_t j=1;j<d;j++){{a0=add(a0,encode_values(x+i*d+j,table_one[j],table_two[j],index,d));b0=add(b0,encode_values(y+i*d+j,table_one[j],table_two[j],index,d));}}st(ex+i,a0);st(ey+i,b0);}}{name}_root(eo,ex,ey,n,tail,&c);ex=eo;}}
 for(size_t i=0;i<n*(loop?2:1);i+={lanes}){{Word value=ld(ex+i);
  for(size_t j=0;j<d;j++){{__m512i ones=_mm512_or_si512(_mm512_and_si512(value.one,weights[j].one),_mm512_and_si512(value.two,weights[j].two));
   __m512i twos=_mm512_or_si512(_mm512_and_si512(value.one,weights[j].two),_mm512_and_si512(value.two,weights[j].one));
   __m512i canonical=modulo_three(_mm512_add_epi{bits}(population_vector(ones),_mm512_slli_epi{bits}(population_vector(twos),1)));
"""
    if bits == 16:
        source += """   _mm512_i32scatter_epi32(out+i*d+j,index,_mm512_cvtepu16_epi32(_mm512_castsi512_si256(canonical)),4);_mm512_i32scatter_epi32(out+(i+16)*d+j,index,_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(canonical,1)),4);
"""
    elif bits == 32:
        source += "   _mm512_i32scatter_epi32(out+i*d+j,index,canonical,4);\n"
    else:
        source += "   _mm512_mask_i32scatter_epi32(out+i*d+j,0xff,index,_mm512_inserti64x4(_mm512_setzero_si512(),_mm512_cvtepi64_epi32(canonical),0),4);\n"
    source += "  }\n }\n}\n"
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
                "-Isrc",
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
    for b in (16, 32, 64):
        generate(b)
