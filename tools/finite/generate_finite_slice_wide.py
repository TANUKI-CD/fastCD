#!/usr/bin/env python3
"""AVX-512 ternary bit planes; eight coefficient/descendant lanes."""

import subprocess

from generate_finite_simd import ROOT, WORK


def generate(build=True):
    old = (WORK / "slicevec4.c").read_text()
    source = old[: old.index("static inline __attribute__((always_inline)) Word ld(")]
    source = source.replace("256", "512").replace("sv4", "sv8")
    source += """
static inline __attribute__((always_inline)) Word ld(const Coef* a){
 __m512i x=_mm512_loadu_si512((const void*)a),y=_mm512_loadu_si512((const void*)(a+4));
 return (Word){_mm512_permutex2var_epi64(x,_mm512_setr_epi64(0,2,4,6,8,10,12,14),y),
 _mm512_permutex2var_epi64(x,_mm512_setr_epi64(1,3,5,7,9,11,13,15),y)};}
static inline __attribute__((always_inline)) void st(Coef* a,Word w){
 _mm512_storeu_si512((void*)a,_mm512_permutex2var_epi64(w.one,_mm512_setr_epi64(0,8,1,9,2,10,3,11),w.two));
 _mm512_storeu_si512((void*)(a+4),_mm512_permutex2var_epi64(w.one,_mm512_setr_epi64(4,12,5,13,6,14,7,15),w.two));}
static inline Word broadcast(Coef a){return (Word){_mm512_set1_epi64((long long)a.one),_mm512_set1_epi64((long long)a.two)};}
static inline Word mask(Word a,__m512i m){return (Word){_mm512_and_si512(a.one,m),_mm512_and_si512(a.two,m)};}
static void sv8_node8(Coef* restrict r,Coef* restrict im,const Coef* xr,const Coef* xi,const Coef* yr,const Coef* yi,size_t n,Coef* restrict work,const Context* restrict ctx){
 sv8_pair_cross((Word*)r,(Word*)im,(const Word*)xr,(const Word*)xi,(const Word*)yr,(const Word*)yi,n,(Word*)work,ctx);}
"""
    block = old[
        old.index("static void sv4_node2(") : old.index("static void sv4_node1(")
    ]
    tail = block[block.index(" Coef gg[4]") :]
    for B in (4, 2, 1):
        source += f"""
static void sv8_node{B}(Coef* restrict real,Coef* restrict imag,const Coef* xr,const Coef* xi,const Coef* yr,const Coef* yi,
 size_t n,Coef* restrict work,const Context* restrict ctx){{
 size_t h=n/2,size=n*{B};Coef* cr=work+4*size;Coef* ci=cr+size;
 if(n>4){{Coef* ar=work;Coef* ai=ar+size;Coef* br=ai+size;Coef* bi=br+size;
  for(size_t i=0;i<h;i+={8 // B}){{
   Word a=ld(xr+i*{B}),b=ld(xi+(h+i)*{B}),c=ld(xi+i*{B}),d=ld(xr+(h+i)*{B});
   Word e=ld(yr+i*{B}),f=ld(yi+(h+i)*{B}),g=ld(yi+i*{B}),t=ld(yr+(h+i)*{B});
   Word ap=sub(a,b),am=add(a,b),ip=add(c,d),im=sub(c,d),bp=sub(e,f),bm=add(e,f),jp=add(g,t),jm=sub(g,t);
   if(i==0){{__m512i keep=_mm512_setr_epi64({",".join("0" if k < B else "-1" for k in range(8))});
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
                    (k // (2 * B)) * B + k % B + 8 * ((k // B) % 2)
                    for k in range(part * 8, part * 8 + 8)
                ]
                index = "_mm512_setr_epi64(" + ",".join(map(str, ix)) + ")"
                v = f"(Word){{_mm512_permutex2var_epi64({left}.one,{index},{right}.one),_mm512_permutex2var_epi64({left}.two,{index},{right}.two)}}"
                at = f"i*{2 * B}+{8 * part}"
                source += (
                    f"   ((Word*){arr})[({at})/8]={v};\n"
                    if B == 4
                    else f"   st({arr}+{at},{v});\n"
                )
        source += f"  }}\n  sv8_node{B * 2}(cr,ci,ar,ai,br,bi,h,ci+size,ctx);\n }}else memset(cr,0,2*size*sizeof(Coef));\n"
        t = (
            tail.replace("256", "512")
            .replace("sv4", "sv8")
            .replace("[4]", "[8]")
            .replace("k<4", "k<8")
            .replace("k%2", f"k%{B}")
            .replace("k+=2", f"k+={B}")
            .replace("i+=2", f"i+={8 // B}")
            .replace("b<2", f"b<{B}")
        )
        t = (
            t.replace("h*2", f"h*{B}")
            .replace("i*2", f"i*{B}")
            .replace("(h+i)*2", f"(h+i)*{B}")
        )
        t = t.replace(
            "_mm512_setr_epi64x(0,0,-1,-1)",
            "_mm512_setr_epi64("
            + ",".join("0" if i < B else "-1" for i in range(8))
            + ")",
        )
        start = t.index("  Word cr0,cr1,ci0,ci1;")
        end = t.index("  Word fr=", start)
        replacement = "  Word cr0,cr1,ci0,ci1;\n"
        for arr in ("cr", "ci"):
            for sign in (0, 1):
                lo = (
                    f"((const Word*){arr})[i*{2 * B}/8]"
                    if B == 4
                    else f"ld({arr}+i*{2 * B})"
                )
                hi = (
                    f"((const Word*){arr})[i*{2 * B}/8+1]"
                    if B == 4
                    else f"ld({arr}+i*{2 * B}+8)"
                )
                idx = [
                    r * 2 * B + b + sign * B for r in range(8 // B) for b in range(B)
                ]
                ix = "_mm512_setr_epi64(" + ",".join(map(str, idx)) + ")"
                replacement += f"  {{Word lo={lo},hi={hi};{arr}{sign}=(Word){{_mm512_permutex2var_epi64(lo.one,{ix},hi.one),_mm512_permutex2var_epi64(lo.two,{ix},hi.two)}};}}\n"
        t = t[:start] + replacement + t[end:]
        source += t
    root = old[old.index("static void sv4_root(") : old.index("static void sh(")]
    root = (
        root.replace("256", "512")
        .replace("sv4", "sv8")
        .replace("[4]", "[8]")
        .replace("k<4", "k<8")
        .replace("i+=4", "i+=8")
        .replace(
            "_mm512_setr_epi64x(0,-1,-1,-1)",
            "_mm512_setr_epi64(0,-1,-1,-1,-1,-1,-1,-1)",
        )
    )
    source += root + old[old.index("static void sh(") :].replace("sv4", "sv8").replace(
        "slicevec4", "slicevec8"
    )
    plain = source.replace("slicevec8", "sliceplain8")
    (WORK / "sliceplain8.c").write_text(plain)
    if build:
        subprocess.run(
            [
                "cc",
                "-O3",
                "-std=c11",
                "-mavx512f",
                "-mpopcnt",
                "-I" + str(ROOT / "include"),
                "-I" + str(ROOT / "tools/finite/finite"),
                "-I" + str(ROOT / "src/finite"),
                "-c",
                str(WORK / "sliceplain8.c"),
                "-o",
                str(WORK / "sliceplain8.o"),
            ],
            check=True,
        )
    source = '#include "finite_maps.h"\n' + source
    begin = source.index("static void sh(")
    end = source.index("void slicevec8_tensor(", begin)
    source = (
        source[:begin]
        + (ROOT / "tools/finite/finite_ternary_io.h").read_text()
        + source[end:]
    )
    begin = source.index(" Coef a[12][3]")
    end = source.index(" Coef* ex=", begin)
    source = (
        source[:begin]
        + """
 (void)rank;(void)input;(void)output;
 Coef a[12][3]={{{0,0}}},w[12]={{0,0}};size_t d=p->field.degree,n=p->dimension;
 const CdlSmallMap* maps=&fastcd_ff_small_maps[0][d-1];uint32_t reduced[25][12];
 fastcd_ff_small_remainders(reduced,&p->field,maps->total,3);
 Coef basis[25][3]={{{0,0}}};
 for(size_t k=0;k<maps->total;k++){__m512i v=_mm512_loadu_si512((const void*)maps->output[k]);
  basis[k][1]=(Coef){_mm512_cmpeq_epi8_mask(v,_mm512_set1_epi8(1)),_mm512_cmpeq_epi8_mask(v,_mm512_set1_epi8(2))};basis[k][2]=sn(basis[k][1]);}
 for(size_t j=0;j<d;j++){__m512i v=_mm512_loadu_si512((const void*)maps->input[j]);
  a[j][1]=(Coef){_mm512_cmpeq_epi8_mask(v,_mm512_set1_epi8(1)),_mm512_cmpeq_epi8_mask(v,_mm512_set1_epi8(2))};a[j][2]=sn(a[j][1]);
  for(size_t k=0;k<maps->total;k++)w[j]=sa(w[j],basis[k][reduced[k][j]]);
 }
 Word one[12];for(size_t j=0;j<d;j++){
  one[j]=broadcast(w[j]);
 }
 __m512i table_one[12],table_two[12];
 for(size_t j=0;j<d;j++){table_one[j]=_mm512_setr_epi64(0,(long long)a[j][1].one,(long long)a[j][2].one,0,0,0,0,0);
  table_two[j]=_mm512_setr_epi64(0,(long long)a[j][1].two,(long long)a[j][2].two,0,0,0,0,0);}
 __m512i index=_mm512_setr_epi32(0,(int)d,2*(int)d,3*(int)d,4*(int)d,5*(int)d,6*(int)d,7*(int)d,0,0,0,0,0,0,0,0);
"""
        + source[end:]
    )
    begin = source.index("  for(size_t i=0;i<n;i++){Coef ax=")
    end = source.index("  sh(f,n);", begin)
    source = (
        source[:begin]
        + """
  for(size_t i=0;i<n;i+=8){Word ax=encode_values(x+i*d,table_one[0],table_two[0],index),au=encode_values(x+(n+i)*d,table_one[0],table_two[0],index);
   Word by=encode_values(y+i*d,table_one[0],table_two[0],index),bu=encode_values(y+(n+i)*d,table_one[0],table_two[0],index);
   for(size_t j=1;j<d;j++){ax=add(ax,encode_values(x+i*d+j,table_one[j],table_two[j],index));au=add(au,encode_values(x+(n+i)*d+j,table_one[j],table_two[j],index));
    by=add(by,encode_values(y+i*d+j,table_one[j],table_two[j],index));bu=add(bu,encode_values(y+(n+i)*d+j,table_one[j],table_two[j],index));}
   st(f+i,add(ax,au));st(u+i,sub(ax,au));st(fp+i,add(by,bu));st(up+i,sub(by,bu));}
"""
        + source[end:]
    )
    source = source.replace(
        "for(size_t i=0;i<n;i++)f[i]=sm(f[i],fp[i]);",
        "for(size_t i=0;i<n;i+=8)st(f+i,multiply(ld(f+i),ld(fp+i)));",
    )
    source = source.replace(
        "for(size_t i=0;i<n;i++)f[i]=sn(f[i]);",
        "for(size_t i=0;i<n;i+=8)st(f+i,neg(ld(f+i)));",
    )
    source = source.replace(
        "for(size_t i=0;i<n;i++){Coef a=f[i],b=g[i];f[i]=sn(sa(a,b));u[i]=sn(ss(a,b));}",
        "for(size_t i=0;i<n;i+=8){Word left=ld(f+i),right=ld(g+i);st(f+i,neg(add(left,right)));st(u+i,neg(sub(left,right)));}",
    )
    begin = source.index(" }else{for(size_t i=0;i<n;i++){Coef u=")
    end = source.index("  sv8_root(eo,", begin)
    source = (
        source[:begin]
        + """
 }else{for(size_t i=0;i<n;i+=8){Word u=encode_values(x+i*d,table_one[0],table_two[0],index),v=encode_values(y+i*d,table_one[0],table_two[0],index);
  for(size_t j=1;j<d;j++){u=add(u,encode_values(x+i*d+j,table_one[j],table_two[j],index));v=add(v,encode_values(y+i*d+j,table_one[j],table_two[j],index));}st(ex+i,u);st(ey+i,v);}
"""
        + source[end:]
    )
    begin = source.index(" for(size_t i=0;i<n*(loop?2:1);i++)")
    source = (
        source[:begin]
        + """
 for(size_t i=0;i<n*(loop?2:1);i+=8){Word value=ld(ex+i);
  for(size_t j=0;j<d;j++){__m512i ones=_mm512_or_si512(_mm512_and_si512(value.one,one[j].one),_mm512_and_si512(value.two,one[j].two));
   __m512i twos=_mm512_or_si512(_mm512_and_si512(value.one,one[j].two),_mm512_and_si512(value.two,one[j].one));
   __m512i count=_mm512_add_epi64(population_vector(ones),_mm512_slli_epi64(population_vector(twos),1));
   __m256i canonical=_mm512_cvtepi64_epi32(modulo_three(count));
   __m512i stored=_mm512_inserti64x4(_mm512_setzero_si512(),canonical,0);
   _mm512_mask_i32scatter_epi32(out+i*d+j,0xff,index,stored,4);
  }
 }
}
"""
    )
    path = WORK / "slicevec8.c"
    path.write_text(source)
    if build:
        subprocess.run(
            [
                "cc",
                "-O3",
                "-std=c11",
                "-mavx512f",
                "-mavx512bw",
                "-I" + str(ROOT / "include"),
                "-I" + str(ROOT / "tools/finite/finite"),
                "-I" + str(ROOT / "src/finite"),
                "-c",
                str(path),
                "-o",
                str(WORK / "slicevec8.o"),
            ],
            check=True,
        )


if __name__ == "__main__":
    generate()
