#!/usr/bin/env python3
"""Ternary bit planes, with four independent descendants per AVX2 word."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".private/finite-optimization/representations"


def generate(build=True):
    s = r"""
#include <immintrin.h>
#include <string.h>
#include "fastcd/finite.h"
typedef struct __attribute__((packed,may_alias)) {uint64_t one,two;} Coef;
typedef struct __attribute__((packed,may_alias)) {__m256i one,two;} Word;
typedef struct {int inverse;} Context;
static inline __attribute__((always_inline)) Coef sa(Coef a,Coef b) {
 uint64_t x=a.one^b.one,y=a.two^b.two;
 return (Coef){(x&~y)|(a.two&b.two),(y&~x)|(a.one&b.one)};
}
static inline Coef sn(Coef a) {return (Coef){a.two,a.one};}
static inline Coef ss(Coef a,Coef b) {return sa(a,sn(b));}
static inline Coef sm(Coef a,Coef b) {return (Coef){(a.one&b.one)|(a.two&b.two),(a.one&b.two)|(a.two&b.one)};}
static inline __attribute__((always_inline)) Word add(Word a,Word b) {
 __m256i x=_mm256_xor_si256(a.one,b.one),y=_mm256_xor_si256(a.two,b.two);
 return (Word){_mm256_or_si256(_mm256_andnot_si256(y,x),_mm256_and_si256(a.two,b.two)),
 _mm256_or_si256(_mm256_andnot_si256(x,y),_mm256_and_si256(a.one,b.one))};
}
static inline __attribute__((always_inline)) Word neg(Word a) {return (Word){a.two,a.one};}
static inline __attribute__((always_inline)) Word sub(Word a,Word b) {return add(a,neg(b));}
static inline __attribute__((always_inline)) Word multiply(Word a,Word b) {
 return (Word){_mm256_or_si256(_mm256_and_si256(a.one,b.one),_mm256_and_si256(a.two,b.two)),
 _mm256_or_si256(_mm256_and_si256(a.one,b.two),_mm256_and_si256(a.two,b.one))};
}
#define FASTCD_FF_NAME(n) sv4_##n
#define VADD(a,b) add(a,b)
#define VSUB(a,b) sub(a,b)
#define VMUL(a,b) multiply(a,b)
#define VDIFF(a,b,c,d) sub(multiply(a,b),multiply(c,d))
#define VHALF(a) neg(a)
#define VNORMALIZE(a) (ctx->inverse==1?(a):neg(a))
#define VZERO ((Word){_mm256_setzero_si256(),_mm256_setzero_si256()})
#include "finite_value_schedule.h"
static inline __attribute__((always_inline)) Word ld(const Coef* a) {
 __m256i x=_mm256_loadu_si256((const void*)a),y=_mm256_loadu_si256((const void*)(a+2));
 return (Word){_mm256_permute4x64_epi64(_mm256_unpacklo_epi64(x,y),0xd8),
 _mm256_permute4x64_epi64(_mm256_unpackhi_epi64(x,y),0xd8)};
}
static inline __attribute__((always_inline)) void st(Coef* a,Word w) {
 __m256i x=_mm256_unpacklo_epi64(w.one,w.two),y=_mm256_unpackhi_epi64(w.one,w.two);
 _mm256_storeu_si256((void*)a,_mm256_permute2x128_si256(x,y,0x20));
 _mm256_storeu_si256((void*)(a+2),_mm256_permute2x128_si256(x,y,0x31));
}
static inline Word broadcast(Coef a) {return (Word){_mm256_set1_epi64x((long long)a.one),_mm256_set1_epi64x((long long)a.two)};}
static inline Word mask(Word a,__m256i m) {return (Word){_mm256_and_si256(a.one,m),_mm256_and_si256(a.two,m)};}
static void sv4_node4(Coef* restrict r,Coef* restrict im,const Coef* xr,const Coef* xi,
 const Coef* yr,const Coef* yi,size_t n,Coef* restrict work,const Context* restrict ctx) {
 sv4_pair_cross((Word*)r,(Word*)im,(const Word*)xr,(const Word*)xi,(const Word*)yr,(const Word*)yi,n,(Word*)work,ctx);
}
"""
    for B in (2, 1):
        C = 4 // B
        final = B == 2

        def zipword(a, b, which, B=B):
            if B == 2:
                op = "0x20" if which == 0 else "0x31"
                return f"(Word){{_mm256_permute2x128_si256({a}.one,{b}.one,{op}),_mm256_permute2x128_si256({a}.two,{b}.two,{op})}}"
            op = "0x20" if which == 0 else "0x31"
            return (
                "(Word){"
                + ",".join(
                    f"_mm256_permute2x128_si256(_mm256_unpacklo_epi64({a}.{q},{b}.{q}),_mm256_unpackhi_epi64({a}.{q},{b}.{q}),{op})"
                    for q in ("one", "two")
                )
                + "}"
            )

        def childload(arr, sign, B=B, final=final):
            lo = (
                f"((const Word*)({arr}))[i*{2 * B}//4]"
                if final
                else f"ld({arr}+i*{2 * B})"
            )
            hi = (
                f"((const Word*)({arr}))[i*{2 * B}//4+1]"
                if final
                else f"ld({arr}+i*{2 * B}+4)"
            )
            # Generate C division, not Python's //.
            lo = lo.replace("//", "/")
            hi = hi.replace("//", "/")
            if B == 2:
                op = "0x20" if sign == 0 else "0x31"

                def expr(q):
                    return f"_mm256_permute2x128_si256(lo.{q},hi.{q},{op})"
            else:
                op = "lo" if sign == 0 else "hi"

                def expr(q):
                    return f"_mm256_permute4x64_epi64(_mm256_unpack{op}_epi64(lo.{q},hi.{q}),0xd8)"

            return (
                f"{{Word lo={lo},hi={hi};"
                + f"{arr}{sign}=(Word){{{expr('one')},{expr('two')}}};}}"
            )

        s += f"""
static void sv4_node{B}(Coef* restrict real,Coef* restrict imag,const Coef* xr,const Coef* xi,
 const Coef* yr,const Coef* yi,size_t n,Coef* restrict work,const Context* restrict ctx) {{
 size_t h=n/2,size=n*{B};Coef* cr=work+4*size;Coef* ci=cr+size;
 if(n>4) {{Coef* ar=work;Coef* ai=ar+size;Coef* br=ai+size;Coef* bi=br+size;
  for(size_t i=0;i<h;i+={C}) {{
   Word a=ld(xr+i*{B}),b=ld(xi+(h+i)*{B}),c=ld(xi+i*{B}),d=ld(xr+(h+i)*{B});
   Word e=ld(yr+i*{B}),f=ld(yi+(h+i)*{B}),g=ld(yi+i*{B}),t=ld(yr+(h+i)*{B});
   Word ap=sub(a,b),am=add(a,b),ip=add(c,d),im=sub(c,d);
   Word bp=sub(e,f),bm=add(e,f),jp=add(g,t),jm=sub(g,t);
   if(i==0){{__m256i keep=_mm256_setr_epi64x({",".join("0" if k < B else "-1" for k in range(4))});
    ap=mask(ap,keep);am=mask(am,keep);ip=mask(ip,keep);im=mask(im,keep);
    bp=mask(bp,keep);bm=mask(bm,keep);jp=mask(jp,keep);jm=mask(jm,keep);}}
"""
        for arr, a, b in (
            ("ar", "ap", "am"),
            ("ai", "ip", "im"),
            ("br", "bp", "bm"),
            ("bi", "jp", "jm"),
        ):
            for k in (0, 1):
                val = zipword(a, b, k)
                offset = f"i*{2 * B}+{4 * k}"
                s += (
                    f"   ((Word*)({arr}))[({offset})/4]={val};\n"
                    if final
                    else f"   st({arr}+{offset},{val});\n"
                )
        s += f"""  }}
  sv4_node{2 * B}(cr,ci,ar,ai,br,bi,h,ci+size,ctx);
 }}else memset(cr,0,2*size*sizeof(Coef));
 Coef gg[4],qq[4],dd[4],ee[4];
 for(size_t k=0;k<4;k++){{size_t b=k%{B};gg[k]=xr[h*{B}+b];qq[k]=xi[h*{B}+b];dd[k]=yr[h*{B}+b];ee[k]=yi[h*{B}+b];}}
 Word gp=ld(gg),gq=ld(qq),dp=ld(dd),dq=ld(ee),gw=add(gp,gq),dw=add(dp,dq);
 Word ap=VZERO,aq=VZERO,aw=VZERO;
 for(size_t i=0;i<h;i+={C}) {{
  Word ur=ld(xr+i*{B}),ui=ld(xi+i*{B}),vr=ld(yr+i*{B}),vi=ld(yi+i*{B});
  Word sr=ld(xr+(h+i)*{B}),si=ld(xi+(h+i)*{B}),tr=ld(yr+(h+i)*{B}),ti=ld(yi+(h+i)*{B});
  if(i==0){{__m256i keep=_mm256_setr_epi64x({",".join("0" if k < B else "-1" for k in range(4))});
   ur=mask(ur,keep);ui=mask(ui,keep);vr=mask(vr,keep);vi=mask(vi,keep);
   sr=mask(sr,keep);si=mask(si,keep);tr=mask(tr,keep);ti=mask(ti,keep);}}
  sv4_ValueLocal a=sv4_vlocal(ur,vr,sr,tr,gp,dp,ctx),b=sv4_vlocal(ui,vi,si,ti,gq,dq,ctx);
  sv4_ValueLocal c=sv4_vlocal(add(ur,ui),add(vr,vi),add(sr,si),add(tr,ti),gw,dw,ctx);
  ap=add(ap,a.scalar);aq=add(aq,b.scalar);aw=add(aw,c.scalar);
  Word cr0,cr1,ci0,ci1;
  {childload("cr", 0)}{childload("cr", 1)}{childload("ci", 0)}{childload("ci", 1)}
  Word fr=add(sub(a.first,b.first),neg(add(cr0,cr1)));
  Word fi=add(sub(sub(c.first,a.first),b.first),neg(add(ci0,ci1)));
  Word gr=add(sub(a.second,b.second),neg(sub(ci1,ci0)));
  Word gi=add(sub(sub(c.second,a.second),b.second),neg(sub(cr0,cr1)));
  st(real+i*{B},fr);st(imag+i*{B},fi);st(real+(h+i)*{B},gr);st(imag+(h+i)*{B},gi);
 }}
 Coef aa[4],bb[4],cc[4];st(aa,ap);st(bb,aq);st(cc,aw);
 for(size_t b=0;b<{B};b++){{Coef a={{0,0}},q={{0,0}},w={{0,0}};
  for(size_t k=b;k<4;k+={B}){{a=sa(a,aa[k]);q=sa(q,bb[k]);w=sa(w,cc[k]);}}
  real[b]=imag[b]=(Coef){{0,0}};real[h*{B}+b]=ss(a,q);imag[h*{B}+b]=ss(ss(w,a),q);
 }}
}}
"""
    s += r"""
static void sv4_root(Coef* restrict out,const Coef* x,const Coef* y,size_t n,Coef* restrict work,const Context* restrict ctx) {
 size_t h=n/2;sv4_node1(out,out+h,x,x+h,y,y+h,h,work,ctx);
 Word gamma=broadcast(x[h]),delta=broadcast(y[h]),alpha=broadcast(x[0]),beta=broadcast(y[0]);
 Word sum=VZERO,dot=VZERO;
 for(size_t i=0;i<h;i+=4){Word u=ld(x+i),v=ld(y+i),s=ld(x+h+i),t=ld(y+h+i);
  if(i==0){__m256i keep=_mm256_setr_epi64x(0,-1,-1,-1);u=mask(u,keep);v=mask(v,keep);s=mask(s,keep);t=mask(t,keep);}
  sv4_ValueLocal a=sv4_vlocal(u,v,s,t,gamma,delta,ctx);sum=add(sum,a.scalar);
  st(out+i,add(ld(out+i),a.first));st(out+h+i,sub(a.second,ld(out+h+i)));
 }
 Coef partial[4],a={0,0};st(partial,sum);for(int k=0;k<4;k++)a=sa(a,partial[k]);out[h]=a;
 for(size_t i=0;i<n;i+=4){Word a=ld(x+i),b=ld(y+i);dot=add(dot,multiply(a,b));
  st(out+i,add(ld(out+i),add(multiply(alpha,b),multiply(beta,a))));}
 st(partial,dot);a=(Coef){0,0};for(int k=0;k<4;k++)a=sa(a,partial[k]);out[0]=ss(sn(sm(x[0],y[0])),a);
}
static void sh(Coef* a,size_t n) {
 for(size_t width=1;width<n;width*=2)for(size_t base=0;base<n;base+=2*width)for(size_t j=0;j<width;j++){
  Coef x=a[base+j],y=a[base+width+j];a[base+j]=sa(x,y);a[base+width+j]=ss(x,y);}
}
void slicevec4_tensor(uint32_t* out,const uint32_t* x,const uint32_t* y,const FastCDPlanFF* p,uint32_t* work,
 int loop,size_t rank,const uint32_t input[48][12],const uint32_t output[48][12]) {
 Coef a[12][3]={{{0,0}}},w[12]={{0,0}};size_t d=p->field.degree,n=p->dimension;
 for(size_t j=0;j<d;j++){for(size_t r=0;r<rank;r++){uint64_t bit=UINT64_C(1)<<r;
  if(input[r][j]==1)a[j][1].one|=bit;if(input[r][j]==2)a[j][1].two|=bit;
  if(output[r][j]==1)w[j].one|=bit;if(output[r][j]==2)w[j].two|=bit;}a[j][2]=sn(a[j][1]);}
 Coef* ex=(Coef*)work;Coef* ey=ex+n;Coef* eo=ey+n;Coef* tail=eo+n;
 Context c={(int)p->inverse_dimension};
 if(loop){Coef* f=ex;Coef* u=ey;Coef* fp=eo;Coef* up=tail;Coef* g=up+n;Coef* scratch=g+n;
  for(size_t i=0;i<n;i++){Coef ax={0,0},au={0,0},by={0,0},bu={0,0};
   for(size_t j=0;j<d;j++){ax=sa(ax,a[j][x[i*d+j]]);au=sa(au,a[j][x[(n+i)*d+j]]);
    by=sa(by,a[j][y[i*d+j]]);bu=sa(bu,a[j][y[(n+i)*d+j]]);}
   f[i]=sa(ax,au);u[i]=ss(ax,au);fp[i]=sa(by,bu);up[i]=ss(by,bu);}
  sh(f,n);sh(fp,n);for(size_t i=0;i<n;i++)f[i]=sm(f[i],fp[i]);sh(f,n);
  if(c.inverse==2)for(size_t i=0;i<n;i++)f[i]=sn(f[i]);
  sv4_root(g,u,up,n,scratch,&c);
  for(size_t i=0;i<n;i++){Coef a=f[i],b=g[i];f[i]=sn(sa(a,b));u[i]=sn(ss(a,b));}
 }else{for(size_t i=0;i<n;i++){Coef u={0,0},v={0,0};for(size_t j=0;j<d;j++){u=sa(u,a[j][x[i*d+j]]);v=sa(v,a[j][y[i*d+j]]);}ex[i]=u;ey[i]=v;}
  sv4_root(eo,ex,ey,n,tail,&c);ex=eo;}
 for(size_t i=0;i<n*(loop?2:1);i++)for(size_t j=0;j<d;j++){
  uint64_t ones=(ex[i].one&w[j].one)|(ex[i].two&w[j].two),twos=(ex[i].one&w[j].two)|(ex[i].two&w[j].one);
  out[i*d+j]=(uint32_t)(__builtin_popcountll(ones)+2*__builtin_popcountll(twos))%3;}
}
"""
    path = WORK / "slicevec4.c"
    path.write_text(s)
    if not build:
        return
    subprocess.run(
        [
            "cc",
            "-O3",
            "-std=c11",
            "-mavx2",
            "-mpopcnt",
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "tools/finite/finite"),
            "-c",
            str(path),
            "-o",
            str(WORK / "slicevec4.o"),
        ],
        check=True,
    )


if __name__ == "__main__":
    generate()
