#!/usr/bin/env python3
"""Adaptive CRT point/descendant packing for small-characteristic fields."""

import re
import subprocess

from generate_finite_multipoint import generate as multi_source
from generate_finite_prime_simd import append as prime_source
from generate_finite_shuffle import common, gather_function, prime32, projection
from generate_finite_simd import ROOT, WORK
from generate_finite_simd import generate as simd_source


def generate(build=True):
    args = (32, 512, "_mm512", ["-mavx512f", "-mavx512bw"])
    simd_source(*args, fast=True, values=True, build=False)
    prime_source(*args, build=False)
    multi_source(*args, build=False)
    raw = (WORK / "multi32.c").read_text()
    (WORK / "multi32.c").unlink()
    (WORK / "value32.c").unlink()
    first = raw.index("static void multi32_cd")
    last = raw.index("static Context multi32_context", first)
    helpers = (
        raw[first:last]
        .replace("uint32_t", "uint16_t")
        .replace("UINT32_MAX", "UINT16_MAX")
        .replace("_epi32", "_epi16")
    )
    helpers = re.sub(
        r"_mm512_setr_epi16\(([^()]*)\)",
        lambda m: "_mm512_set_epi16(" + ",".join(reversed(m.group(1).split(","))) + ")",
        helpers,
    )
    code = common(*args[:3]) + prime32() + helpers
    gather = (
        gather_function(*args[:3])
        .replace("size_t point)", "size_t point,size_t points)")
        .replace(
            "0,32,64,96,128,160,192,224,256,288,320,352,384,416,448,480",
            ",".join(f"{i}*(int)points" for i in range(16)),
        )
        .replace("pair+16*32", "pair+16*points")
    )
    project = (
        projection(*args[:3])
        .replace("const Word* result", "const uint16_t* result")
        .replace("size_t rank,", "size_t rank,size_t points,")
        .replace("rank-first<32?rank-first:32", "rank-first<points?rank-first:points")
        .replace("(const uint16_t*)(result+i)", "result+i*points")
        .replace("gather_coordinates(data,r)", "gather_coordinates(data,r,points)")
    )
    code += gather + project
    code += """
static void save_points(uint16_t* out,Word a,size_t points){uint16_t v[32];_mm512_storeu_si512((void*)v,a);memcpy(out,v,points*sizeof(uint16_t));}
static void small_cd(uint16_t* out,const uint16_t* x,const uint16_t* y,size_t n,uint16_t* work,size_t points,const Context* c){
 switch(points){case 8:multi32_cd8(out,x,y,n,work,c);break;case 16:multi32_cd16(out,x,y,n,work,c);break;
 case 32:shuffle32_cd((Word*)out,(const Word*)x,(const Word*)y,n,(Word*)work,c);break;}}
static void small_h(uint16_t* a,size_t n,size_t points,const Context* c){switch(points){case 8:multi32_hadamard8(a,n,c);break;
 case 16:multi32_hadamard16(a,n,c);break;case 32:shuffle32_hadamard((Word*)a,n,c);break;}}
void sm32_tensor(uint32_t* out,const uint32_t* x,const uint32_t* y,const FastCDPlanFF* p,uint32_t* work,int loop,
 size_t rank,const uint32_t input[48][12],const uint32_t output[48][12]){
 Context c=context(p);size_t n=p->dimension,size=n*(loop?2:1),d=p->field.degree,points=8,cost=SIZE_MAX;
 for(size_t k=8;k<=32;k*=2){size_t r=(rank+k-1)/k*k;if(r<=cost){cost=r;points=k;}}
 uint16_t* data=(uint16_t*)work;size_t words=n*points;
 for(size_t first=0;first<rank;first+=points){Word a[12];
  for(size_t j=0;j<d;j++){uint16_t v[32];for(size_t k=0;k<32;k++)v[k]=(uint16_t)(k<points&&first+k<rank?input[first+k][j]:0);a[j]=_mm512_loadu_si512((const void*)v);}
  uint16_t* f=data;uint16_t* u=f+words;uint16_t* fp=u+words;uint16_t* up=fp+words;uint16_t* g=up+words;
  if(loop){
   for(size_t i=0;i<n;i++){Word xl=scalar(x[i*d],a[0],&c),xu=scalar(x[(n+i)*d],a[0],&c),yl=scalar(y[i*d],a[0],&c),yu=scalar(y[(n+i)*d],a[0],&c);
    for(size_t j=1;j<d;j++){xl=add(xl,scalar(x[i*d+j],a[j],&c),&c);xu=add(xu,scalar(x[(n+i)*d+j],a[j],&c),&c);
     yl=add(yl,scalar(y[i*d+j],a[j],&c),&c);yu=add(yu,scalar(y[(n+i)*d+j],a[j],&c),&c);}
    save_points(f+i*points,add(xl,xu,&c),points);save_points(u+i*points,sub(xl,xu,&c),points);
    save_points(fp+i*points,add(yl,yu,&c),points);save_points(up+i*points,sub(yl,yu,&c),points);}
   small_h(f,n,points,&c);small_h(fp,n,points,&c);
   for(size_t i=0;i<words;i+=32){Word v=multiply(_mm512_loadu_si512((const void*)(f+i)),_mm512_loadu_si512((const void*)(fp+i)),&c);_mm512_storeu_si512((void*)(f+i),v);}
   small_h(f,n,points,&c);
   if(p->inverse_dimension!=1)for(size_t i=0;i<words;i+=32){Word v=normalize(_mm512_loadu_si512((const void*)(f+i)),&c);_mm512_storeu_si512((void*)(f+i),v);}
   small_cd(g,u,up,n,g+words,points,&c);
   for(size_t i=0;i<words;i+=32){Word a=_mm512_loadu_si512((const void*)(f+i)),b=_mm512_loadu_si512((const void*)(g+i));
    Word s=half(add(a,b,&c),&c),t=half(sub(a,b,&c),&c);_mm512_storeu_si512((void*)(f+i),s);_mm512_storeu_si512((void*)(u+i),t);}
  }else{
   for(size_t i=0;i<n;i++){Word vx=scalar(x[i*d],a[0],&c),vy=scalar(y[i*d],a[0],&c);for(size_t j=1;j<d;j++){vx=add(vx,scalar(x[i*d+j],a[j],&c),&c);vy=add(vy,scalar(y[i*d+j],a[j],&c),&c);}
    save_points(f+i*points,vx,points);save_points(u+i*points,vy,points);}
   small_cd(fp,f,u,n,fp+words,points,&c);f=fp;
  }
  project(out,f,size,d,first,rank,points,output,&c);
 }
}
"""
    code = (
        code.replace("shuffle32", "sm32")
        .replace("multi32", "sm32_points")
        .replace("value32", "sm32")
    )
    path = WORK / "sm32.c"
    path.write_text(code)
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
                "-c",
                str(path),
                "-o",
                str(WORK / "sm32.o"),
            ],
            check=True,
        )


if __name__ == "__main__":
    generate()
