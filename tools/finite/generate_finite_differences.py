#!/usr/bin/env python3
"""Structured optimal-rank input evaluation by finite differences.

For p>=2d-1 the CRT points are 0,...,2d-2. Monomial coefficients first map
to Delta^k f(0)=k!*sum_j S(j,k)*a_j. Thereafter each next point needs only
additions. SIMD lanes hold coordinates while evaluating, then point lanes
for the algebra kernel. Every map and conversion belongs to the call.
"""

import subprocess

from finite_coordinate_projection import code as project_coordinates
from finite_transpose import calls as transpose_calls
from finite_transpose import code as transpose_code
from generate_finite_multipoint import point_width
from generate_finite_simd import ROOT, WORK


def generate(
    lanes, bits, prefix, flags, multipoint=False, build=True, vector_packing=False
):
    p = prefix
    name = f"{'tile' if vector_packing else 'hybrid' if multipoint else 'diff'}{lanes}"
    old = f"value{lanes}"
    text = (WORK / f"{old}.c").read_text()
    source = text[: text.index(f"void {old}_run(")]
    first = text.index("static uint32_t horizontal(")
    last = text.index(f"void {old}_tensor(", first)
    source += text[first:last]
    # Only even 32-bit lanes of these intermediates are consumed by mul_epu32.
    for word in ("even", "odd"):
        source = source.replace(
            f"{p}_mullo_epi32({word},c->negative_inverse)",
            f"{p}_mul_epu32({word},c->negative_inverse)",
        )
    z = f"{p}_setzero_si{bits}()"

    def load(a):
        return f"{p}_loadu_si{bits}((const void*)({a}))"

    if multipoint:
        source += f"""void multi{lanes}_cd_encoded(uint32_t*,const uint32_t*,const uint32_t*,const FastCDPlanFF*,uint32_t*,size_t);
void multi{lanes}_finish_encoded(uint32_t*,const FastCDPlanFF*,uint32_t*,size_t);
"""
    for d in range(2, 13):
        points = point_width(2 * d - 1, lanes) if multipoint else lanes
        if multipoint:
            source += f"""static inline void {name}_store{d}(uint32_t* out,Word value) {{
    uint32_t words[{lanes}];{p}_storeu_si{bits}((void*)words,value);memcpy(out,words,{points}*sizeof(uint32_t));
}}
"""
            state_declaration = f"uint32_t* state=work;uint32_t* tail=state+(loop?4*n*groups:2*n*groups)*{points};"
            producer = f"""if(loop) {{
                Word au={load("ap")},bu={load("bp")};uint32_t* s=state+group*4*n*{points};
                {name}_store{d}(s+(i+coord)*{points},add(ax,au,&c));{name}_store{d}(s+(n+i+coord)*{points},sub(ax,au,&c));
                {name}_store{d}(s+(2*n+i+coord)*{points},add(by,bu,&c));{name}_store{d}(s+(3*n+i+coord)*{points},sub(by,bu,&c));
            }} else {{uint32_t* s=state+group*2*n*{points};{name}_store{d}(s+(i+coord)*{points},ax);{name}_store{d}(s+(n+i+coord)*{points},by);}}"""
            core = f"""uint32_t* s=state+group*(loop?4*n:2*n)*{points};
        if(loop)multi{lanes}_finish_encoded(s,p,tail,{points});
        else{{multi{lanes}_cd_encoded(tail,s,s+n*{points},p,tail+n*{points},{points});memcpy(s,tail,n*{points}*sizeof(uint32_t));}}"""
        else:
            state_declaration = (
                "Word* state=(Word*)work;Word* tail=state+(loop?4*n*groups:2*n*groups);"
            )
            producer = f"""if(loop) {{
                Word au={load("ap")},bu={load("bp")};Word* s=state+group*4*n;
                s[i+coord]=add(ax,au,&c);s[n+i+coord]=sub(ax,au,&c);
                s[2*n+i+coord]=add(by,bu,&c);s[3*n+i+coord]=sub(by,bu,&c);
            }}else{{Word* s=state+group*2*n;s[i+coord]=ax;s[n+i+coord]=by;}}"""
            core = f"""Word* s=state+group*(loop?4*n:2*n);
        if(loop) {{
            Word* f=s;Word* u=f+n;Word* fp=u+n;Word* up=fp+n;Word* g=tail;Word* scratch=g+n;
            {old}_hadamard(f,n,&c);{old}_hadamard(fp,n,&c);
            for(size_t i=0;i<n;i++)f[i]=multiply(f[i],fp[i],&c);
            {old}_hadamard(f,n,&c);
            for(size_t i=0;i<n;i++)f[i]=multiply(f[i],c.normalization,&c);
            {old}_cd(g,u,up,n,scratch,&c);
            for(size_t i=0;i<n;i++){{Word a=f[i],b=g[i];f[i]=half(add(a,b,&c),&c);u[i]=half(sub(a,b,&c),&c);}}
        }}else{{{old}_cd(tail,s,s+n,n,tail+n,&c);memcpy(s,tail,n*sizeof(Word));}}"""
        index = [j * d for j in range(lanes)]
        if bits == 128:

            def gather(a, index=index):
                return (
                    f"{p}_setr_epi32("
                    + ",".join(f"(int)({a})[{q}]" for q in index)
                    + ")"
                )
        elif bits == 256:

            def gather(a, index=index):
                return (
                    f"{p}_i32gather_epi32({a},{p}_setr_epi32("
                    + ",".join(map(str, index))
                    + "),4)"
                )
        else:

            def gather(a, index=index):
                return (
                    f"{p}_i32gather_epi32({p}_setr_epi32("
                    + ",".join(map(str, index))
                    + f"),{a},4)"
                )

        if vector_packing:
            source += transpose_code(f"{name}_pack{d}", lanes, 32, 2 * d - 1, points)
        source += f"""
static inline __attribute__((always_inline)) void {name}_rows{d}(uint32_t outx[23][{lanes}],uint32_t outy[23][{lanes}],
    const uint32_t* x,const uint32_t* y,const uint32_t coefficients[12][12],const uint32_t quotient[12][12],
    Word r,Word rq,const Context* restrict ctx) {{
    Word xx[{d}],yy[{d}],dx[{d}],dy[{d}];
#pragma GCC unroll 12
    for(size_t j=0;j<{d};j++) {{
        xx[j]=fixed({gather("x+j")},r,rq,ctx);yy[j]=fixed({gather("y+j")},r,rq,ctx);
    }}
    dx[0]=xx[0];dy[0]=yy[0];dx[1]=xx[1];dy[1]=yy[1];
#pragma GCC unroll 12
    for(size_t j=2;j<{d};j++){{dx[1]=add(dx[1],xx[j],ctx);dy[1]=add(dy[1],yy[j],ctx);}}
#pragma GCC unroll 12
    for(size_t k=2;k<{d};k++) {{
        Word a={p}_set1_epi32((int)coefficients[k][k]),b={p}_set1_epi32((int)quotient[k][k]);
        Word u=fixed(xx[k],a,b,ctx),v=fixed(yy[k],a,b,ctx);
#pragma GCC unroll 12
        for(size_t j=k+1;j<{d};j++) {{
            a={p}_set1_epi32((int)coefficients[k][j]);b={p}_set1_epi32((int)quotient[k][j]);
            u=add(u,fixed(xx[j],a,b,ctx),ctx);v=add(v,fixed(yy[j],a,b,ctx),ctx);
        }}
        dx[k]=u;dy[k]=v;
    }}
    for(size_t point=0;point<{2 * d - 1};point++) {{
        {p}_storeu_si{bits}((void*)outx[point],dx[0]);{p}_storeu_si{bits}((void*)outy[point],dy[0]);
#pragma GCC unroll 12
        for(size_t k=0;k<{d - 1};k++){{dx[k]=add(dx[k],dx[k+1],ctx);dy[k]=add(dy[k],dy[k+1],ctx);}}
    }}
}}
static void {name}_d{d}(uint32_t* restrict out,const uint32_t* x,const uint32_t* y,
    const FastCDPlanFF* p,uint32_t* restrict work,int loop,size_t rank,
    const uint32_t input[48][12],const uint32_t output[48][12]) {{
    (void)input;
    uint32_t prime=p->field.prime,inv=1;
    for(int i=0;i<5;i++)inv*=2-prime*inv;
    uint32_t r=(uint32_t)((UINT64_C(1)<<32)%prime),ri=1,b=r,e=prime-2;
    while(e){{if(e&1)ri=fastcd_ff_prime_mul(ri,b,&p->field);b=fastcd_ff_prime_mul(b,b,&p->field);e>>=1;}}
    Context c={{{p}_set1_epi32((int)prime),{p}_set1_epi32((int)(0u-inv)),{z},
        {p}_set1_epi32((int)fastcd_ff_prime_mul(p->inverse_dimension,r,&p->field))}};
    uint32_t stirling[12][12]={{{{1}}}},coefficients[12][12]={{{{0}}}},quotient[12][12]={{{{0}}}},factorial=1;
    for(size_t j=1;j<{d};j++)for(size_t k=1;k<=j;k++)
        stirling[j][k]=fastcd_ff_prime_mul((uint32_t)k,stirling[j-1][k],&p->field)+stirling[j-1][k-1],
        stirling[j][k]=stirling[j][k]>=prime?stirling[j][k]-prime:stirling[j][k];
    for(size_t k=0;k<{d};k++) {{
        if(k)factorial=fastcd_ff_prime_mul(factorial,(uint32_t)k,&p->field);
        for(size_t j=k;j<{d};j++){{coefficients[k][j]=fastcd_ff_prime_mul(factorial,stirling[j][k],&p->field);
            quotient[k][j]=(uint32_t)(((uint64_t)coefficients[k][j]<<32)/prime);}}
    }}
    Word rv={p}_set1_epi32((int)r),rq={p}_set1_epi32((int)(((uint64_t)r<<32)/prime));
    size_t n=p->dimension,size=n*(loop?2:1),groups=(rank+{points}-1)/{points};
    {state_declaration}
    for(size_t i=0;i<n;i+={lanes}) {{
        uint32_t xl[23][{lanes}],yl[23][{lanes}],xu[23][{lanes}],yu[23][{lanes}];
        {name}_rows{d}(xl,yl,x+i*{d},y+i*{d},coefficients,quotient,rv,rq,&c);
        if(loop){name}_rows{d}(xu,yu,x+(n+i)*{d},y+(n+i)*{d},coefficients,quotient,rv,rq,&c);
        for(size_t coord=0;coord<{lanes};coord++)for(size_t group=0;group<groups;group++) {{
            uint32_t a[{lanes}],b[{lanes}],ap[{lanes}],bp[{lanes}];
            for(size_t lane=0;lane<{lanes};lane++){{size_t row=group*{points}+lane;
                a[lane]=lane<{points}&&row<rank?xl[row][coord]:0;b[lane]=lane<{points}&&row<rank?yl[row][coord]:0;
                if(loop){{ap[lane]=lane<{points}&&row<rank?xu[row][coord]:0;bp[lane]=lane<{points}&&row<rank?yu[row][coord]:0;}}}}
            Word ax={load("a")},by={load("b")};
            {producer}
        }}
    }}
    for(size_t group=0;group<groups;group++) {{
        {core}
    }}
    {project_coordinates(lanes, bits, p, d, "state", "(loop?4*n:2*n)", "size", "output", "ri", points)}
}}
"""
    source += f"""void {name}_tensor(uint32_t* o,const uint32_t* a,const uint32_t* b,
    const FastCDPlanFF* p,uint32_t* w,int loop,size_t rank,const uint32_t input[48][12],const uint32_t output[48][12]) {{
    switch(p->field.degree) {{
"""
    for d in range(2, 13):
        source += (
            f"    case {d}: {name}_d{d}(o,a,b,p,w,loop,rank,input,output);break;\n"
        )
    source += "    }\n}\n"
    if vector_packing:
        for d in range(2, 13):
            points = point_width(2 * d - 1, lanes)
            groups = (2 * d - 2 + points) // points
            first = source.index(f"static void {name}_d{d}(")
            start = source.index(
                f"        for(size_t coord=0;coord<{lanes};coord++)", first
            )
            end = source.index("    for(size_t group=0;group<groups;group++)", start)
            source = (
                source[:start]
                + transpose_calls(
                    f"{name}_pack{d}", "uint32_t", lanes, points, groups, f"n*{points}"
                )
                + "    }\n"
                + source[end:]
            )
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
        (4, 128, "_mm", ["-msse4.1"]),
        (8, 256, "_mm256", ["-mavx2"]),
        (16, 512, "_mm512", ["-mavx512f", "-mavx512dq"]),
    ):
        generate(*args)
        generate(*args, multipoint=True)
