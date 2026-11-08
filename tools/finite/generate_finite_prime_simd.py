#!/usr/bin/env python3
"""Single-product SIMD: pack independent CD descendants, vectorize ancestors.

This generator appends a prime-field backend to each register-value Montgomery
translation unit. Its workspace stays linear in N, independently of lane count.
"""

import subprocess

from generate_finite_simd import ROOT, WORK, generate


def append(lanes, bits, prefix, flags, build=True):
    p = prefix
    name = f"value{lanes}"

    def load(x):
        return f"{p}_loadu_si{bits}((const void*)({x}))"

    def store(x, v):
        return f"{p}_storeu_si{bits}((void*)({x}),{v})"

    z = f"{p}_setzero_si{bits}()"
    source = []
    # Child lanes occupy the low branch bits, and adjacent groups are coordinates
    for batch in [1 << k for k in reversed(range(lanes.bit_length()))]:
        func = f"{name}_node{batch}"
        signature = f"""static void {func}(uint32_t* restrict real,uint32_t* restrict imag,
    const uint32_t* xr,const uint32_t* xi,const uint32_t* yr,const uint32_t* yi,
    size_t n,uint32_t* restrict work,const Context* restrict ctx)"""
        if batch == lanes:
            source += [
                signature + " {",
                f"    {name}_pair_cross((Word*)real,(Word*)imag,(const Word*)xr,(const Word*)xi,",
                "        (const Word*)yr,(const Word*)yi,n,(Word*)work,ctx);",
                "}",
            ]
            continue
        coords = lanes // batch
        index = [r * 2 * batch + b for r in range(coords) for b in range(batch)]

        def gather(array, batch=batch, index=index):
            if bits == 128:
                return (
                    f"{p}_setr_epi32("
                    + ",".join(f"(int){array}[i*{2 * batch}+{q}]" for q in index)
                    + ")"
                )
            idx = f"{p}_setr_epi32(" + ",".join(map(str, index)) + ")"
            return (
                f"{p}_i32gather_epi32({array}+i*{2 * batch},{idx},4)"
                if bits == 256
                else f"{p}_i32gather_epi32({idx},{array}+i*{2 * batch},4)"
            )

        source += [
            signature + " {",
            f"    size_t h=n/2,size=n*{batch};",
            "    uint32_t* cr=work+4*size;uint32_t* ci=cr+size;",
            "    if(n>4) {",
            "        uint32_t* ar=work;uint32_t* ai=ar+size;uint32_t* br=ai+size;uint32_t* bi=br+size;",
            f"        for(size_t i=0;i<h;i++) for(size_t b=0;b<{batch};b++) {{",
            f"            size_t k=i*{batch}+b,l=(h+i)*{batch}+b,j=i*{2 * batch}+b;",
            "            uint32_t prime=(uint32_t)_mm_cvtsi128_si32("
            + ("ctx->prime" if bits == 128 else f"{p}_castsi{bits}_si128(ctx->prime)")
            + ");",
            "            uint32_t a=xr[k],b0=xi[l],c=xi[k],d=xr[l],e=yr[k],f=yi[l],g=yi[k],t=yr[l];",
            "            uint32_t ap=a+prime-b0,am=a+b0,ip=c+d,im=c+prime-d;",
            "            uint32_t bp=e+prime-f,bm=e+f,jp=g+t,jm=g+prime-t;",
            f"            ar[j]=ap>=prime?ap-prime:ap;ar[j+{batch}]=am>=prime?am-prime:am;",
            f"            ai[j]=ip>=prime?ip-prime:ip;ai[j+{batch}]=im>=prime?im-prime:im;",
            f"            br[j]=bp>=prime?bp-prime:bp;br[j+{batch}]=bm>=prime?bm-prime:bm;",
            f"            bi[j]=jp>=prime?jp-prime:jp;bi[j+{batch}]=jm>=prime?jm-prime:jm;",
            f"            if(i==0) ar[j]=ar[j+{batch}]=ai[j]=ai[j+{batch}]=br[j]=br[j+{batch}]=bi[j]=bi[j+{batch}]=0;",
            "        }",
            f"        {name}_node{2 * batch}(cr,ci,ar,ai,br,bi,h,ci+size,ctx);",
            "    } else {memset(cr,0,2*size*sizeof(uint32_t));}",
            f"    uint32_t gv[{lanes}],hv[{lanes}],dv[{lanes}],ev[{lanes}],keep[{lanes}];",
            f"    for(size_t k=0;k<{lanes};k++) {{",
            f"        size_t b=k%{batch};",
            f"        gv[k]=xr[h*{batch}+b];hv[k]=xi[h*{batch}+b];",
            f"        dv[k]=yr[h*{batch}+b];ev[k]=yi[h*{batch}+b];keep[k]=k<{batch}?0:UINT32_MAX;",
            "    }",
            f"    Word gp={load('gv')},gq={load('hv')},dp={load('dv')},dq={load('ev')},mask={load('keep')};",
            "    Word gw=add(gp,gq,ctx),dw=add(dp,dq,ctx);",
            f"    Word ap={z},aq={z},aw={z};",
            f"    for(size_t i=0;i<h;i+={coords}) {{",
            f"        Word ur={load(f'xr+i*{batch}')},ui={load(f'xi+i*{batch}')};",
            f"        Word vr={load(f'yr+i*{batch}')},vi={load(f'yi+i*{batch}')};",
            f"        Word sr={load(f'xr+(h+i)*{batch}')},si={load(f'xi+(h+i)*{batch}')};",
            f"        Word tr={load(f'yr+(h+i)*{batch}')},ti={load(f'yi+(h+i)*{batch}')};",
            f"        if(i==0) {{ur={p}_and_si{bits}(ur,mask);ui={p}_and_si{bits}(ui,mask);",
            f"            vr={p}_and_si{bits}(vr,mask);vi={p}_and_si{bits}(vi,mask);",
            f"            sr={p}_and_si{bits}(sr,mask);si={p}_and_si{bits}(si,mask);",
            f"            tr={p}_and_si{bits}(tr,mask);ti={p}_and_si{bits}(ti,mask);}}",
            f"        {name}_ValueNonrecursive a={name}_vnonrecursive(ur,vr,sr,tr,gp,dp,ctx);",
            f"        {name}_ValueNonrecursive b={name}_vnonrecursive(ui,vi,si,ti,gq,dq,ctx);",
            f"        {name}_ValueNonrecursive c={name}_vnonrecursive(add(ur,ui,ctx),add(vr,vi,ctx),add(sr,si,ctx),add(tr,ti,ctx),gw,dw,ctx);",
            "        ap=add(ap,a.scalar,ctx);aq=add(aq,b.scalar,ctx);aw=add(aw,c.scalar,ctx);",
            f"        Word pr={gather('cr')},pi={gather('ci')};",
            f"        const uint32_t* cmr=cr+{batch};const uint32_t* cmi=ci+{batch};",
            f"        Word mr={gather('cmr')},mi={gather('cmi')};",
            "        Word fr=add(sub(a.first,b.first,ctx),half(add(pr,mr,ctx),ctx),ctx);",
            "        Word fi=add(sub(sub(c.first,a.first,ctx),b.first,ctx),half(add(pi,mi,ctx),ctx),ctx);",
            "        Word gr=add(sub(a.second,b.second,ctx),half(sub(mi,pi,ctx),ctx),ctx);",
            "        Word gi=add(sub(sub(c.second,a.second,ctx),b.second,ctx),half(sub(pr,mr,ctx),ctx),ctx);",
            f"        {store(f'real+i*{batch}', 'fr')};{store(f'imag+i*{batch}', 'fi')};",
            f"        {store(f'real+(h+i)*{batch}', 'gr')};{store(f'imag+(h+i)*{batch}', 'gi')};",
            "    }",
            f"    uint32_t av[{lanes}],bv[{lanes}],cv[{lanes}];",
            f"    {store('av', 'ap')};{store('bv', 'aq')};{store('cv', 'aw')};",
            "    uint32_t prime=(uint32_t)_mm_cvtsi128_si32("
            + ("ctx->prime" if bits == 128 else f"{p}_castsi{bits}_si128(ctx->prime)")
            + ");",
            f"    for(size_t b=0;b<{batch};b++) {{",
            "        uint32_t a=0,q=0,w=0;",
            f"        for(size_t k=b;k<{lanes};k+={batch}) {{",
            "            uint32_t u=a+av[k],v=q+bv[k],s=w+cv[k];",
            "            a=u>=prime?u-prime:u;q=v>=prime?v-prime:v;w=s>=prime?s-prime:s;",
            "        }",
            "        uint32_t r=a+prime-q,s=w+prime-a;",
            "        r=r>=prime?r-prime:r;s=s>=prime?s-prime:s;s=s+prime-q;s=s>=prime?s-prime:s;",
            f"        real[b]=imag[b]=0;real[h*{batch}+b]=r;imag[h*{batch}+b]=s;",
            "    }",
            "}",
        ]
    source += [
        f"""static void {name}_prime_cd(uint32_t* restrict out,const uint32_t* x,const uint32_t* y,
    size_t n,uint32_t* restrict work,const Context* restrict ctx) {{
    size_t h=n/2;
    {name}_node1(out,out+h,x,x+h,y,y+h,h,work,ctx);
    Word gamma={p}_set1_epi32((int)x[h]),delta={p}_set1_epi32((int)y[h]);
    Word alpha={p}_set1_epi32((int)x[0]),beta={p}_set1_epi32((int)y[0]);
    Word scalar={z},dot={z};
    uint32_t kv[{lanes}];for(size_t j=0;j<{lanes};j++)kv[j]=j?UINT32_MAX:0;
    Word mask={load("kv")};
    for(size_t i=0;i<h;i+={lanes}) {{
        Word u={load("x+i")},v={load("y+i")},s={load("x+h+i")},t={load("y+h+i")};
        if(i==0) {{u={p}_and_si{bits}(u,mask);v={p}_and_si{bits}(v,mask);
            s={p}_and_si{bits}(s,mask);t={p}_and_si{bits}(t,mask);}}
        {name}_ValueNonrecursive a={name}_vnonrecursive(u,v,s,t,gamma,delta,ctx);
        scalar=add(scalar,a.scalar,ctx);
        Word f=add({load("out+i")},a.first,ctx),g=sub(a.second,{load("out+h+i")},ctx);
        {store("out+i", "f")};{store("out+h+i", "g")};
    }}
    out[h]=horizontal(scalar,ctx);
    for(size_t i=0;i<n;i+={lanes}) {{
        Word a={load("x+i")},b={load("y+i")};
        dot=add(dot,multiply(a,b,ctx),ctx);
        Word v=add(multiply(alpha,b,ctx),multiply(beta,a,ctx),ctx);
        v=add(v,{load("out+i")},ctx);{store("out+i", "v")};
    }}
    Word aa=multiply(alpha,beta,ctx);aa=add(aa,aa,ctx);
    uint32_t av[{lanes}];{store("av", "aa")};
    uint32_t prime=(uint32_t)_mm_cvtsi128_si32({("ctx->prime" if bits == 128 else f"{p}_castsi{bits}_si128(ctx->prime)")});
    uint32_t value=av[0]+prime-horizontal(dot,ctx);out[0]=value>=prime?value-prime:value;
}}
"""
    ]
    perm = []
    for width in (1 << k for k in range(lanes.bit_length() - 1)):
        idx = [i ^ width for i in range(lanes)]
        if bits == 128:
            expr = (
                f"{p}_shuffle_epi32(a,{sum(q << (2 * i) for i, q in enumerate(idx))})"
            )
        elif bits == 256:
            expr = (
                f"{p}_permutevar8x32_epi32(a,{p}_setr_epi32("
                + ",".join(map(str, idx))
                + "))"
            )
        else:
            expr = (
                f"{p}_permutexvar_epi32({p}_setr_epi32("
                + ",".join(map(str, idx))
                + "),a)"
            )
        # A negative lane uses b-a and the other lanes use a+b
        mask = [-1 if i & width else 0 for i in range(lanes)]
        maskexpr = f"{p}_setr_epi32(" + ",".join(map(str, mask)) + ")"
        perm += [
            f"        {{Word b={expr},u=add(a,b,ctx),v=sub(b,a,ctx),m={maskexpr};",
            f"        a={p}_or_si{bits}({p}_and_si{bits}(v,m),{p}_andnot_si{bits}(m,u));}}",
        ]
    source += [
        f"""static void {name}_prime_hadamard(uint32_t* data,size_t n,const Context* restrict ctx) {{
    for(size_t base=0;base<n;base+={lanes}) {{
        Word a={load("data+base")};
{chr(10).join(perm)}
        {store("data+base", "a")};
    }}
    for(size_t width={lanes};width<n;width*=2) for(size_t base=0;base<n;base+=2*width)
        for(size_t j=0;j<width;j+={lanes}) {{
            Word a={load("data+base+j")},b={load("data+base+width+j")};
            Word u=add(a,b,ctx),v=sub(a,b,ctx);
            {store("data+base+j", "u")};{store("data+base+width+j", "v")};
        }}
}}
void {name}_prime_tensor(uint32_t* restrict out,const uint32_t* x,const uint32_t* y,
    const FastCDPlanFF* p,uint32_t* restrict work,int loop,size_t rank,
    const uint32_t input[48][12],const uint32_t output[48][12]) {{
    (void)rank;(void)input;(void)output;
    uint32_t prime=p->field.prime,inv=1;
    for(int i=0;i<5;i++) inv*=2-prime*inv;
    uint32_t r=(uint32_t)((UINT64_C(1)<<32)%prime),r2=fastcd_ff_prime_mul(r,r,&p->field);
    Context c={{ {p}_set1_epi32((int)prime),{p}_set1_epi32((int)(0u-inv)),
        {p}_set1_epi32((int)r2),{p}_set1_epi32((int)fastcd_ff_prime_mul(p->inverse_dimension,r,&p->field)) }};
    size_t n=p->dimension,size=n*(loop?2:1);
    uint32_t* ex=work;uint32_t* ey=ex+size;uint32_t* eo=ey+size;uint32_t* scratch=eo+size;
    for(size_t i=0;i<size;i+={lanes}) {{
        Word a=multiply({load("x+i")},c.r2,&c),b=multiply({load("y+i")},c.r2,&c);
        {store("ex+i", "a")};{store("ey+i", "b")};
    }}
    if(loop) {{
        uint32_t* f=scratch;uint32_t* u=f+n;uint32_t* fp=u+n;uint32_t* up=fp+n;uint32_t* g=up+n;
        for(size_t i=0;i<n;i+={lanes}) {{
            Word a={load("ex+i")},b={load("ex+n+i")},d={load("ey+i")},e={load("ey+n+i")};
            Word af=add(a,b,&c),au=sub(a,b,&c),bf=add(d,e,&c),bu=sub(d,e,&c);
            {store("f+i", "af")};{store("u+i", "au")};{store("fp+i", "bf")};{store("up+i", "bu")};
        }}
        {name}_prime_hadamard(f,n,&c);{name}_prime_hadamard(fp,n,&c);
        for(size_t i=0;i<n;i+={lanes}) {{Word a=multiply({load("f+i")},{load("fp+i")},&c);{store("f+i", "a")};}}
        {name}_prime_hadamard(f,n,&c);
        {name}_prime_cd(g,u,up,n,g+n,&c);
        for(size_t i=0;i<n;i+={lanes}) {{
            Word a=multiply({load("f+i")},c.normalization,&c),b={load("g+i")};
            Word v=half(add(a,b,&c),&c),w=half(sub(a,b,&c),&c);
            {store("eo+i", "v")};{store("eo+n+i", "w")};
        }}
    }} else {name}_prime_cd(eo,ex,ey,n,scratch,&c);
    Word one={p}_set1_epi32(1);
    for(size_t i=0;i<size;i+={lanes}) {{Word a=multiply({load("eo+i")},one,&c);{store("out+i", "a")};}}
}}
"""
    ]
    path = WORK / f"{name}.c"
    path.write_text(path.read_text() + "\n" + "\n".join(source))
    if build:
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
        generate(*args, fast=True, values=True)
        append(*args)
