#!/usr/bin/env python3
"""Mix point lanes and descendant lanes without padding every field to width L."""

import subprocess

from finite_vector_packing import optimize_nodes
from generate_finite_simd import ROOT, WORK


def point_width(rank, lanes):
    candidates = [
        1 << k for k in range(lanes.bit_length()) if (1 << k) >= max(1, lanes // 4)
    ]
    return min(candidates, key=lambda n: (((rank + n - 1) // n) * n, -n))


def generate(lanes, bits, prefix, flags, build=True):
    p = prefix
    old = f"value{lanes}"
    name = f"multi{lanes}"
    text = (WORK / f"{old}.c").read_text()
    code = text[: text.index(f"void {old}_run(")]
    first = text.index("static uint32_t horizontal(")
    last = text.index(f"void {old}_tensor(", first)
    code += text[first:last]
    first = text.index(f"static void {old}_node{lanes}(")
    last = text.index(f"void {old}_prime_tensor(", first)
    code += text[first:last]
    for word in ("even", "odd"):
        code = code.replace(
            f"{p}_mullo_epi32({word},c->negative_inverse)",
            f"{p}_mul_epu32({word},c->negative_inverse)",
        )
    z = f"{p}_setzero_si{bits}()"

    def load(a):
        return f"{p}_loadu_si{bits}((const void*)({a}))"

    def store(a, b):
        return f"{p}_storeu_si{bits}((void*)({a}),{b})"

    scalarp = "ctx->prime" if bits == 128 else f"{p}_castsi{bits}_si128(ctx->prime)"
    points = sorted({point_width(2 * d - 1, lanes) for d in range(2, 13)})
    for points_count in points:
        P = points_count
        coords = lanes // P
        if P == lanes:
            continue
        code += f"""
static void {name}_cd{P}(uint32_t* restrict out,const uint32_t* x,const uint32_t* y,
    size_t n,uint32_t* restrict work,const Context* restrict ctx) {{
    size_t h=n/2;{old}_node{P}(out,out+h*{P},x,x+h*{P},y,y+h*{P},h,work,ctx);
    uint32_t gv[{lanes}],dv[{lanes}],av[{lanes}],bv[{lanes}],kv[{lanes}];
    for(size_t k=0;k<{lanes};k++){{size_t point=k%{P};gv[k]=x[h*{P}+point];dv[k]=y[h*{P}+point];
        av[k]=x[point];bv[k]=y[point];kv[k]=k<{P}?0:UINT32_MAX;}}
    Word gamma={load("gv")},delta={load("dv")},alpha={load("av")},beta={load("bv")},keep={load("kv")};
    Word sum={z},dot={z};
    for(size_t i=0;i<h;i+={coords}) {{
        Word u={load(f"x+i*{P}")},v={load(f"y+i*{P}")},s={load(f"x+(h+i)*{P}")},t={load(f"y+(h+i)*{P}")};
        if(i==0){{u={p}_and_si{bits}(u,keep);v={p}_and_si{bits}(v,keep);s={p}_and_si{bits}(s,keep);t={p}_and_si{bits}(t,keep);}}
        {old}_ValueLocal a={old}_vlocal(u,v,s,t,gamma,delta,ctx);sum=add(sum,a.scalar,ctx);
        Word f=add({load(f"out+i*{P}")},a.first,ctx),g=sub(a.second,{load(f"out+(h+i)*{P}")},ctx);
        {store(f"out+i*{P}", "f")};{store(f"out+(h+i)*{P}", "g")};
    }}
    uint32_t partial[{lanes}];{store("partial", "sum")};
    uint32_t prime=(uint32_t)_mm_cvtsi128_si32({scalarp});
    for(size_t point=0;point<{P};point++){{uint32_t a=0;
        for(size_t k=point;k<{lanes};k+={P}){{uint32_t b=a+partial[k];a=b>=prime?b-prime:b;}}
        out[h*{P}+point]=a;
    }}
    for(size_t i=0;i<n*{P};i+={lanes}){{Word a={load("x+i")},b={load("y+i")};
        dot=add(dot,multiply(a,b,ctx),ctx);Word term=add(multiply(alpha,b,ctx),multiply(beta,a,ctx),ctx);
        term=add(term,{load("out+i")},ctx);{store("out+i", "term")};}}
    Word aa=multiply(alpha,beta,ctx);aa=add(aa,aa,ctx);{store("av", "aa")};{store("partial", "dot")};
    for(size_t point=0;point<{P};point++){{uint32_t a=0;
        for(size_t k=point;k<{lanes};k+={P}){{uint32_t b=a+partial[k];a=b>=prime?b-prime:b;}}
        uint32_t v=av[point]+prime-a;out[point]=v>=prime?v-prime:v;
    }}
}}
"""
        tile = max(8, coords)
        registers = tile // coords
        code += f"""static void {name}_hadamard{P}(uint32_t* a,size_t n,const Context* restrict ctx) {{
    for(size_t base=0;base<n;base+={tile}) {{
"""
        previous = []
        for j in range(registers):
            variable = f"a0_{j}"
            previous.append(variable)
            code += f"        Word {variable}={load(f'a+base*{P}+{j * lanes}')};\n"
        for stage in range(tile.bit_length() - 1):
            width = 1 << stage
            next_values = []
            if width < coords:
                idx = [i ^ (width * P) for i in range(lanes)]
                if bits == 128:

                    def perm(a, idx=idx):
                        return f"{p}_shuffle_epi32({a},{sum(q << (2 * i) for i, q in enumerate(idx))})"
                elif bits == 256:

                    def perm(a, idx=idx):
                        return (
                            f"{p}_permutevar8x32_epi32({a},{p}_setr_epi32("
                            + ",".join(map(str, idx))
                            + "))"
                        )
                else:

                    def perm(a, idx=idx):
                        return (
                            f"{p}_permutexvar_epi32({p}_setr_epi32("
                            + ",".join(map(str, idx))
                            + f"),{a})"
                        )

                mask = (
                    f"{p}_setr_epi32("
                    + ",".join("-1" if i & (width * P) else "0" for i in range(lanes))
                    + ")"
                )
                for j, a in enumerate(previous):
                    b = f"b{stage}_{j}"
                    v = f"a{stage + 1}_{j}"
                    next_values.append(v)
                    code += f"        Word {b}={perm(a)},u{stage}_{j}=add({a},{b},ctx),v{stage}_{j}=sub({b},{a},ctx);\n"
                    code += f"        Word {v}={p}_or_si{bits}({p}_and_si{bits}(v{stage}_{j},{mask}),{p}_andnot_si{bits}({mask},u{stage}_{j}));\n"
            else:
                jump = width // coords
                for j in range(registers):
                    v = f"a{stage + 1}_{j}"
                    next_values.append(v)
                    op = "sub" if j & jump else "add"
                    left = previous[j - jump] if j & jump else previous[j]
                    right = previous[j] if j & jump else previous[j + jump]
                    code += f"        Word {v}={op}({left},{right},ctx);\n"
            previous = next_values
        for j, v in enumerate(previous):
            code += f"        {store(f'a+base*{P}+{j * lanes}', v)};\n"
        code += f"""    }}
    for(size_t width={tile};width<n;width*=2)for(size_t base=0;base<n;base+=2*width)
        for(size_t j=0;j<width;j+={coords}){{Word u={load(f"a+(base+j)*{P}")},v={load(f"a+(base+width+j)*{P}")};
            Word s=add(u,v,ctx),t=sub(u,v,ctx);{store(f"a+(base+j)*{P}", "s")};{store(f"a+(base+width+j)*{P}", "t")};}}
}}
"""
    code += f"""static Context {name}_context(const FastCDPlanFF* p) {{
    uint32_t prime=p->field.prime,inv=1;for(int i=0;i<5;i++)inv*=2-prime*inv;
    uint32_t r=(uint32_t)((UINT64_C(1)<<32)%prime);
    Context c={{{p}_set1_epi32((int)prime),{p}_set1_epi32((int)(0u-inv)),{z},
        {p}_set1_epi32((int)fastcd_ff_prime_mul(p->inverse_dimension,r,&p->field))}};return c;
}}
void {name}_cd_encoded(uint32_t* out,const uint32_t* x,const uint32_t* y,const FastCDPlanFF* p,uint32_t* work,size_t points) {{
    Context c={name}_context(p);switch(points) {{
"""
    for P in points:
        call = (
            f"{old}_cd((Word*)out,(const Word*)x,(const Word*)y,p->dimension,(Word*)work,&c)"
            if P == lanes
            else f"{name}_cd{P}(out,x,y,p->dimension,work,&c)"
        )
        code += f"    case {P}: {call};break;\n"
    code += "    }\n}\n"
    code += f"""void {name}_finish_encoded(uint32_t* state,const FastCDPlanFF* p,uint32_t* tail,size_t points) {{
    Context c={name}_context(p);size_t n=p->dimension,words=n*points;
    uint32_t* f=state;uint32_t* u=f+words;uint32_t* fp=u+words;uint32_t* up=fp+words;uint32_t* g=tail;uint32_t* scratch=g+words;
    switch(points) {{
"""
    for P in points:

        def h(a, P=P):
            return (
                f"{old}_hadamard((Word*){a},n,&c)"
                if P == lanes
                else f"{name}_hadamard{P}({a},n,&c)"
            )

        cd = (
            f"{old}_cd((Word*)g,(const Word*)u,(const Word*)up,n,(Word*)scratch,&c)"
            if P == lanes
            else f"{name}_cd{P}(g,u,up,n,scratch,&c)"
        )
        code += f"    case {P}: {{\n        {h('f')};{h('fp')};\n"
        code += f"        for(size_t i=0;i<words;i+={lanes}){{Word a=multiply({load('f+i')},{load('fp+i')},&c);{store('f+i', 'a')};}}\n"
        code += f"        {h('f')};\n"
        code += f"        if(p->inverse_dimension!=1)for(size_t i=0;i<words;i+={lanes}){{Word a=multiply({load('f+i')},c.normalization,&c);{store('f+i', 'a')};}}\n"
        code += f"        {cd};break;\n    }}\n"
    code += f"""    }}
    for(size_t i=0;i<words;i+={lanes}){{Word a={load("f+i")},b={load("g+i")};
        Word s=half(add(a,b,&c),&c),t=half(sub(a,b,&c),&c);{store("f+i", "s")};{store("u+i", "t")};}}
}}
"""
    path = WORK / f"{name}.c"
    path.write_text(code)
    code = optimize_nodes(code, old, lanes, bits, prefix, 32)
    path.write_text(code)
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
