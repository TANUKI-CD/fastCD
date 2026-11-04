"""Explicit SIMD transforms and sign-lane interleaving for CD descendants."""


def optimize_nodes(text, name, lanes, bits, prefix, scalar_bits):
    p = prefix

    def constant(values):
        if bits == 512 and scalar_bits == 16:
            return f"{p}_set_epi16(" + ",".join(map(str, reversed(values))) + ")"
        return f"{p}_setr_epi{scalar_bits}(" + ",".join(map(str, values)) + ")"

    def zip_words(a, b, batch):
        if bits == 512:
            indices = [
                (j // (2 * batch)) * batch + j % batch + (lanes if j & (batch) else 0)
                for j in range(lanes)
            ]
            return (
                f"{p}_permutex2var_epi{scalar_bits}({a},{constant(indices)},{b})",
                f"{p}_permutex2var_epi{scalar_bits}({a},{constant([q + lanes // 2 for q in indices])},{b})",
            )
        width = batch * scalar_bits
        if width == 128:
            return (
                f"{p}_permute2x128_si256({a},{b},0x20)",
                f"{p}_permute2x128_si256({a},{b},0x31)",
            )
        lo = f"{p}_unpacklo_epi{width}({a},{b})"
        hi = f"{p}_unpackhi_epi{width}({a},{b})"
        if bits == 128:
            return lo, hi
        return (
            f"{p}_permute2x128_si256({lo},{hi},0x20)",
            f"{p}_permute2x128_si256({lo},{hi},0x31)",
        )

    for batch in (1 << k for k in range(lanes.bit_length() - 1)):
        first = text.index(f"static void {name}_node{batch}(")
        start = text.index("        for(size_t i=0;i<h;i++)", first)
        end = text.index(f"        {name}_node{batch * 2}(", start)
        coord = lanes // batch
        mask = constant([0 if j < 2 * batch else -1 for j in range(lanes)])
        lines = [f"        for(size_t i=0;i<h;i+={coord}) {{"]
        for out, low, high, plus, minus in (
            ("ar", "xr", "xi", "sub", "add"),
            ("ai", "xi", "xr", "add", "sub"),
            ("br", "yr", "yi", "sub", "add"),
            ("bi", "yi", "yr", "add", "sub"),
        ):
            lines += [
                f"            Word {out}a={p}_loadu_si{bits}((const void*)({low}+i*{batch}));",
                f"            Word {out}b={p}_loadu_si{bits}((const void*)({high}+(h+i)*{batch}));",
                f"            Word {out}p={plus}({out}a,{out}b,ctx),{out}m={minus}({out}a,{out}b,ctx);",
            ]
            lo, hi = zip_words(out + "p", out + "m", batch)
            lines += [
                f"            Word {out}lo={lo},{out}hi={hi};",
                f"            if(i==0){out}lo={p}_and_si{bits}({out}lo,{mask});",
                f"            {p}_storeu_si{bits}((void*)({out}+i*{2 * batch}),{out}lo);",
                f"            {p}_storeu_si{bits}((void*)({out}+i*{2 * batch}+{lanes}),{out}hi);",
            ]
        lines += ["        }\n"]
        text = text[:start] + "\n".join(lines) + text[end:]
    return text
