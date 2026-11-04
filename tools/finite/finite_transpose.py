"""Pack coordinate vectors into point vectors using register transposes."""


def code(name, lanes, scalar, rank, points):
    assert lanes * scalar == 512
    ctype = f"uint{scalar}_t"

    def index(values):
        return (
            ("_mm512_set_epi16(" + ",".join(map(str, reversed(values))) + ")")
            if scalar == 16
            else ("_mm512_setr_epi32(" + ",".join(map(str, values)) + ")")
        )

    text = f"""static inline __attribute__((always_inline)) void {name}({ctype}* restrict out,
 const {ctype} low[{rank}][{lanes}],const {ctype} high[{rank}][{lanes}],size_t group,int kind,const Context* ctx){{
 Word value[{points}];
#pragma GCC unroll {points}
 for(size_t j=0;j<{points};j++){{size_t row=group*{points}+j;Word a=_mm512_setzero_si512(),b=a;
  if(row<{rank}){{a=_mm512_loadu_si512((const void*)low[row]);if(kind<2)b=_mm512_loadu_si512((const void*)high[row]);}}
  value[j]=kind==0?add(a,b,ctx):kind==1?sub(a,b,ctx):a;
 }}
"""
    width = 1
    while width < points:
        ix = [
            (j // (2 * width)) * width + j % width + (lanes if j & width else 0)
            for j in range(lanes)
        ]
        for base in range(0, points, 2 * width):
            for j in range(width):
                a, b = base + j, base + j + width
                text += f" {{Word a=value[{a}],b=value[{b}];value[{a}]=_mm512_permutex2var_epi{scalar}(a,{index(ix)},b);value[{b}]=_mm512_permutex2var_epi{scalar}(a,{index([i + lanes // 2 for i in ix])},b);}}\n"
        width *= 2
    for i in range(points):
        reverse = int(f"{i:0{points.bit_length() - 1}b}"[::-1], 2)
        text += f" _mm512_storeu_si512((void*)(out+{i * lanes}),value[{reverse}]);\n"
    return text + "}\n"


def calls(name, ctype, lanes, points, groups, words):
    def cast(a):
        return f"(const {ctype} (*)[{lanes}]){a}"

    text = ""
    for group in range(groups):
        text += f" {{ {ctype}* s=state+{group}*(loop?4:2)*{words};\n"

        def call(offset, lo, hi, kind, group=group):
            return f" {name}(s+{offset},{cast(lo)},{cast(hi) if hi else 'NULL'},{group},{kind},&c);\n"

        text += (
            " if(loop){\n"
            + call(f"i*{points}", "xl", "xu", 0)
            + call(f"{words}+i*{points}", "xl", "xu", 1)
            + call(f"2*{words}+i*{points}", "yl", "yu", 0)
            + call(f"3*{words}+i*{points}", "yl", "yu", 1)
        )
        text += (
            " }else{\n"
            + call(f"i*{points}", "xl", None, 2)
            + call(f"{words}+i*{points}", "yl", None, 2)
            + " }\n }\n"
        )
    return text
