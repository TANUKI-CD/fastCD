"""Generate projection with SIMD lanes across public output coordinates."""


def code(lanes, bits, prefix, d, state, stride, size, output, ri, point_lanes=None):
    p = prefix
    point_lanes = point_lanes or lanes
    if bits == 128:

        def gather(a):
            return (
                f"{p}_setr_epi32("
                + ",".join(f"(int)({a})[{j * point_lanes}]" for j in range(lanes))
                + ")"
            )
    else:
        indices = (
            f"{p}_setr_epi32("
            + ",".join(str(j * point_lanes) for j in range(lanes))
            + ")"
        )
        gather = (
            (lambda a: f"{p}_i32gather_epi32({a},{indices},4)")
            if bits == 256
            else (lambda a: f"{p}_i32gather_epi32({indices},{a},4)")
        )
    z = f"{p}_setzero_si{bits}()"
    scatter = (
        (
            f"{p}_i32scatter_epi32(out+i*{d}+column,{p}_setr_epi32("
            + ",".join(str(j * d) for j in range(lanes))
            + "),result,4);"
        )
        if bits == 512
        else f"""
                uint32_t canonical[{lanes}];{p}_storeu_si{bits}((void*)canonical,result);
                for(size_t lane=0;lane<{lanes};lane++)out[(i+lane)*{d}+column]=canonical[lane];"""
    )
    return f"""
    uint32_t project[48][12],project_q[48][12];
    for(size_t row=0;row<rank;row++)for(size_t j=0;j<{d};j++){{
        project[row][j]=fastcd_ff_prime_mul({output}[row][j],{ri},&p->field);
        project_q[row][j]=(uint32_t)(((uint64_t)project[row][j]<<32)/prime);
    }}
    for(size_t i=0;i<{size};i+={lanes}) {{
        Word values[48];
        for(size_t row=0;row<rank;row++){{
            const uint32_t* base=(const uint32_t*)({state})+((row/{point_lanes})*({stride})+i)*{point_lanes}+row%{point_lanes};
            values[row]={gather("base")};
        }}
#pragma GCC unroll 12
        for(size_t first=0;first<{d};first+=4) {{
            Word acc[4][4];
#pragma GCC unroll 4
            for(size_t k=0;k<4;k++) {{
#pragma GCC unroll 4
                for(size_t j=0;j<4;j++){{
                    if(first+j<{d})acc[k][j]=k<rank?fixed(values[k],
                        {p}_set1_epi32((int)project[k][first+j]),{p}_set1_epi32((int)project_q[k][first+j]),&c):{z};
                }}
            }}
            for(size_t row=4;row<rank;row+=4) {{
#pragma GCC unroll 4
                for(size_t k=0;k<4;k++)if(row+k<rank) {{
#pragma GCC unroll 4
                    for(size_t j=0;j<4;j++)if(first+j<{d}) {{
                        Word term=fixed(values[row+k],{p}_set1_epi32((int)project[row+k][first+j]),
                            {p}_set1_epi32((int)project_q[row+k][first+j]),&c);
                        acc[k][j]=add(acc[k][j],term,&c);
                    }}
                }}
            }}
#pragma GCC unroll 4
            for(size_t j=0;j<4;j++)if(first+j<{d}) {{
                size_t column=first+j;
                Word result=add(add(acc[0][j],acc[1][j],&c),add(acc[2][j],acc[3][j],&c),&c);
                {scatter}
            }}
        }}
    }}
"""
