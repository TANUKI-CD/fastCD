#!/usr/bin/env python3
"""Use descendant packing with 16-bit Barrett/Montgomery prime arithmetic."""

from generate_finite_shuffle import prime32


def append(text, prefix, mont):
    helpers = """
static inline Word difference(Word a,Word b,Word x,Word y,const Context* c){return sub(multiply(a,b,c),multiply(x,y,c),c);}
static inline Word normalize(Word a,const Context* c){return multiply(a,c->normalization,c);}
static Context context(const FastCDPlanFF* p){
 uint32_t prime=p->field.prime,inv=1;for(int k=0;k<4;k++)inv*=2-prime*inv;
 uint32_t r=R_VALUE;
 Context c={_mm512_set1_epi16((short)prime),_mm512_set1_epi16((short)INV_VALUE),
 _mm512_set1_epi16((short)fastcd_ff_prime_mul(p->inverse_dimension,r,&p->field))};return c;
}
""".replace("R_VALUE", "65536%prime" if mont else "1").replace(
        "INV_VALUE", "(0u-inv)" if mont else "(65536/prime)"
    )
    core = (
        prime32()
        .replace("shuffle32", prefix)
        .replace("horizontal(", "prime_horizontal(")
    )
    if mont:
        original = (
            "for(size_t i=0;i<size;i++){ex[i]=(uint16_t)x[i];ey[i]=(uint16_t)y[i];}"
        )
        replacement = """uint32_t residue=65536%p->field.prime;
    Word r2=_mm512_set1_epi16((short)fastcd_ff_prime_mul(residue,residue,&p->field));
    for(size_t i=0;i<size;i+=32){
        Word ax=_mm512_loadu_si512((const void*)(x+i)),bx=_mm512_loadu_si512((const void*)(x+i+16));
        Word ay=_mm512_loadu_si512((const void*)(y+i)),by=_mm512_loadu_si512((const void*)(y+i+16));
        Word u=_mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(ax)),_mm512_cvtepi32_epi16(bx),1);
        Word v=_mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(ay)),_mm512_cvtepi32_epi16(by),1);
        _mm512_storeu_si512((void*)(ex+i),multiply(u,r2,&c));_mm512_storeu_si512((void*)(ey+i),multiply(v,r2,&c));
    }"""
        assert original in core
        core = core.replace(original, replacement)
        original = "for(size_t i=0;i<size;i++)out[i]=result[i];"
        replacement = """Word one=_mm512_set1_epi16(1);
    for(size_t i=0;i<size;i+=32){Word a=multiply(_mm512_loadu_si512((const void*)(result+i)),one,&c);
        _mm512_storeu_si512((void*)(out+i),_mm512_cvtepu16_epi32(_mm512_castsi512_si256(a)));
        _mm512_storeu_si512((void*)(out+i+16),_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(a,1)));
    }"""
        assert original in core
        core = core.replace(original, replacement)
    return text + "\n" + helpers + core
