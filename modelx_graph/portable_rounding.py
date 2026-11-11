"""Native fixed-width arithmetic for the existing exact rounding algorithm.

No Python runtime, compiler integer extensions, or platform intrinsics. The
small unsigned type is private to the rounding helper, not a model value ABI.
"""

PORTABLE_U128_CXX = r'''
class mx_u128 {
public:
    uint64_t lo, hi;
    mx_u128(uint64_t a=0): lo(a), hi(0) {}
    mx_u128(uint64_t a, uint64_t b): lo(a), hi(b) {}
    explicit operator bool() const { return lo || hi; }
    explicit operator uint64_t() const { return lo; }
    bool operator==(mx_u128 b) const { return lo==b.lo && hi==b.hi; }
    bool operator!=(mx_u128 b) const { return !(*this==b); }
    bool operator<(mx_u128 b) const { return hi<b.hi || (hi==b.hi && lo<b.lo); }
    bool operator>(mx_u128 b) const { return b<*this; }
    bool operator>=(mx_u128 b) const { return !(*this<b); }
    mx_u128 operator&(mx_u128 b) const { return mx_u128(lo&b.lo,hi&b.hi); }
    mx_u128 operator+(mx_u128 b) const {
        uint64_t l=lo+b.lo;
        return mx_u128(l,hi+b.hi+(l<lo));
    }
    mx_u128 operator-(mx_u128 b) const {
        return mx_u128(lo-b.lo,hi-b.hi-(lo<b.lo));
    }
    mx_u128 operator<<(int n) const {
        if(n<=0) return *this;
        if(n>=128) return mx_u128();
        if(n>=64) return mx_u128(0,lo<<(n-64));
        return mx_u128(lo<<n,(hi<<n)|(lo>>(64-n)));
    }
    mx_u128 operator>>(int n) const {
        if(n<=0) return *this;
        if(n>=128) return mx_u128();
        if(n>=64) return mx_u128(hi>>(n-64),0);
        return mx_u128((lo>>n)|(hi<<(64-n)),hi>>n);
    }
    mx_u128& operator>>=(int n) { *this=*this>>n; return *this; }
    mx_u128 operator++(int) { mx_u128 old=*this; *this=*this+mx_u128(1); return old; }
    mx_u128 operator*(mx_u128 b) const {
        uint32_t a32[4]={(uint32_t)lo,(uint32_t)(lo>>32),(uint32_t)hi,(uint32_t)(hi>>32)};
        uint32_t b32[4]={(uint32_t)b.lo,(uint32_t)(b.lo>>32),(uint32_t)b.hi,(uint32_t)(b.hi>>32)};
        uint32_t r[4]={0,0,0,0};
        for(int i=0;i<4;i++) {
            uint64_t carry=0;
            for(int j=0;j<4-i;j++) {
                uint64_t v=(uint64_t)a32[i]*b32[j]+r[i+j]+carry;
                r[i+j]=(uint32_t)v;
                carry=v>>32;
            }
        }
        return mx_u128((uint64_t)r[0]|((uint64_t)r[1]<<32),
                       (uint64_t)r[2]|((uint64_t)r[3]<<32));
    }
    mx_u128 operator/(mx_u128 b) const {
        // The private rounding algorithm only supplies nonzero denominators.
        mx_u128 q, rem;
        for(int i=127;i>=0;i--) {
            bool carry=(rem.hi>>63)!=0;
            rem=rem<<1;
            rem.lo|=(i>=64 ? (hi>>(i-64)) : (lo>>i))&1;
            if(carry || rem>=b) {
                rem=rem-b;
                if(i>=64) q.hi|=((uint64_t)1)<<(i-64);
                else q.lo|=((uint64_t)1)<<i;
            }
        }
        return q;
    }
};
static int mx_clz64(uint64_t x) {
    int n=0;
    if(!x) return 64;
    while(!(x&(((uint64_t)1)<<63))) { x<<=1; n++; }
    return n;
}
'''
