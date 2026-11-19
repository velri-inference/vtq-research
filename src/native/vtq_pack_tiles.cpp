// Experimental encoder-side serializer, reference: src/vtq_format.py. No decoder change.
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <algorithm>
using B=uint8_t;
extern "C" int vtq_io_abi(){ return 1; }
// Transpose even/odd row planes, low spatial index is the low bit.
template<int bit> static inline void plane(B *dst, const B u[16][16]) {
    for(int odd=0;odd<2;++odd)for(int j=0;j<16;++j){unsigned v=0;
        for(int p=0;p<8;++p)v|=((u[2*p+odd][j]>>bit)&1u)<<p;
        dst[16*odd+j]=B(v);
    }
}
static inline void eo(B *d,const B u[16][16],int bit){switch(bit){case 0:return plane<0>(d,u);case 1:return plane<1>(d,u);case 2:return plane<2>(d,u);case 3:return plane<3>(d,u);default:return plane<4>(d,u);}}
template<int n> static inline void subword(B *d,const B u[16][16]){
    constexpr int pmax=8/n;
    for(int hp=0;hp<8/pmax;++hp)for(int odd=0;odd<2;++odd)for(int j=0;j<16;++j){unsigned v=0;
        for(int p=0;p<pmax;++p)v|=(u[hp*2*pmax+2*p+odd][j]&((1u<<n)-1))<<(n*p);
        d[32*hp+16*odd+j]=B(v);
    }
}
extern "C" int vtq_pack(const B*c,const B*unc,const B*gain,const B*usb,const int16_t*lut,B*out,
                        int M,int ns,int k,int frac,int h,int st,size_t capacity){
    if(M<1||ns<1||k<2||k>6||frac<0||frac>3||h<1||h>3)return -1;
    const int last_len=frac==0?512:frac==1?256:frac==2?384:128;
    if(st!=512+512*(k-2)+last_len+32*h+16)return -2;
    size_t ng=(size_t(M)+15)/16;
    if(capacity/size_t(st)/size_t(ns)<ng)return -3;
    for(size_t b=0;b<ng;++b)for(int s=0;s<ns;++s){B *tile=out+(b*ns+s)*st,*hdr=tile+st-16-32*h;
        int valid=std::min(16,M-int(16*b));
        for(int r=0;r<16;++r)tile[st-16+r]=r<valid?usb[(b*16+r)*ns+s]:0;
        for(int g=0;g<16;++g){alignas(32) B u[16][16];unsigned gh[3]={};
            for(int r=0;r<16;++r){unsigned a=0;B*cc=tile+32*g+2*r;
                if(r<valid){size_t rb=(b*16+r)*ns+s;std::memcpy(cc,c+32*rb+2*g,2);std::memcpy(u[r],unc+256*rb+16*g,16);a=gain[16*rb+g];}
                else{std::memset(cc,0,2);std::memset(u[r],0,16);}
                for(int z=0;z<h;++z)gh[z]|=((a>>z)&1u)<<r;
            }
            for(int z=0;z<h;++z){hdr[2*(g*h+z)]=B(gh[z]);hdr[2*(g*h+z)+1]=B(gh[z]>>8);}
            B*d=tile+512;
            if(!frac&&k>=4){eo(d+32*g,u,k-2);d+=512;
                if(k==6)subword<4>(d+128*g,u);
                else{subword<2>(d+64*g,u);if(k==5)eo(d+1024+32*g,u,2);}continue;
            }
            for(int z=k-2;z>=1;--z){eo(d+32*g,u,z);d+=512;}
            if(!frac){eo(d+32*g,u,0);continue;}
            if(frac==1){for(int r=0;r<16;++r){int nib[2];for(int half=0;half<2;++half){unsigned cw=0;
                for(int j=0;j<8;++j)cw=(cw<<1)|(u[r][8*half+j]&1u);nib[half]=lut[cw];if(nib[half]<0)return -4;}
                d[16*g+r]=B(nib[0]|(nib[1]<<4));}continue;}
            for(int pair=0;pair<8;++pair){unsigned p[3]={};for(int n=0;n<8;++n){const B*q=u[2*pair+n/4]+4*(n%4);
                if(frac==2){if((q[0]^q[1]^q[2]^q[3])&1)return -5;}
                else if(((q[0]^q[1])|(q[0]^q[2])|(q[0]^q[3]))&1)return -6;
                for(int z=0;z<(frac==2?3:1);++z)p[z]|=(q[z]&1u)<<n;
            }for(int z=0;z<(frac==2?3:1);++z)d[128*z+8*g+pair]=B(p[z]);}
        }
    }return 0;
}
