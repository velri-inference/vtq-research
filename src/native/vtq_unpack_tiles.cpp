// Experimental streaming CPU unpacker for the Python file loader, not a matvec.
#include <algorithm>
#include <cstdint>
#include <cstddef>
using B=uint8_t;
static inline unsigned eo(const B*p,int g,int r,int j){return (p[32*g+16*(r&1)+j]>>(r/2))&1u;}
static inline unsigned lowbit(const B*p,int g,int r,int j,int frac,const B*ham){
    if(!frac)return eo(p,g,r,j);
    if(frac==1){unsigned q=(p[16*g+r]>>(4*(j/8)))&15;return (ham[q]>>(7-(j&7)))&1;}
    const int pos=8*g+r/2,sh=4*(r&1)+j/4;if(frac==3)return (p[pos]>>sh)&1;
    unsigned t=j&3;if(t!=3)return (p[128*t+pos]>>sh)&1;return ((p[pos]^p[128+pos]^p[256+pos])>>sh)&1;
}
extern "C" int vtq_unpack(const B*src,size_t len,float*out,size_t out_len,const float*tab,const float*gain,const B*labels,const B*ham,const float*dr,int m,int ns,int k,int frac,int h,int st){
    if(m<=0||ns<=0||k<2||k>6||frac<0||frac>3||h<1||h>3)return -1;
    if(st!=512+512*(k-2)+(frac==0?512:frac==1?256:frac==2?384:128)+32*h+16)return -2;
    size_t nbg=(size_t(m)+15)/16,K=size_t(ns)*256;
    if(len/st/ns<nbg||out_len/K<size_t(m))return -3;
    for(size_t b=0;b<nbg;++b)for(int s=0;s<ns;++s){const B*t=src+(b*ns+s)*st,*p=t+512,*hdr=t+st-16-32*h;
        for(int g=0;g<16;++g)for(int r=0;r<std::min(16,m-int(16*b));++r){size_t row=16*b+r;float*o=out+row*K+256*s+16*g;const B*cur=t+32*g+2*r,*prev=t+32*g+2*((r+15)&15);
            uint32_t cb=(uint32_t(prev[1])<<24)|(uint32_t(prev[0])<<16)|(uint32_t(cur[1])<<8)|cur[0];int gi=0;
            for(int z=0;z<h;++z)gi|=((hdr[2*(g*h+z)+r/8]>>(r&7))&1)<<z;
            float sc=t[st-16+r]*(1.f/128.f);
            for(int j=0;j<16;++j){unsigned u;
                if(!frac&&k>=4){unsigned hi=eo(p,g,r,j)<<(k-2);const B*v=p+512;
                    if(k==6)u=hi|((v[128*g+32*(r/4)+16*(r&1)+j]>>(4*((r/2)&1)))&15);
                    else{u=hi|((v[64*g+32*(r/8)+16*(r&1)+j]>>(2*((r/2)&3)))&3);if(k==5)u|=eo(v+1024,g,r,j)<<2;}
                }else{const B*v=p;u=0;for(int z=k-2;z>=1;--z,v+=512)u|=eo(v,g,r,j)<<z;u|=lowbit(v,g,r,j,frac,ham);}
                unsigned lb=labels[(cb>>(15-j))&1023];o[j]=((tab[(lb<<(k-1))|u]*gain[gi])*sc)*dr[row];
            }
        }
    }return 0;
}
