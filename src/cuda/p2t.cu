// Research P2T: radix-4 symbol trellis on PAIRS, not scalar two-step VTQ.
// One branch selects a 2-D vector jointly; predecessor cost is shared by all
// four successors. Two-pass tail biting. All workspace is owned by caller.
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int T=128, G=4, SUB=8, V=16;

// For every (pair, group header, label), nearest one of 16 signed 2-D vectors.
__global__ void p2t_branches(const float* x,const float* book,const float* gain,
                             float* cost,uint8_t* pick,int n,int L,const float* metric) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n*T*G*L) return;
    int l=i%L,g=(i/L)%G,p=i/(L*G);
    float a=x[2*p],b=x[2*p+1],gg=gain[g];
    float aa=1.f,ab=0.f,bb=1.f;
    if(metric){int j=(p&7)*3;aa=metric[j];ab=metric[j+1];bb=metric[j+2];}
    float best=3.4e38f;int idx=0;
    #pragma unroll
    for(int u=0;u<V;u++) {
        float da=a-gg*book[(l*V+u)*2],db=b-gg*book[(l*V+u)*2+1];
        float e=da*da*aa+db*db*bb+2.f*ab*da*db;
        if(e<best) {best=e;idx=u;}
    }
    cost[i]=best;pick[i]=(uint8_t)idx;
}

template<int S> __global__ void p2t_dp(const float* cost,const uint8_t* labels,
                                        const int* start,int* finish,uint8_t* bp,uint8_t* gsel,
                                        int L,int steps,int rot) {
    constexpr int Q=S/4; int n=blockIdx.x,j=threadIdx.x;
    __shared__ float state[G*S];
    __shared__ float red[Q];__shared__ int arg[Q];
    int s0=start?start[n]:-1;
    for(int g=0;g<G;g++) for(int h=0;h<4;h++) {
        int s=4*j+h;state[g*S+s]=(s0<0||s==s0)?0.f:3.4e38f;
    }
    const unsigned l0=labels[4*j], l1=labels[4*j+1], l2=labels[4*j+2], l3=labels[4*j+3];
    __syncthreads();
    for(int t=0;t<steps;t++) {
        float* cur=state; float4 next[G];
        if(t && !(t%SUB)) {
            #pragma unroll
            for(int h=0;h<4;h++) {
                int s=4*j+h;float v=cur[s];int bg=0;
                #pragma unroll
                for(int g=1;g<G;g++) if(cur[g*S+s]<v) {v=cur[g*S+s];bg=g;}
                gsel[((n*(T/SUB)+t/SUB)*S)+s]=(uint8_t)bg;
                #pragma unroll
                for(int g=0;g<G;g++)cur[g*S+s]=v;
            }
            __syncthreads();
        }
        int p=(rot+t)&(T-1);
        #pragma unroll
        for(int g=0;g<G;g++) {
            float v=cur[g*S+j];int bh=0;
            #pragma unroll
            for(int h=1;h<4;h++) {float u=cur[g*S+j+h*Q];if(u<v){v=u;bh=h;}}
            bp[((n*T+t)*G+g)*Q+j]=(uint8_t)bh;
            const float* e=cost+((n*T+p)*G+g)*L;
            next[g]=make_float4(v+e[l0],v+e[l1],v+e[l2],v+e[l3]);
        }
        __syncthreads();
        #pragma unroll
        for(int g=0;g<G;g++) *reinterpret_cast<float4*>(cur+g*S+4*j)=next[g];
        __syncthreads();
    }
    float* last=state;float v=3.4e38f;int ix=0;
    for(int g=0;g<G;g++)for(int h=0;h<4;h++){
        int s=4*j+h;if((s0<0||s==s0)&&last[g*S+s]<v){v=last[g*S+s];ix=g*S+s;}
    }
    red[j]=v;arg[j]=ix;__syncthreads();
    for(int d=Q/2;d;d/=2){if(j<d&&red[j+d]<red[j]){red[j]=red[j+d];arg[j]=arg[j+d];}__syncthreads();}
    if(!j) finish[n]=arg[0];
}

__global__ void p2t_trace(const uint8_t* labels,const int* finish,const uint8_t* bp,const uint8_t* gsel,
                          const uint8_t* pick,int* mid,uint8_t* out,int n,int S,int L,int steps,int rot,
                          const float* book,const float* gains,float* rec) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
    int s=finish[r]%S,g=finish[r]/S;
    for(int t=steps-1;t>=0;t--){
        int p=(rot+t)&(T-1);int j=s>>2;
        if(out){int l=labels[s],u=pick[((r*T+p)*G+g)*L+l];out[r*T+p]=(uint8_t)((g<<6)|(u<<2)|(s&3));
            if(rec){rec[2*(r*T+p)]=gains[g]*book[2*(l*V+u)];rec[2*(r*T+p)+1]=gains[g]*book[2*(l*V+u)+1];}}
        s=j+(S/4)*bp[((r*T+t)*G+g)*(S/4)+j];
        if(mid&&t==steps/2){mid[r]=s;return;}
        if(t && !(t%SUB))g=gsel[(r*(T/SUB)+t/SUB)*S+s];
    }
}

extern "C" int p2t_encode(const float* x,const float* book,const float* gains,const uint8_t* labels,
    float* cost,uint8_t* pick,uint8_t* bp,uint8_t* gsel,int* mid,int* finish,uint8_t* out,
    const float* metric,float* rec,int n,int S,int L,void* stream){
    if(n<=0||n>4096||(S!=256&&S!=1024&&S!=2048)||L<4||L>256)return -1;
    cudaStream_t st=(cudaStream_t)stream;
    p2t_branches<<<(n*T*G*L+255)/256,256,0,st>>>(x,book,gains,cost,pick,n,L,metric);
    for(int pass=0;pass<2;pass++){
        int steps=pass?T:32,rot=pass?64:48;
        if(S==256)p2t_dp<256><<<n,64,0,st>>>(cost,labels,pass?mid:nullptr,finish,bp,gsel,L,steps,rot);
        else if(S==2048)p2t_dp<2048><<<n,512,0,st>>>(cost,labels,pass?mid:nullptr,finish,bp,gsel,L,steps,rot);
        else p2t_dp<1024><<<n,256,0,st>>>(cost,labels,pass?mid:nullptr,finish,bp,gsel,L,steps,rot);
        p2t_trace<<<(n+63)/64,64,0,st>>>(labels,finish,bp,gsel,pick,pass?nullptr:mid,pass?out:nullptr,n,S,L,steps,rot,book,gains,rec);
    }
    return (int)cudaGetLastError();
}
