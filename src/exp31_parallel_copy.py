"""Plan the suffix for a fixed menu/rate, then fill it concurrently with the
matrix region via a distinct file description. Reject any layout disagreement
before writing a valid header. No extra disk-sized staging copy.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from src.streaming_quant import _weight_of
from src.vtq_source_copy import FrozenTensorCopier


def replaced_keys(store,prefix,f,i,n):
    key=f'{prefix}.layers.{i}.{n}'
    if f=='moe30' and n.startswith('mlp.experts.'):
        _,_,e,kind=n.split('.');fused=f'{prefix}.layers.{i}.mlp.experts.{kind}'
        if store.has(fused):return [fused]
        if kind=='gate_up_proj':return [key.rsplit('.',1)[0]+'.'+p+'.weight' for p in ('gate_proj','up_proj')]
    return [key+'.weight']


def begin_copy(st,writer,path,f,store,prefix,layers,fisher,fmt):
    def align(x):return (x+63)//64*64
    skip=set();boundary=0
    for i,n in fisher:
        w=_weight_of(layers[i],n);m,k=w.shape;assert k%256==0;skip.update(replaced_keys(store,prefix,f,i,n));boundary+=sum(map(align,(((m+15)//16)*(k//256)*fmt.st_bytes,5*4,2*m,2*k)))
    assert skip.issubset(store.files);unchanged=sorted(set(store.files)-skip);writer.f.flush();shadow=copy.copy(writer);shadow.tensors={};shadow.pos=boundary
    def work():
        size=0
        with path.open('r+b') as out,FrozenTensorCopier() as c:
            out.seek(writer.data_off+boundary);shadow.f=out
            for name in unchanged:size+=c.add(shadow,name,store.files[name])
        return shadow.pos,shadow.tensors,size
    ex=st.enter_context(ThreadPoolExecutor(max_workers=1));future=ex.submit(work)
    def finish(actual_skip):
        if actual_skip!=skip or align(writer.pos)!=boundary:raise ValueError('planned unchanged region does not match emitted matrices')
        writer._pad(64);writer.f.flush();final_pos,metadata,nbytes=future.result();assert not (writer.tensors.keys() & metadata.keys());writer.tensors.update(metadata);writer.pos=final_pos;writer.f.seek(writer.data_off+final_pos);return len(unchanged),nbytes
    return finish
