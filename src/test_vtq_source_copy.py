"""Check sendfile and bounded fallback, including fallback after a partial send."""
import errno,os,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from src.vtq_format import VtqWriter,VtqFile
from src.vtq_source_copy import FrozenTensorCopier


def test_copy():
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);inp=td/'model.safetensors';tensors={'odd':torch.arange(97,dtype=torch.uint8),'empty':torch.empty(0,dtype=torch.float16),'scalar':torch.tensor(-3,dtype=torch.int32),'big':torch.arange(2000000,dtype=torch.float32).to(torch.bfloat16)};save_file(tensors,str(inp));expected=None
        for mode in ('reference','native','fallback','partial'):
            dst=td/(mode+'.vtq');wr=VtqWriter(dst,[],reserve=4096);real=os.sendfile
            def copy(out_fd,in_fd,offset,count):
                if mode=='partial' and count>9:return real(out_fd,in_fd,offset,count-9)
                if mode in ('partial','fallback'):raise OSError(errno.EINVAL,'forced fallback')
                return real(out_fd,in_fd,offset,count)
            with FrozenTensorCopier() as c,patch('os.sendfile',copy):
                for i,(name,t) in enumerate(tensors.items()):
                    wr.add_tensor(f'prefix{i}',torch.tensor([i],dtype=torch.uint8))
                    if mode=='reference':wr.add_tensor(name,t)
                    else:assert c.add(wr,name,inp)==t.numel()*t.element_size()
            wr.finalize({'quantization':{'formats':[]}});raw=dst.read_bytes()
            if expected is None:expected=raw
            assert raw==expected,mode
            rd=VtqFile(dst)
            try:
                for name,t in tensors.items():assert torch.equal(rd.tensor(name),t)
            finally:rd.mm.close();rd.f.close()
    print('source copy byte identity: regular, bounded fallback and partial-send fallback',flush=True)

if __name__=='__main__':test_copy()
