"""Paired real .vtq writer timing. The encoder runs once; each encoder result
is serialized twice with counterbalanced order. Report measured writer time
separately from common encoder time, not as two timed full-process runs.
"""
import argparse,hashlib,json,time,tempfile,os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import torch
from src import vtq_format as vf
from src.exp25_pack import NativePack
from src.exp09_group_order import run_model_ab


class PairedWriter:
    def __init__(self,lib):
        self.native=NativePack(lib);self.base=vf.pack_supertiles_vtq;self.rows=[];self.loader=None
        if os.environ.get('VTQ_LOADER_AB_SAMPLE'):
            from src.exp26_loader_ab import LoaderAB
            self.loader=LoaderAB(os.environ['VTQ_LOADER_AB_SAMPLE'])
    def __call__(self,path,p,fmt,method):
        i=len(self.rows);row=dict(format=fmt.name,shape=list(p['u_all'].shape[:2]),times={},bytes={},sha={},order=['numpy','native'] if i%2==0 else ['native','numpy']);paths={}
        for tag in row['order']:
            f=path if tag=='native' else path.with_name('reference_numpy.vtq');paths[tag]=f
            with patch.object(vf,'pack_supertiles_vtq',self.native if tag=='native' else self.base):
                torch.cuda.synchronize();t=time.perf_counter();wr=vf.VtqWriter(f,[fmt],reserve=4096);wr.add_matrix('w',p,verify=False);wr.finalize({'quantization':{'formats':[vf.format_json_vtq(fmt)]}});torch.cuda.synchronize();row['times'][tag]=time.perf_counter()-t
            with f.open('rb') as stream:row['sha'][tag]=hashlib.file_digest(stream,'sha256').hexdigest()
            row['bytes'][tag]=f.stat().st_size
        assert row['sha']['native']==row['sha']['numpy'] and row['bytes']['native']==row['bytes']['numpy'],(i,fmt.name)
        self.rows.append(row)
        if self.loader is not None:
            out,row['loader']=self.loader.check(paths['native'],p,fmt,i);return out
        rd=vf.VtqFile(paths['native'])
        try:return rd.decode_matrix('w',device='cuda')
        finally:rd.mm.close();rd.f.close()
    def report(self):
        times={t:sum(x['times'][t] for x in self.rows) for t in ('numpy','native')};r=dict(matrices=len(self.rows),all_component_files_equal=True,writer_s=times,writer_ratio=times['native']/times['numpy'],rows=self.rows)
        if self.loader:
            r['loader']=dict(materialization_s={t:sum(x['loader']['times'][t] for x in self.rows) for t in times},max_extra_gpu_bytes={t:max(x['loader']['extra_gpu_bytes'][t] for x in self.rows) for t in times},equal_fp32_hashes=len(self.rows))
        return r


@contextmanager
def unchanged(method):yield


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--lib',default='/content/libexp25pack.so');ap.add_argument('--text',type=Path,default=Path('data/wikitext2_test.raw'));ap.add_argument('--specs',nargs='+',default=['VTQ_300']);ap.add_argument('--seeds',type=int,nargs='+',default=[13000]);a=ap.parse_args();a.mode='confirm';a.winner='natural';pw=PairedWriter(a.lib);dest=a.output
    with tempfile.TemporaryDirectory() as td:
        a.output=Path(td)/'ab.json';run_model_ab(a,['natural'],unchanged,roundtrip=pw);r=json.loads(a.output.read_text())
    r['writer']=pw.report();r['protocol']='one quantization, paired writes, ref/candidate checksums and decode from candidate';dest.write_text(json.dumps(r)+'\n');print('writer times',r['writer']['writer_s'],flush=True)


if __name__=='__main__':main()
