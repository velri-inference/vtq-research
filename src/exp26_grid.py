"""Run one family of the all-model gate on G4, stopping before operator deadline.
Separate worker processes for four RSS peaks, using this model's largest observed
actual packed matrix. Keep each completed family as a separate recoverable file.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('family',choices=['gpt2','qwen','llama1','llama3','llama8','qwen8','qwen35','gemma4','qwen27','moe30']);a=ap.parse_args();f=a.family;root=Path('/content/trellis-night');dst=Path(f'results/exp26_{f}_materialize.json');sample=root/'exp26_samples'/f;inv=json.loads((root/'results/exp25_model_inventory.json').read_text());model=inv[f]['path'];env=dict(os.environ,VTQ_LOADER_AB_SAMPLE=str(sample));key='v2:VTQ_300'
    def left():
        n=int(float((root/'universal_deadline').read_text())-time.time())-120
        if n<20:raise TimeoutError('budget reserved for artifacts and shutdown')
        return n
    if not dst.exists():
        if f in ('gpt2','qwen'):mod='src.exp25_writer_ab';args=['--specs','VTQ_300','--seeds','14000']
        elif f in ('qwen35','gemma4','qwen27'):
            mod='src.exp21_hybrid_skip';data='data/qwen38' if f=='qwen27' else f'data/{f}';args=['--writer-ab','--data',data,'--offset','14000']
        elif f=='moe30':mod='src.exp25_moe';args=['--data','data/qwen3']
        else:
            mod='src.exp25_dense';data='data/qwen3' if f=='qwen8' else 'data';fp={'llama1':'fisher_1b.pt','llama3':'fisher_3b.pt','llama8':'fisher_llama8b.pt','qwen8':'fisher.pt'}[f];args=['--data',data,'--fisher',fp]
        subprocess.run([sys.executable,'-u','-m',mod,'--model',model,'--output',str(dst),*args],env=env,timeout=left(),check=True)
    r=json.loads(dst.read_text());w=r['writer'];assert w['loader']['equal_fp32_hashes']==w['matrices'] and w['all_component_files_equal'];rss={}
    for op in ('pack','unpack'):
        rss[op]={}
        for mode in ('numpy','native'):
            p=subprocess.run([sys.executable,'-m','src.exp26_loader_ab','--sample',str(sample),'--action',op,'--mode',mode],capture_output=True,text=True,timeout=left());
            if p.returncode:raise RuntimeError(p.stderr[-2500:])
            rss[op][mode]=json.loads(p.stdout.splitlines()[-1]);
        assert rss[op]['numpy']['sha']==rss[op]['native']['sha']
    r['isolated_cpu_rss_largest_matrix']=rss;r['note']='peak GPU comparisons apply to individual synchronized loader calls only; isolated CPU peak is for the largest observed matrix, not a whole-process inference peak.';tmp=dst.with_suffix('.tmp');tmp.write_text(json.dumps(r)+'\n');tmp.replace(dst);print(f,w['writer_s'],w['loader'],{op:{m:x['extra_peak'] for m,x in modes.items()} for op,modes in rss.items()},flush=True)


if __name__=='__main__':main()
