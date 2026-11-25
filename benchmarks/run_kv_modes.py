"""One representative family per GPU: actual KV4 vs KV16 quality/speed/memory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from run_e2e_suite import CALIBRATIONS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=list(CALIBRATIONS), required=True)
    p.add_argument("--output-root", type=Path, default=Path('benchmarks/kv_modes'))
    p.add_argument("--reuse-existing", action="store_true", help="Explicitly reuse completed outputs when resuming; choose a new output root for a fresh run")
    args = p.parse_args()
    root = args.output_root/args.family
    if root.exists() and any(root.iterdir()) and not args.reuse_existing:
        p.error('Choose a new --output-root, or explicitly pass --reuse-existing to resume')
    root.mkdir(parents=True, exist_ok=True)
    previous = root/'summary.json'
    if not previous.exists():
        previous = root/'diagnostics_scalar_decode/summary.json'
    previous_hashes = json.loads(previous.read_text()).get('source_sha256') if previous.exists() else None
    reused_outputs = []
    cal = Path('benchmarks/cache')/CALIBRATIONS[args.family]
    manifest = json.loads((cal/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['lwc_epochs']==0
    model = manifest['model']
    source = [*Path('duquant_cuda').glob('*.py'), *Path('duquant_cuda').glob('*.cu'),
              *Path('duquant_cuda').glob('*.cuh'), Path(__file__),
              Path('benchmarks/eval_ppl.py'), Path('benchmarks/bench_model.py'),
              Path('benchmarks/verify_kv_decode.py')]
    hashes = {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source}

    def run(script, output, extra):
        if output.exists():
            old=json.loads(output.read_text())
            assert old.get('status','complete')=='complete' and old['model']==model
            if script=='verify_kv_decode.py':
                assert old['status']=='complete' and old['packed_attention_oracle']['passed']
            print('Existing:',output,flush=True)
            reused_outputs.append(str(output))
            return
        output.parent.mkdir(parents=True,exist_ok=True)
        print('Starting:',output,flush=True)
        with output.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable,'benchmarks/'+script,'--model',model,
                            '--output',str(output),*extra],stdout=log,stderr=subprocess.STDOUT,check=True)

    results={}
    for bits in (16,4):
        folder=root/f'kv{bits}'
        base=['--kv-bits',str(bits),'--calibration',str(cal)]
        for mode in ('cuda','integer_reference'):
            run('eval_ppl.py',folder/f'ppl_{mode}.json',['--mode',mode,*base])
        run('verify_kv_decode.py',folder/'decode_reference.json',base)
        for prompt in (128,1024):
            run('bench_model.py',folder/f'model_cuda_p{prompt}_g32.json',
                ['--mode','cuda','--prompt',str(prompt),'--new-tokens','32','--repeats','10',*base])
        a=json.loads((folder/'ppl_cuda.json').read_text())
        b=json.loads((folder/'ppl_integer_reference.json').read_text())
        for key in ('quantization','dataset','scored_tokens','block_count'):
            assert a[key]==b[key],key
        with np.load(a['diagnostic_arrays']) as x,np.load(b['diagnostic_arrays']) as y:
            check=dict(predictions_equal=bool(np.array_equal(x['predictions'],y['predictions'])),
                       sampled_logits_equal=bool(np.array_equal(x['sample_logits'],y['sample_logits'])),
                       max_block_nll_delta=max(abs(c['nll_sum']-d['nll_sum']) for c,d in zip(a['blocks'],b['blocks'])))
        assert check['predictions_equal'] and check['sampled_logits_equal'] and check['max_block_nll_delta']==0,check
        results[str(bits)]=dict(ppl=a['ppl'],integer_reference=check,
                               decode=json.loads((folder/'decode_reference.json').read_text()))
    for prompt in (128,1024):
        run('bench_model.py',root/f'model_fp16_p{prompt}_g32.json',
            ['--mode','fp16','--kv-bits','16','--prompt',str(prompt),'--new-tokens','32','--repeats','10'])
    for bits in (16,4):
        generations=[]
        for prompt in (128,1024):
            fp=json.loads((root/f'model_fp16_p{prompt}_g32.json').read_text())
            qt=json.loads((root/f'kv{bits}'/f'model_cuda_p{prompt}_g32.json').read_text())
            assert fp['visible_devices']==qt['visible_devices'] and fp['gpu']==qt['gpu']
            mem=lambda x:dict(model_gib=x['memory']['model_storage_bytes']/2**30,
                kv_gib=x['memory']['kv_cache_bytes']/2**30,
                decode_workspace_gib=x['memory']['kv_decode_workspace_bytes']/2**30,
                peak_allocated_gib=x['memory']['inference_peak_allocated_bytes']/2**30,
                reserved_gib=x['memory']['inference_reserved_bytes']/2**30,
                device_used_gib=x['memory']['inference']['device_used_bytes']/2**30)
            generations.append(dict(prompt=prompt,output=32,fp16_ms=fp['medians']['wall_ms'],
                cuda_ms=qt['medians']['wall_ms'],speedup=fp['medians']['wall_ms']/qt['medians']['wall_ms'],
                fp16_memory=mem(fp),cuda_memory=mem(qt)))
        results[str(bits)]['generation']=generations
    for f,h in hashes.items():
        assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,'Source changed during evaluation'
    prior=json.loads((Path('benchmarks/families')/args.family/'ppl_cuda.json').read_text())
    assert prior['quantization']['calibration_manifest_sha256']==hashlib.sha256((cal/'manifest.json').read_bytes()).hexdigest()
    result=dict(status='complete',family=args.family,model=model,results=results,
                prior_q4kv4_fake_cache_ppl=prior['ppl'],source_sha256=hashes,
                reused_outputs=reused_outputs, previous_run_source_sha256=previous_hashes,
                calibration_sha256=hashlib.sha256((cal/'manifest.json').read_bytes()).hexdigest())
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(args.family,'COMPLETE',flush=True)


if __name__=='__main__':
    main()
