"""Predeclared train-only folding-scale/clipping reconstruction experiment."""
from duquant_reconstruct import *
from search_ptq import sha
import argparse,time,random,numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=3);args=p.parse_args()
    target=BASE/'duquant_reconstruct_training';target.mkdir(exist_ok=True);TRIAL_REPORT.mkdir(exist_ok=True)
    manifest_path=target/f'manifest_shard{args.shard}.json'
    if manifest_path.exists():raise FileExistsError('Existing experiment is preserved')
    torch.manual_seed(0);random.seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False
    tokens=torch.load(BASE/'train.pt',weights_only=True).cuda()
    with torch.no_grad():hidden=F.embedding(tokens,extras()['embedding'].cuda())
    parent=BASE/'duquant/manifest.json';parent_meta=json.loads(parent.read_text());assert parent_meta['status']=='complete'
    manifest=dict(status='running',experiment='DuQuant frozen rotations + layerwise learned folding scales and scalar clipping',family='qwen3',
        epochs=15,batch_size=4,scale_lr=.005,clipping_lr=.05,optimizer='AdamW default weight decay; cosine decay to 5e-6; grad norm clip 10',
        scale_range='exp(clamp(log_scale,-2,2)); GQA O scales tied per shared V head',weight_clip_range=[.2,1.],activation_kv_clip_range=[.4,1.],
        training=json.loads((BASE/'train.json').read_text()),parent_manifest_sha256=sha(parent),fixed_rotations=True,fixed_permutation=True,
        target='Original frozen FP16 layer output using original FP16 prefixes',selection='Final epoch, no evaluation-based selection',
        source_sha256={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/duquant_reconstruct.py'),Path('duquant_cuda/lwc.py'),Path('duquant_cuda/calibration.py')]},layers=[])
    logs=[]
    for i in range(28):
        with torch.no_grad():
            w=load_weights(i);fp=QwenBlock(w);cache=fp.cache(1,2048);expected=torch.empty_like(hidden)
            for j in range(128):expected[j:j+1]=fp(hidden[j:j+1],cache)
            del w,fp,cache
        if i%args.shards!=args.shard:hidden=expected;continue
        torch.manual_seed(i);print('Reconstruction layer',i+1,flush=True);start=time.perf_counter()
        filename=BASE/'duquant'/f'layer_{i:02d}.pt';assert sha(filename)==parent_meta['layers'][i]['sha256']
        state=torch.load(filename,weights_only=True,map_location='cuda');block=Reconstruction(state)
        opt=torch.optim.AdamW([dict(params=list(block.scale_logs.parameters()),lr=.005),
            dict(params=[*block.wclips.parameters(),*block.aclips.parameters(),block.kvclips],lr=.05)])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=15*32,eta_min=5e-6);losses=[]
        for epoch in range(15):
            total=0
            for j in range(32):
                opt.zero_grad()
                with torch.autocast('cuda',dtype=torch.float16):
                    actual=block(hidden[j*4:j*4+4]);loss=(actual.float()-expected[j*4:j*4+4].float()).square().mean();objective=loss/loss.detach()
                if not torch.isfinite(loss):raise FloatingPointError((i,epoch,j,'loss'))
                objective.backward()
                torch.nn.utils.clip_grad_norm_(block.parameters(),10.,error_if_nonfinite=True)
                if j==0:
                    for name,param in block.named_parameters():
                        if param.grad is None or not torch.isfinite(param.grad).all():raise FloatingPointError((i,epoch,j,name))
                opt.step();scheduler.step();total+=loss.item()
            losses.append(total/32);print(' epoch',epoch+1,'MSE',losses[-1],'seconds',time.perf_counter()-start,flush=True)
        parameters=block.export();out=target/f'layer_{i:02d}.pt';torch.save(parameters,out)
        manifest['layers'].append(dict(index=i,file=out.name,sha256=sha(out)))
        logs.append(dict(layer=i,seconds=time.perf_counter()-start,epochs_mse=losses))
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');(TRIAL_REPORT/f'training_shard{args.shard}.json').write_text(json.dumps(logs,indent=2)+'\n')
        hidden=expected
        del state,block,opt,scheduler,loss,objective,actual;torch.cuda.empty_cache()
    manifest.update(status='complete',shard=args.shard,shards=args.shards);manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
