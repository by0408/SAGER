"""Main Challenge original/SAGER paired generation and scoring, batch one."""
import argparse,functools,gzip,json,os,time,traceback
from pathlib import Path
from .common import read_json,read_jsonl,write_json,sha256,code_identity
def main(out,index):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    p=read_json(out/'PROTOCOL.json');cfg=p['config'];uuid=p['gpu_uuids'][index]
    assert os.environ['CUDA_VISIBLE_DEVICES']==uuid and torch.cuda.device_count()==1
    if torch.cuda.mem_get_info()[0]<cfg['minimum_free_MiB']*1024**2:raise RuntimeError('Insufficient free VRAM; refusing quantization, offload or interference with other jobs')
    if code_identity()!=p['code_sha256']:raise ValueError('Changed source files')
    torch.manual_seed(cfg['seed']);os.environ['TORCHINDUCTOR_CACHE_DIR']=str(out/('compiler_cache_%d'%index))
    from .models import load_model
    from .metrics import ToxicityScorer,score_ppl,score_ppl_token_ids,repetition
    from .decode import generate_batch
    from .backends.probe import GPUProbe
    from .backends.window_solver import GPUSearch
    from .editor import BatchEditor
    model,tok,adapter,identity=load_model(cfg)
    assert model.dtype==torch.float16 and adapter.dimension==cfg['expected_dimension'] and len(adapter.blocks)==cfg['expected_blocks']
    assert all(param.device==torch.device('cuda:0') for param in model.parameters())
    assert tok.bos_token_id==1 and tok.eos_token_id==2
    assert cfg['editing']['candidate_refresh_interval']==20
    factory=functools.partial(GPUSearch,mode='layout_common_no_poll')
    editor=BatchEditor(adapter,cfg,identity,backend='compile_graph',solver_factory=factory,
        probe_factory=GPUProbe.from_state,reuse_prepared=True)
    scorer=None if p['no_metrics'] else ToxicityScorer(cfg)
    prompts=read_jsonl(cfg['challenge_data']);assigned=[i for i in range(1199) if i%p['worker_count']==index]
    write_json(out/('RUNTIME%d.json'%index),dict(pid=os.getpid(),sid=os.getsid(0),gpu_uuid=uuid,gpu_name=torch.cuda.get_device_name(0),
        torch=torch.__version__,threads=1,interop=1,adapter=adapter.describe(),refresh_interval=cfg['editing']['candidate_refresh_interval'],assigned_count=len(assigned)))
    for i in assigned:
        if (out/'STOP_REQUESTED.json').exists():break
        target=out/'pairs'/('p%04d.json'%i)
        if target.exists():
            prior=read_json(target)
            if prior['fingerprint']!=p['fingerprint'] or prior['gpu_uuid']!=uuid:raise ValueError('Foreign/moved pair')
            continue
        prompt=prompts[i]['prompt'];ids=tok(prompt)['input_ids']
        if len(ids)>cfg['max_input_tokens'] or len(ids)+10>model.config.max_position_embeddings:raise ValueError('Context exceeds protocol; no silent truncation')
        methods={}
        for name,ed in [('base',None),('sager',editor)]:
            write_json(out/('STATUS%d.json'%index),dict(stage='generating',index=i,method=name,target_pairs=len(assigned),unix_time=time.time()))
            result=generate_batch(model,tok,[prompt],editor=ed,adapter=adapter,max_new_tokens=10)
            row=result['rows'][0];hist=row.pop('layers');row['prompt_ids']=ids
            if ed:
                assert len(hist)==len(editor.layers)*row['token_count']
                for h in hist:
                    if h['triggered']:
                        b=result['backend_status']['layers'][str(h['layer'])]
                        if b['active']!='compile_graph' or b['error'] is not None:raise RuntimeError('Requested compiled graph backend did not actually run')
                        if h['rotation_info']['rounds']>500 or h['rotation_info']['distinct_coordinates']>800:raise ArithmeticError('Budget violation')
            hpath=out/'diagnostics'/('p%04d_%s.json.gz'%(i,name));hpath.parent.mkdir(exist_ok=True)
            tmp=hpath.with_suffix('.tmp');tmp.write_bytes(gzip.compress(json.dumps(hist,allow_nan=False).encode()));tmp.replace(hpath)
            row.update(triggers=sum(h['triggered'] for h in hist),rotations=sum(h['rotation_info']['rounds'] for h in hist if h['triggered']),
                history_file=hpath.relative_to(out).as_posix(),history_sha256=sha256(hpath),repeated_trigram_fraction=repetition(row['new_token_ids']),generation_timing=result['timing'])
            if scorer:
                row['detoxify']=scorer.score(row['continuation']);row['ppl_g']=score_ppl(prompt,row['continuation'],model,tok)
                row['ppl_g_token_ids']=score_ppl_token_ids(ids,row['new_token_ids'],model)
            methods[name]=row
        write_json(target,dict(index=i,prompt=prompt,fingerprint=p['fingerprint'],gpu_uuid=uuid,methods=methods))
    if code_identity()!=p['code_sha256']:raise ValueError('Source changed during main experiment')
    write_json(out/('STATUS%d.json'%index),dict(stage='stopped' if (out/'STOP_REQUESTED.json').exists() else 'complete',unix_time=time.time()))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--worker-index',type=int,required=True);args=parser.parse_args()
    try:
        import torch
        with torch.inference_mode():main(args.run_dir,args.worker_index)
    except BaseException as e:write_json(args.run_dir/('FAILURE%d.json'%args.worker_index),dict(error=repr(e),traceback=traceback.format_exc()));raise
