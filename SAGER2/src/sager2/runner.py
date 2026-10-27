"""Detached multi-GPU replication; each original/edited pair stays on one GPU."""
import os,subprocess,sys,time
from pathlib import Path
from .common import ROOT,read_json,read_jsonl,write_json,digest,code_identity,model_identity
def resolve_gpus(requested):
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    known={}
    for line in data.splitlines():
        idx,uuid=[x.strip() for x in line.split(',')];known[idx]=uuid;known[uuid]=uuid
    ids=[known[x] for x in requested]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Give distinct explicit GPU indices or UUIDs')
    return ids
def launch(cfg,gpus,output,resume=False,no_metrics=False):
    from .check import check
    checked=check(cfg,True);cfg=dict(cfg,output_dir=str(Path(output).resolve()))
    if not no_metrics:
        if not Path(cfg['detoxify_checkpoint']).is_file() or not Path(cfg['detoxify_tokenizer']).is_dir():raise FileNotFoundError('Set local Detoxify checkpoint/tokenizer paths first')
    ids=resolve_gpus(gpus);out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    protocol=dict(schema=1,config=cfg,gpu_uuids=ids,worker_count=len(ids),methods=['base','sager'],target_pairs=1199,no_metrics=no_metrics,
        preflight=checked,model_identity=model_identity(cfg['model_path']),code_sha256=code_identity(),
        decoding='greedy, raw/no-chat, nativeBOS, KV cache, max10 new tokens, repetition penalty1, EOS respected',
        pairing='same model and same physical GPU for original and edited generation',
        ppl_protocol='ARGRE_tail_mask_raw_no_strip_no_response_bos_v2',
        precision='full FP16 language model; FP64 probes/search; original trained benign-mean mu gates',
        scope='One main1199-prompt run. MultiGPU replicas are sharding, not independent repeats. Reorganized source has separate packaging validation.')
    protocol['fingerprint']=digest(protocol)
    target=out/'PROTOCOL.json'
    if target.exists():
        if not resume:raise FileExistsError('Use a new output directory, or explicit --resume')
        if read_json(target)!=protocol:raise ValueError('Resume protocol/code/data/model/GPU assignment differs')
        if (out/'LAUNCH.json').exists():
            pid=read_json(out/'LAUNCH.json')['pid'];proc=Path('/proc')/str(pid)/'cmdline'
            if proc.exists() and str(out).encode() in proc.read_bytes():raise RuntimeError('This run supervisor is already active')
        if (out/'STOP_REQUESTED.json').exists():
            prior=read_json(out/'STOP_REQUESTED.json');write_json(out/('STOP_HISTORY_%d.json'%time.time_ns()),prior);(out/'STOP_REQUESTED.json').unlink()
    else:
        if any(out.iterdir()):raise FileExistsError('Refuse a nonempty unrelated output directory')
        write_json(target,protocol)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
        env[key]='1'
    with (out/'supervisor.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','sager2.supervisor','--run-dir',str(out)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(pid=p.pid,run_dir=str(out),fingerprint=protocol['fingerprint'],gpu_uuids=ids,detached=True,unix_time=time.time())
    write_json(out/'LAUNCH.json',launch);return launch
def summarize(out):
    import math,statistics
    out=Path(out);p=read_json(out/'PROTOCOL.json');pairs=[]
    for f in sorted((out/'pairs').glob('p*.json')):
        x=read_json(f)
        if x['fingerprint']!=p['fingerprint']:raise ValueError('Foreign pair in output directory')
        pairs.append(x)
    ids=[x['index'] for x in pairs]
    if len(set(ids))!=len(ids) or any(not 0<=i<1199 for i in ids):raise ValueError('Invalid source-index coverage')
    methods={}
    for name in p['methods']:
        rr=[x['methods'][name] for x in pairs];loss=[x['ppl_g']['nll'] for x in rr if x.get('ppl_g',{}).get('nll') is not None]
        exact=[x['ppl_g_token_ids']['nll'] for x in rr if x.get('ppl_g_token_ids',{}).get('nll') is not None]
        tox=[x['detoxify']['toxicity'] for x in rr if x.get('detoxify') is not None]
        methods[name]=dict(samples=len(rr),mean_toxicity=statistics.mean(tox) if tox else None,toxicity_scored=len(tox),
            ppl_g=math.exp(statistics.mean(loss)) if loss else None,ppl_valid_samples=len(loss),
            ppl_g_token_ids=math.exp(statistics.mean(exact)) if exact else None,ppl_token_ids_valid_samples=len(exact),
            generated_tokens=sum(x['token_count'] for x in rr),triggers=sum(x.get('triggers',0) for x in rr),rotations=sum(x.get('rotations',0) for x in rr))
    result=dict(complete=len(ids)==1199,completed_pairs=len(ids),target_pairs=1199,fingerprint=p['fingerprint'],methods=methods,
        ppl_protocol=p['ppl_protocol'],independent_runs=1,updated_unix=time.time())
    write_json(out/'summary.json',result);return result
