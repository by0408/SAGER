"""Portable paths, byte identities and atomic JSON for the main experiment."""
import hashlib,json,os,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.%d.%d.tmp'%(os.getpid(),threading.get_ident()))
    q.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def resolve(path):
    p=Path(os.path.expandvars(os.path.expanduser(str(path))));return p if p.is_absolute() else ROOT/p
def load_config(path):
    cfg=read_json(resolve(path))
    if (ROOT/'local_paths.json').exists():
        local=read_json(ROOT/'local_paths.json')
        cfg['model_path']=local.get('models',{}).get(cfg['model_key'],cfg['model_path'])
        for key in ('detoxify_checkpoint','detoxify_tokenizer'):
            cfg[key]=local.get(key,cfg[key])
    for key in ('model_path','training_data','challenge_data','hidden_dir','probe_dir','output_dir','detoxify_checkpoint','detoxify_tokenizer'):
        if key=='training_data' and cfg.get(key) is None:continue
        cfg[key]=str(resolve(cfg[key]))
    edit=cfg['editing']
    required=dict(rounds=500,max_angle_degrees=5,max_distinct_coordinates=800,top_gradient=2,top_hidden=2,screen_candidates=100,exact_candidates=100,stop_at_threshold=False,skip_newline_edit=True,newline_token_id=13)
    if any(edit[k]!=v for k,v in required.items()):raise ValueError('Main experiment settings differ from the released protocol')
    q=edit['candidate_refresh_interval']
    if q!=20:raise ValueError('All main models use a20-iteration candidate refresh interval')
    if cfg['max_new_tokens']!=10 or cfg['dtype']!='float16':raise ValueError('Main experiment requires full FP16 and max10 new tokens')
    return cfg
def require_training_data(cfg):
    path=cfg.get('training_data')
    if not path:raise FileNotFoundError('Training data is not bundled for '+cfg['model_key']+'. Supply the original dataset with --training-data data/external/original.jsonl')
    if not Path(path).is_file():raise FileNotFoundError('Missing training data: '+str(path))
    expected=cfg.get('expected_training_data_sha256')
    if expected and sha256(path)!=expected:raise ValueError('Training dataset SHA differs from the original '+cfg['model_key']+' partition')
    return path
def model_identity(path):
    """Portable identity: config/index SHA plus shard names/sizes; no full-weight SHA claim."""
    p=Path(path).resolve();files=sorted(set(p.glob('*.safetensors'))|set(p.glob('pytorch_model*.bin')))
    if not files:raise FileNotFoundError('No local language-model weights in '+str(p))
    return dict(config_sha256=sha256(p/'config.json'),weight_files=[dict(name=f.name,bytes=f.stat().st_size) for f in files],
                index_sha256={f.name:sha256(f) for f in sorted(p.glob('*.index.json'))})
def parse_layers(spec,count):
    if isinstance(spec,list):layers=[int(x) for x in spec]
    elif ':' in spec:
        a,b=spec.split(':');layers=list(range(int(a),count+1 if b=='end' else int(b)+1))
    else:layers=[int(x) for x in spec.split(',')]
    if not layers or len(set(layers))!=len(layers) or any(x<1 or x>count for x in layers):raise ValueError('Invalid1-based editing layers')
    return sorted(layers)
def code_identity():
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted((ROOT/'src/sager2').rglob('*.py'))}
