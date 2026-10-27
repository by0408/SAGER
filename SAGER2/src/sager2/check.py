"""CPU-only preflight; checks real files without loading a language model."""
from collections import Counter
from pathlib import Path
from .common import ROOT,read_json,read_jsonl,sha256,model_identity,parse_layers,require_training_data
def check(cfg,include_model=True):
    data=read_jsonl(require_training_data(cfg));challenge=read_jsonl(cfg['challenge_data'])
    from .data import validate_rows
    audit=validate_rows(data,cfg['challenge_data'])
    if len(challenge)!=1199 or any(set(r)!={'prompt'} for r in challenge):raise ValueError('Expected original1199 Challenge prompts')
    if audit['partition_counts']['train']!=cfg['expected_training_count']:raise ValueError('Wrong model-specific training data')
    root=Path(cfg['probe_dir'])
    if not root.exists():
        if include_model:raise FileNotFoundError('Trained probes are not bundled. Run extract/train, then supply --probe-dir to doctor/evaluate: '+str(root))
        return dict(passed=True,model_key=cfg['model_key'],model_checked=False,probe_checked=False,
            training_rows=len(data),partitions=audit['partition_counts'],challenge_rows=len(challenge),
            training_data_sha256=sha256(cfg['training_data']),challenge_sha256=sha256(cfg['challenge_data']),
            refresh_interval=cfg['editing']['candidate_refresh_interval'],scope='Dataset/config checks only; trained probes are absent by design')
    manifest=read_json(root/'manifest.json');thresholds=read_json(root/'thresholds.json')
    layers=parse_layers(cfg['layers'],cfg['expected_blocks'])
    if not manifest['complete'] or manifest['layers']!=layers or manifest['dimension']!=cfg['expected_dimension']:raise ValueError('Probe manifest mismatch')
    if manifest['training_data_sha256']!=sha256(cfg['training_data']):raise ValueError('Changed probe training dataset')
    if {r['layer'] for r in thresholds}!=set(layers):raise ValueError('Wrong threshold layers')
    for row in thresholds:
        if sha256(root/('block%d.pt'%row['layer']))!=row['checkpoint_sha256']:raise ValueError('Changed checkpoint')
    if include_model:
        if model_identity(cfg['model_path'])!=manifest['model_identity']:raise ValueError('Model identity differs from released probes')
        for name,expected in manifest.get('tokenizer_sha256',{}).items():
            if sha256(Path(cfg['model_path'])/name)!=expected:raise ValueError('Changed tokenizer '+name)
    return dict(passed=True,model_key=cfg['model_key'],model_checked=include_model,training_rows=len(data),partitions=audit['partition_counts'],
        challenge_rows=len(challenge),editing_layers=layers,probe_checkpoints=len(thresholds),training_data_sha256=sha256(cfg['training_data']),
        challenge_sha256=sha256(cfg['challenge_data']),probe_manifest_sha256=sha256(root/'manifest.json'),thresholds_sha256=sha256(root/'thresholds.json'),
        probe_sha256={str(r['layer']):r['checkpoint_sha256'] for r in thresholds},refresh_interval=cfg['editing']['candidate_refresh_interval'])
