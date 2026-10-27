"""The public entry point contains only main-experiment operations."""
import argparse,json,os,time
from pathlib import Path
from .common import ROOT,load_config,resolve,read_json,write_json,require_training_data
def main():
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    parser=argparse.ArgumentParser(prog='sager2',description='SAGER main Challenge experiment: preparation, paired generation and evaluation')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('doctor','extract','train','evaluate'):
        p=sub.add_parser(command);p.add_argument('--config',required=True);p.add_argument('--model-path')
        p.add_argument('--training-data',help='Explicit original training dataset; required for13B/30B whose data are not bundled')
        if command in ('doctor','evaluate'):p.add_argument('--probe-dir',help='Directory produced by the train command; trained probes are not bundled')
        if command=='doctor':p.add_argument('--data-only',action='store_true',help='Check data/config without model weights; also check probes if present')
        if command=='extract':p.add_argument('--hidden-dir',required=True,help='New output directory for hidden caches')
        if command=='train':
            p.add_argument('--hidden-dir',required=True);p.add_argument('--probe-dir',required=True,help='New probe output directory; bundled probes are never overwritten')
        if command=='evaluate':
            p.add_argument('--gpus',nargs='+',required=True,help='Explicit physical GPU indices or UUIDs; full model replica per GPU')
            p.add_argument('--output-dir');p.add_argument('--resume',action='store_true');p.add_argument('--no-metrics',action='store_true',help='Save raw paired continuations and diagnostics only')
    for command in ('status','stop'):
        p=sub.add_parser(command);p.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    if args.command in ('status','stop'):
        out=resolve(args.run_dir)
        if not (out/'PROTOCOL.json').is_file():raise FileNotFoundError('Not a SAGER2 run directory')
        if args.command=='stop':
            write_json(out/'STOP_REQUESTED.json',dict(reason='User cooperative stop',unix_time=time.time()));print('Stop requested; each worker finishes its current paired text.');return
        names=['summary.json','FINAL_STATUS.json','WORKERS.json']
        print(json.dumps({n:read_json(out/n) for n in names if (out/n).exists()},ensure_ascii=False,indent=2));return
    cfg=load_config(args.config)
    if args.model_path:cfg['model_path']=str(resolve(args.model_path))
    if args.training_data:cfg['training_data']=str(resolve(args.training_data))
    if getattr(args,'probe_dir',None):cfg['probe_dir']=str(resolve(args.probe_dir))
    if args.command=='doctor':
        from .check import check
        print(json.dumps(check(cfg,not args.data_only),ensure_ascii=False,indent=2));return
    if args.command=='evaluate':
        from .runner import launch
        print(json.dumps(launch(cfg,args.gpus,args.output_dir or cfg['output_dir'],args.resume,args.no_metrics),ensure_ascii=False,indent=2));return
    require_training_data(cfg)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg['hidden_dir']=str(resolve(args.hidden_dir))
    if args.command=='extract':
        from .extract import extract
        extract(cfg)
    elif args.command=='train':
        cfg['probe_dir']=str(resolve(args.probe_dir))
        if Path(cfg['probe_dir']).is_relative_to(ROOT/'checkpoints'):raise ValueError('Train into a new directory outside bundled checkpoints/')
        from .train import train
        train(cfg)
if __name__=='__main__':main()
