"""Owns only the workers of one detached main-experiment run."""
import argparse,os,subprocess,sys,time,traceback
from pathlib import Path
from .common import ROOT,read_json,write_json
from .runner import summarize
def main(out):
    p=read_json(out/'PROTOCOL.json');workers=[];handles=[]
    for i,uuid in enumerate(p['gpu_uuids']):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid);log=(out/('worker%d.log'%i)).open('ab');handles.append(log)
        proc=subprocess.Popen([sys.executable,'-u','-m','sager2.worker','--run-dir',str(out),'--worker-index',str(i)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append((i,uuid,proc))
    write_json(out/'WORKERS.json',[dict(worker_index=i,gpu_uuid=u,pid=w.pid) for i,u,w in workers]);exits={}
    try:
        while len(exits)<len(workers):
            for i,u,w in workers:
                code=w.poll()
                if code is not None and i not in exits:
                    exits[i]=code;write_json(out/('WORKER%d_EXIT.json'%i),dict(pid=w.pid,actual_exit_code=code,gpu_uuid=u,unix_time=time.time()))
                    if code and not (out/'STOP_REQUESTED.json').exists():write_json(out/'STOP_REQUESTED.json',dict(reason='Worker failure; siblings stop cooperatively',worker=i))
            summarize(out)
            if len(exits)<len(workers):time.sleep(5)
        result=summarize(out);unchanged=p['code_sha256']==__import__('sager2.common',fromlist=['code_identity']).code_identity()
        write_json(out/'FINAL_STATUS.json',dict(complete=result['complete'] and all(c==0 for c in exits.values()) and unchanged,
            worker_exit_codes=exits,source_SHA_unchanged=unchanged,fingerprint=p['fingerprint'],unix_time=time.time()))
    finally:
        for f in handles:f.close()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    try:main(args.run_dir)
    except BaseException as e:write_json(args.run_dir/'SUPERVISOR_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
