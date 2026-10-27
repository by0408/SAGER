"""Top100 window refreshed per20 slots; rotations use separate angle/apply kernels."""
import math, threading
import torch
from .probe import GPUProbe
from .window_ops import window_twenty as frozen_window

_COMPILED={};_LOCK=threading.RLock()
TRACE_FIELDS=('i','j','phi','gain','score_before','score_after','performed','slot_eligible')

def _ten(z,seen,used,running,budget,counts,attempted,refreshes,W,b,v,c,dims,phi,check_budget,trace,WT,score,common,single):
    p=z@W.T+b;g=(v*(p>0))@W
    raw=torch.cat((g.abs().topk(2,dim=1).indices,z.abs().topk(2,dim=1).indices),1).sort(1).values
    unique=torch.cat((torch.ones_like(raw[:,:1],dtype=torch.bool),raw[:,1:]!=raw[:,:-1]),1)
    anchors=torch.where(unique,raw,z.shape[1]).sort(1).values;valid_anchor=anchors<z.shape[1];k=anchors.shape[1]
    flags=(anchors[:,:,None]==dims[None,None,:]).any(1)
    aa=anchors.clamp_max(z.shape[1]-1)[:,:,None].expand(1,k,z.shape[1]);bb=dims[None,None,:].expand(1,k,z.shape[1])
    valid_pair=valid_anchor[:,:,None]&(aa!=bb)&(~flags[:,None,:]|(aa<bb))
    ii,jj=torch.minimum(aa,bb).reshape(1,-1),torch.maximum(aa,bb).reshape(1,-1)
    BB=z.gather(1,ii)*g.gather(1,jj)-z.gather(1,jj)*g.gather(1,ii)
    eligible=valid_pair.reshape(1,-1)
    if check_budget:
        added=(~seen.gather(1,ii)).long()+(~seen.gather(1,jj)).long()
        eligible=eligible&(added<=(budget-used)[:,None])
    ranks,ids=BB.abs().masked_fill(~eligible,-torch.inf).topk(100,dim=1)
    return frozen_window(z,seen,used,running,counts,attempted,refreshes,WT,v,c,p,ii.gather(1,ids),jj.gather(1,ids),ranks,budget,phi,check_budget,trace,score,common,single)

def _chunk(*args):
    return _ten(*args)

class GPUSearch:
    def __init__(self,probe,device=None,top=2,screen=100,exact=100,max_angle_degrees=5,backend='compile_graph',incremental=True,mode='layout_common_no_poll'):
        self.probe=probe;self.device=probe.W.device;self.d=probe.W.shape[1]
        assert self.d in (4096,5120,6656) and (probe.W.shape[0],top,screen,exact)==(32,2,100,100) and backend=='compile_graph'
        assert incremental
        self.phi=math.radians(max_angle_degrees);assert self.phi==math.radians(5)
        assert mode == 'layout_common_no_poll';self.mode=mode;self.common=True;self.single=False;self.no_poll=True;self.WT=probe.W.T.contiguous()
        self.dims=torch.arange(self.d,device=self.device);self.workspaces={}
        self._status=dict(requested=backend,active=None,error=None,precision='FP64',incremental=True,telemetry='summary_only',graph_slots=20,multi_rotation_kernel_fusion=False,window_candidates=100,selection="exact_gain_max_each_step",graph_replays_total=0)
    def backend_status(self):return dict(self._status)
    def _workspace(self,trace=False,check_budget=False):
        key=(trace,check_budget)
        if key in self.workspaces:return self.workspaces[key]
        p=self.probe;z=torch.zeros((1,self.d),device=self.device,dtype=torch.float64);seen=torch.zeros_like(z,dtype=torch.bool)
        used=torch.zeros(1,device=self.device,dtype=torch.int64);running=torch.zeros(1,device=self.device,dtype=torch.bool)
        counts=used.clone();attempted=used.clone();refreshes=used.clone();budget=torch.full((),800,device=self.device,dtype=torch.int64);score=torch.zeros((),device=self.device,dtype=torch.float64)
        args=(z,seen,used,running,budget,counts,attempted,refreshes,p.W,p.b,p.v,p.c,self.dims,self.phi,check_budget,trace,self.WT,score,self.common,self.single)
        ckey=(self.d,self.common,self.single,trace,check_budget)
        if ckey not in _COMPILED:_COMPILED[ckey]=torch.compile(_chunk,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        fn=_COMPILED[ckey];stream=torch.cuda.Stream(device=self.device);stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):rows=fn(*args)
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):rows=fn(*args)
        ws=dict(score=score,z=z,seen=seen,used=used,running=running,budget=budget,counts=counts,attempted=attempted,refreshes=refreshes,graph=graph,rows=rows)
        self.workspaces[key]=ws;return ws
    def _graph_workspace(self,batch):
        assert batch==1;return self._workspace(False,False)
    @torch.inference_mode()
    def rotate(self,h,threshold,rounds=500,max_distinct=800,active_mask=None,stop_at_threshold=False,return_steps=False,prepared=None):
        assert rounds>=0 and rounds%20==0 and not stop_at_threshold
        h=torch.as_tensor(h,device=self.device,dtype=torch.float64)
        if h.ndim==1:h=h[None,:]
        assert h.shape==(1,self.d)
        if prepared is None:
            norm=torch.linalg.vector_norm(h,dim=1);valid=torch.isfinite(h).all(1)&torch.isfinite(norm)&(norm>0)
            z0=torch.where(valid[:,None],h/norm.clamp_min(1e-300)[:,None],torch.zeros_like(h));initial=self.probe.score(z0)
            active=valid&(initial<threshold)
            if active_mask is not None:active&=active_mask
        else:norm,z0,initial,active=prepared
        if rounds==0 or (prepared is None and not bool(active.any().item())):
            zero=torch.zeros(1,device=self.device,dtype=torch.int64)
            return h.clone(),dict(rounds=zero,attempted_slots=zero,candidate_refreshes=zero,distinct_coordinates=zero,initial_score=initial,exact_final_score=initial,telemetry='summary_only'),{}
        # Each frozen Top100 window uses at most four anchors plus100 partners; keep the800-coordinate guard.
        coordinate_upper_bound=(rounds//20)*104
        check_budget=max_distinct<coordinate_upper_bound
        with _LOCK:
            ws=self._workspace(return_steps,check_budget);ws['z'].copy_(z0);ws['seen'].zero_();ws['used'].zero_();ws['running'].copy_(active)
            ws['budget'].fill_(max_distinct);ws['counts'].zero_();ws['attempted'].zero_();ws['refreshes'].zero_();hist=[]
            for start in range(0,rounds,20):
                if not self.no_poll and start and not bool(ws['running'].any().item()):break
                ws['graph'].replay();self._status['graph_replays_total']+=1
                if return_steps:hist.append(ws['rows'].clone())
            z=ws['z'];out=torch.where(active[:,None],z*norm[:,None],h).clone();final=self.probe.score(z).clone();distinct=ws['seen'].sum(1).clone()
            info=dict(rounds=ws['counts'].clone(),attempted_slots=ws['attempted'].clone(),candidate_refreshes=ws['refreshes'].clone(),
                distinct_coordinates=distinct,initial_score=initial,exact_final_score=final,final_unit_norm_error=(torch.linalg.vector_norm(z,dim=1)-1).abs(),
                coordinate_budget_violated=distinct>max_distinct,telemetry='summary_only; per-step verification and candidate statistics omitted from timed generation')
            self._status.update(active='compile_graph',error=None,mode=self.mode,host_stop_reads=(0 if self.no_poll else max(0,(start//20))),coordinate_upper_bound=coordinate_upper_bound,budget_checks_enabled=check_budget)
            if return_steps:
                rows=torch.cat(hist);steps={n:rows[:,:,k] for k,n in enumerate(TRACE_FIELDS)}
            else:steps={}
            return out,info,steps
