"""Batch-one early bypass; summary logging deferred until generation ends."""
import torch
from .editor_newline import BatchEditor as Original
from .editor_base import _score
from .models import hidden_tensor,replace_hidden

class BatchEditor(Original):
    def __init__(self,*a,reuse_prepared=False,**kw):
        super().__init__(*a,**kw);self.reuse_prepared=reuse_prepared
    def make_hook(self,layer):
        def hook(module,args,output):
            h=hidden_tensor(output);assert h.shape[0]==1 and h.shape[-1]==self.adapter.dimension and h.device==self.layer_devices[layer]
            self.batch_size=1;assert self.alive_mask is not None and self.alive_mask.shape==(1,)
            step=self.calls.get(layer,0);self.calls[layer]=step+1;alive=self._alive_by_device[str(h.device)]
            before=h[:,-1,:].detach().double();norm=torch.linalg.vector_norm(before,dim=-1)
            invalid=alive&((norm<=0)|~torch.isfinite(norm)|~torch.isfinite(before).all(-1))
            safe=norm.clamp_min(torch.finfo(torch.float64).tiny);z=before/safe[:,None]
            probe=self.probes[layer];threshold=self.thresholds[layer];score=_score(probe,z)
            triggered=alive&~invalid&(score<threshold)&~self.newline_skip_mask.to(h.device)
            # The prior solver already performed a host any().item() check;
            # perform it before allocations, duplicate normalization and score.
            if not bool(triggered.any().item()):
                after=score;relative=torch.zeros_like(norm);info=None;result=output
            else:
                kw=dict(rounds=self.rounds,max_distinct=self.max_distinct,active_mask=triggered,stop_at_threshold=False,return_steps=False)
                if self.reuse_prepared:kw['prepared']=(norm,z,score,triggered)
                edited,info,_=self.searches[layer].rotate(before,threshold,**kw)
                runtime=edited.to(h.dtype);changed=h.clone();changed[:,-1,:]=runtime
                r64=runtime.double();rn=torch.linalg.vector_norm(r64,dim=-1)
                after=_score(probe,r64/rn.clamp_min(torch.finfo(torch.float64).tiny)[:,None]);relative=(rn/safe-1).abs()
                result=replace_hidden(output,changed)
            self._pending.append(dict(step0=step,layer=layer,threshold=threshold,alive=alive.clone(),invalid_hidden=invalid,
                score_before=score,triggered=triggered,score_after=after,runtime_relative_norm_change=relative,rotation_info=info))
            return result
        return hook
    def export_records(self):
        assert not self.hooks
        if self._exported is not None:return self._exported
        history=[(ids.cpu().tolist(),skip.cpu().tolist()) for ids,skip in self.input_history];rows=[]
        for item in self._pending:
            x=self._deferred_cpu(item)
            if not x['alive'][0]:continue
            assert not x['invalid_hidden'][0]
            info=x['rotation_info'];rotation={k:self._row(v,0,1) for k,v in info.items()} if x['triggered'][0] else None
            if rotation:
                assert not rotation.get('coordinate_budget_violated',False)
                if 'final_unit_norm_error' in rotation:assert rotation['final_unit_norm_error']<1e-10
            ids,skip=history[x['step0']]
            rows.append(dict(step0=x['step0'],layer=x['layer'],threshold=x['threshold'],score_before=x['score_before'][0],
                triggered=x['triggered'][0],score_after=x['score_after'][0],runtime_relative_norm_change=x['runtime_relative_norm_change'][0],
                current_input_token_id=ids[0],skipped_due_to_newline=bool(skip[0]),rotation_info=rotation))
        self._exported=[rows];return self._exported
