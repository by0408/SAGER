from contextlib import contextmanager
import torch
from .editor_base import BatchEditor as CoreBatchEditor, _score
from .models import hidden_tensor, replace_hidden

class BatchEditor(CoreBatchEditor):
    def __init__(self,*args,newline_id=None,skip_enabled=None,**kwargs):
        super().__init__(*args,**kwargs)
        cfg=args[1] if len(args)>1 else kwargs['cfg']
        edit=cfg['editing']
        self.newline_id=int(edit.get('newline_token_id',13) if newline_id is None else newline_id)
        self.skip_enabled=bool(edit.get('skip_newline_edit',True) if skip_enabled is None else skip_enabled)
        self.newline_skip_mask=None
        self.input_history=[]

    def before_forward(self,module,args,kwargs):
        ids=kwargs.get('input_ids')
        if ids is None and args:ids=args[0]
        if ids is None or ids.ndim!=2:raise ValueError('Require native input_ids')
        last=ids[:,-1].detach()
        self.newline_skip_mask=(last==self.newline_id) if self.skip_enabled else torch.zeros_like(last,dtype=torch.bool)
        self.input_history.append((last.clone(),self.newline_skip_mask.clone()))

    @contextmanager
    def active(self,rotation_log=None):
        self.input_history=[]
        handle=self.adapter.model.register_forward_pre_hook(self.before_forward,with_kwargs=True)
        try:
            with super().active(rotation_log=rotation_log):yield self
        finally:handle.remove()


