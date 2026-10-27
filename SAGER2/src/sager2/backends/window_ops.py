"""One rotation at a time. Layout/common-value experiments; no cross-rotation fusion."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.library import triton_op,wrap_triton
from .angle_body import _choose

@triton.jit
def _angles(Z,Seen,Used,Running,W,V,Cp,Pre,Ip,Jp,Ranks,Budget,Score,Angles,Gains,Allowed,
            FIRST:tl.constexpr,COMMON:tl.constexpr,CHECK:tl.constexpr,PHI:tl.constexpr):
 k=tl.program_id(0);q=tl.arange(0,32);i=tl.load(Ip+k);j=tl.load(Jp+k);rank=tl.load(Ranks+k);running=tl.load(Running)
 allowed=running&(rank==rank)&(tl.abs(rank)<float('inf'))
 if CHECK:
  add=(1-tl.load(Seen+i).to(tl.int64))+(1-tl.load(Seen+j).to(tl.int64));allowed=allowed&(add<=tl.load(Budget)-tl.load(Used))
 angle=tl.full((),0.,tl.float64);gain=tl.full((),-float('inf'),tl.float64)
 if allowed:
  zi=tl.load(Z+i);zj=tl.load(Z+j);wi=tl.load(W+i*32+q);wj=tl.load(W+j*32+q);p=tl.load(Pre+q);v=tl.load(V+q);c=tl.load(Cp)
  if COMMON and not FIRST:before=tl.load(Score)
  else:before=tl.sum(tl.maximum(p,0)*v,0)+c
  A=wi*zi+wj*zj;B=wj*zi-wi*zj;C=p-A
  angle=_choose(A,B,C,p,v,c,before,32,PHI)
  gain=tl.sum(tl.maximum(C+A*libdevice.cos(angle)+B*libdevice.sin(angle),0)*v,0)+c-before
  gain=tl.where((gain==gain)&(tl.abs(gain)<float('inf')),gain,-float('inf'))
 tl.store(Angles+k,angle);tl.store(Gains+k,gain);tl.store(Allowed+k,allowed)

@triton.jit
def _apply(Z,Seen,Used,Running,Counts,Attempts,Refreshes,W,V,Cp,Pre,Ip,Jp,Angles,Gains,Allowed,Score,Rows,
           FIRST:tl.constexpr,COMMON:tl.constexpr,CHECK:tl.constexpr,TRACE:tl.constexpr):
 k=tl.arange(0,128);gains=tl.load(Gains+k,k<100,-float('inf'));best=tl.max(gains,0);index=tl.min(tl.where((k<100)&(gains==best),k,2147483647),0)
 i=tl.load(Ip+index);j=tl.load(Jp+index);eligible=tl.load(Allowed+index);running=tl.load(Running);angle=tl.load(Angles+index);gain=tl.where(eligible,best,0.)
 perform=running&eligible&(best>1.e-9);zi=tl.load(Z+i);zj=tl.load(Z+j);co=libdevice.cos(angle);si=libdevice.sin(angle)
 ni=tl.where(perform,zi*co-zj*si,zi);nj=tl.where(perform,zi*si+zj*co,zj)
 q=tl.arange(0,32);p=tl.load(Pre+q);v=tl.load(V+q);c=tl.load(Cp);before=tl.sum(tl.maximum(p,0)*v,0)+c
 wi=tl.load(W+i*32+q);wj=tl.load(W+j*32+q);p=p+wi*(ni-zi)+wj*(nj-zj)
 if CHECK:
  added=(1-tl.load(Seen+i).to(tl.int64))+(1-tl.load(Seen+j).to(tl.int64));tl.store(Used,tl.load(Used)+tl.where(perform,added,0))
 tl.store(Z+i,ni);tl.store(Z+j,nj);tl.store(Seen+i,True,perform);tl.store(Seen+j,True,perform);tl.store(Pre+q,p)
 tl.store(Counts,tl.load(Counts)+perform.to(tl.int64));tl.store(Attempts,tl.load(Attempts)+running.to(tl.int64))
 if FIRST:tl.store(Refreshes,tl.load(Refreshes)+running.to(tl.int64))
 tl.store(Running,running&perform)
 if COMMON or TRACE:after=tl.sum(tl.maximum(p,0)*v,0)+c
 if COMMON:tl.store(Score,after)
 if TRACE:
  tl.store(Rows+0,i.to(tl.float64));tl.store(Rows+1,j.to(tl.float64));tl.store(Rows+2,angle);tl.store(Rows+3,gain)
  tl.store(Rows+4,before);tl.store(Rows+5,after);tl.store(Rows+6,perform.to(tl.float64));tl.store(Rows+7,eligible.to(tl.float64))

@triton_op('sager_roundopt_v1::angles',mutates_args={})
def angles(z:torch.Tensor,seen:torch.Tensor,used:torch.Tensor,running:torch.Tensor,W:torch.Tensor,v:torch.Tensor,c:torch.Tensor,p:torch.Tensor,ii:torch.Tensor,jj:torch.Tensor,ranks:torch.Tensor,budget:torch.Tensor,score:torch.Tensor,first:bool,common:bool,phi:float,check:bool)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
 a=torch.empty((100,),device=z.device,dtype=z.dtype);g=torch.empty_like(a);allowed=torch.empty((100,),device=z.device,dtype=torch.bool)
 wrap_triton(_angles)[(100,)](z,seen,used,running,W,v,c,p,ii,jj,ranks,budget,score,a,g,allowed,first,common,check,phi,num_warps=1,enable_fp_fusion=False)
 return a,g,allowed

@triton_op('sager_roundopt_v1::apply',mutates_args={'z','seen','used','running','counts','attempts','refreshes','p','score'})
def apply(z:torch.Tensor,seen:torch.Tensor,used:torch.Tensor,running:torch.Tensor,counts:torch.Tensor,attempts:torch.Tensor,refreshes:torch.Tensor,W:torch.Tensor,v:torch.Tensor,c:torch.Tensor,p:torch.Tensor,ii:torch.Tensor,jj:torch.Tensor,a:torch.Tensor,g:torch.Tensor,allowed:torch.Tensor,score:torch.Tensor,first:bool,common:bool,check:bool,trace:bool)->torch.Tensor:
 rows=torch.empty((1,1,8) if trace else (0,1,8),device=z.device,dtype=z.dtype)
 wrap_triton(_apply)[(1,)](z,seen,used,running,counts,attempts,refreshes,W,v,c,p,ii,jj,a,g,allowed,score,rows,first,common,check,trace,num_warps=1,enable_fp_fusion=False)
 return rows

def window_twenty(z,seen,used,running,counts,attempts,refreshes,W,v,c,p,ii,jj,ranks,budget,phi,check,trace,score,common,single):
 rows=[]
 for slot in range(20):
  a,g,allowed=angles(z,seen,used,running,W,v,c,p,ii,jj,ranks,budget,score,slot==0,common,phi,check)
  rows.append(apply(z,seen,used,running,counts,attempts,refreshes,W,v,c,p,ii,jj,a,g,allowed,score,slot==0,common,check,trace))
 return torch.cat(rows,dim=0)
