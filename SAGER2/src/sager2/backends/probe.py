"""FP64 analytic probe shared by all six refresh-20 model configurations."""
import math
import torch

class GPUProbe:
    """Effective analytic weights: W already includes the training sqrt(d)."""
    def __init__(self, W, b, v, c, device="cuda:0"):
        self.W = torch.as_tensor(W, device=device, dtype=torch.float64).contiguous()
        self.b = torch.as_tensor(b, device=device, dtype=torch.float64).contiguous()
        self.v = torch.as_tensor(v, device=device, dtype=torch.float64).reshape(-1).contiguous()
        self.c = torch.as_tensor(c, device=device, dtype=torch.float64).reshape(())
        self.d, self.width = self.W.shape[1], self.W.shape[0]
        if self.b.shape != (self.width,) or self.v.shape != (self.width,):
            raise ValueError("Incompatible analytic probe shapes")

    @classmethod
    def from_probe(cls, probe, device="cuda:0"):
        return cls(probe.W, probe.b, probe.v, probe.c, device)

    @classmethod
    def from_state(cls, state, device="cuda:0"):
        W = state["fc1.weight"].detach().to(device=device, dtype=torch.float64)
        return cls(W * math.sqrt(W.shape[1]), state["fc1.bias"],
                   state["fc2.weight"], state["fc2.bias"], device)

    def pre(self, z):
        return z @ self.W.T + self.b

    def score(self, z):
        return self.pre(z).clamp_min(0) @ self.v + self.c
