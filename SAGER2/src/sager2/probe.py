import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ReLUProbe(nn.Module):
    def __init__(self, dimension, width=32, dropout=0.1):
        super().__init__()
        self.dimension = dimension
        self.fc1 = nn.Linear(dimension, width)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, 1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        return self.fc2(self.dropout(F.relu(self.fc1(z * math.sqrt(self.dimension))))).squeeze(-1)


class Probe:
    """FP64 analytic evaluator; effective W absorbs the training sqrt(d) scale."""
    def __init__(self, state):
        self.W = state["fc1.weight"].cpu().double().numpy().copy()
        self.W *= math.sqrt(self.W.shape[1])
        self.b = state["fc1.bias"].cpu().double().numpy().copy()
        self.v = state["fc2.weight"].cpu().double().numpy().reshape(-1).copy()
        self.c = float(state["fc2.bias"].cpu().double().item())
        if not all(np.isfinite(x).all() for x in (self.W, self.b, self.v)) or not math.isfinite(self.c):
            raise ValueError("Non-finite probe parameters")

    def pre(self, z):
        return self.W @ z + self.b

    def score(self, z):
        return float(np.maximum(self.pre(z), 0) @ self.v + self.c)

    def gradient(self, z):
        return self.W.T @ (self.v * (self.pre(z) > 0))
