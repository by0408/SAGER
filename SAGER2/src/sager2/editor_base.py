"""Batch hooks for the independent, experimental GPU SAGER backend.

The original ``sager`` package is deliberately left unchanged.  All probe and
rotation arithmetic here is float64.  Decoder states retain the model dtype.
Diagnostics stay on the device until generation has finished and are exported
in the original per-sample Editor record schema.

Caller contract: left pad the prompt batch, set explicit position IDs, use an
ordinary KV cache, and call ``set_alive`` before *every* model forward.  A row
which has generated EOS must be marked inactive before its next forward.
Only the last sequence position is edited.  For a left-padded prefill this is
the last real prompt token, and for cached decoding it is the sole new token.
"""

from contextlib import contextmanager
from pathlib import Path

import torch

from .common import read_json, sha256
from .models import hidden_tensor, replace_hidden


def _score(probe, z):
    """Effective W already incorporates the training sqrt(d) scale."""
    return (z @ probe.W.T + probe.b).clamp_min(0) @ probe.v + probe.c


class BatchEditor:
    """Gate and edit each row independently, with deferred device diagnostics.

    ``solver_factory`` may be injected for validation.  It is called as
    ``solver_factory(probe, device=..., top=..., screen=..., exact=...,
    max_angle_degrees=..., backend=...)`` and must implement the GPU solver's
    ``rotate`` and ``backend_status`` interfaces.  A ``probe_factory(state,
    device=...)`` can likewise be injected; its result exposes W, b, v and c.
    """

    def __init__(self, adapter, cfg, identity, solver_factory=None,
                 backend="eager", record_rotations=False, probe_factory=None,
                 measure_search=False):
        root = Path(cfg["probe_dir"])
        manifest = read_json(root / "manifest.json")
        self.layers = adapter.layers(cfg)
        if not manifest["complete"] or manifest["model_identity"] != identity:
            raise ValueError("Probe belongs to a different model or is incomplete")
        if (manifest["dimension"] != adapter.dimension or
                manifest["positive_class"] != "benign=1"):
            raise ValueError("Probe dimension or class orientation mismatch")
        if (manifest["input_scale"] != "sqrt(d)*unit_L2" or
                not set(self.layers) <= set(manifest["layers"])):
            raise ValueError("Wrong probe scale or missing layers")

        edit = cfg["editing"]
        self.rounds = int(edit.get("rounds", 200))
        self.max_distinct = int(edit.get("max_distinct_coordinates", 400))
        top_g, top_z = int(edit.get("top_gradient", 2)), int(edit.get("top_hidden", 2))
        if self.rounds < 1 or self.max_distinct < 2:
            raise ValueError("Invalid editing budget")
        if top_g != top_z:
            raise ValueError("GPU backend requires equal gradient and hidden Top-K")
        if edit.get("stop_at_threshold", False):
            raise ValueError("This experiment requires no threshold stopping")
        if backend not in ("eager", "compile", "graph", "compile_graph"):
            raise ValueError("Unknown GPU search backend")

        if solver_factory is None or probe_factory is None:
            from .backends.probe import GPUProbe
            from .backends.window_solver import GPUSearch
            if solver_factory is None:
                solver_factory = GPUSearch
            if probe_factory is None:
                probe_factory = GPUProbe.from_state

        self.adapter, self.cfg = adapter, cfg
        self.backend = backend
        self.record_rotations = bool(record_rotations)
        self.measure_search = bool(measure_search)
        self.thresholds, self.probes, self.searches = {}, {}, {}
        self.checkpoint_hashes, self.layer_devices = {}, {}
        rows = {int(r["layer"]): r for r in read_json(root / "thresholds.json")}
        for layer in self.layers:
            row = rows[layer]
            path = root / ("block%d.pt" % layer)
            checkpoint_hash = sha256(path)
            if checkpoint_hash != row["checkpoint_sha256"]:
                raise ValueError("Changed probe checkpoint at layer %d" % layer)
            try:
                device = next(adapter.blocks[layer - 1].parameters()).device
            except StopIteration:
                device = adapter.input_device
            if device.type == "meta":
                raise ValueError("Offloaded meta blocks are unsupported by this backend")
            state = torch.load(path, map_location="cpu", weights_only=True)
            probe = probe_factory(state, device=device)
            if (probe.W.dtype != torch.float64 or probe.W.device != device or
                    tuple(probe.W.shape)[1] != adapter.dimension):
                raise ValueError("GPU probe must use resident FP64 effective weights")
            self.layer_devices[layer] = device
            self.probes[layer] = probe
            self.thresholds[layer] = float(row["benign_mean_logit_threshold"])
            self.checkpoint_hashes[layer] = checkpoint_hash
            self.searches[layer] = solver_factory(
                probe, device=device, top=top_g,
                screen=int(edit.get("screen_candidates", 2000)),
                exact=int(edit.get("exact_candidates", 100)),
                max_angle_degrees=float(edit.get("max_angle_degrees", 5)),
                backend=backend)
        self.hooks = []
        self.alive_mask = None
        self._alive_by_device = {}
        self.batch_size = None
        self.calls = {}
        self._pending = []
        self._exported = None
        self._exported_rotations = None

    def set_alive(self, mask):
        """Set [B] bool mask before a forward; no device data is read on CPU."""
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.bool,
                                   device=self.adapter.input_device)
        if mask.ndim != 1 or mask.dtype != torch.bool:
            raise ValueError("Alive mask must be a one-dimensional bool tensor")
        if self.batch_size is not None and mask.shape[0] != self.batch_size:
            raise ValueError("Batch size changed inside an editing context")
        self.alive_mask = mask.detach()
        # Prepare any needed device copies once per token, not per layer.
        self._alive_by_device = {
            str(device): self.alive_mask.to(device=device)
            for device in set(self.layer_devices.values())}

    set_active_mask = set_alive

    @contextmanager
    def active(self, rotation_log=None):
        if self.hooks:
            raise RuntimeError("Editor context already active")
        self.calls, self._pending = {}, []
        self._exported, self._exported_rotations = None, None
        self.batch_size = None
        self.rotation_log = rotation_log
        try:
            self.hooks = [self.adapter.blocks[layer - 1].register_forward_hook(
                self.make_hook(layer)) for layer in self.layers]
            yield self
        finally:
            for handle in self.hooks:
                handle.remove()
            self.hooks = []


    @staticmethod
    def _deferred_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {k: BatchEditor._deferred_cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [BatchEditor._deferred_cpu(v) for v in value]
        return value

    @staticmethod
    def _row(value, sample, batch_size):
        if isinstance(value, list) and len(value) == batch_size:
            return value[sample]
        return value





    def backend_status(self):
        return {"requested": self.backend,
                "precision": "FP64 search and probes, model dtype output",
                "batch_size": self.batch_size,
                "probe_residency": "all selected layer parameters cached on block device",
                "diagnostic_export": "deferred until generation context closes",
                "layers": {str(layer): self.searches[layer].backend_status()
                           for layer in self.layers}}
