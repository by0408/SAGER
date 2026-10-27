import time
from pathlib import Path
import numpy as np
import torch
from .common import read_jsonl, write_json, sha256
from .data import validate_rows
from .models import load_model, hidden_tensor


@torch.no_grad()
def extract(cfg):
    root = Path(cfg["hidden_dir"])
    if root.exists():
        raise FileExistsError("Choose a new hidden_dir: " + str(root))
    rows = read_jsonl(cfg["training_data"])
    audit = validate_rows(rows, cfg["challenge_data"])
    model, tok, adapter, identity = load_model(cfg)
    layers = adapter.layers(cfg)
    root.mkdir(parents=True)
    max_length = int(cfg.get("max_input_tokens", 512))
    max_length = min(max_length, int(model.config.max_position_embeddings))
    tok.padding_side, tok.truncation_side = "right", "left"
    arrays = {l: np.lib.format.open_memmap(root / ("block%d.npy" % l), mode="w+", dtype="float32", shape=(len(rows), adapter.dimension)) for l in layers}
    context, hooks = {}, []

    def make_hook(layer):
        def hook(module, args, output):
            h = hidden_tensor(output)
            positions = context["positions"].to(h.device)
            selected = h[torch.arange(h.shape[0], device=h.device), positions].float()
            norm = torch.linalg.vector_norm(selected, dim=-1, keepdim=True)
            if not torch.isfinite(selected).all() or (norm <= 0).any():
                raise ValueError("Invalid hidden")
            arrays[layer][context["start"]:context["end"]] = (selected / norm).cpu().numpy()
            context["seen"].add(layer)
        return hook

    try:
        hooks = [adapter.blocks[l - 1].register_forward_hook(make_hook(l)) for l in layers]
        size = int(cfg.get("extract_batch_size", 8))
        start = time.time()
        for at in range(0, len(rows), size):
            batch = rows[at:at + size]
            encoded = tok([r["text"] for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            context.update(start=at, end=at + len(batch), positions=encoded.attention_mask.sum(-1) - 1, seen=set())
            # Decoder outputs are sufficient: skip the unused, large vocabulary logits.
            model.model(**{k: v.to(adapter.input_device) for k, v in encoded.items()}, use_cache=False)
            if context["seen"] != set(layers):
                raise RuntimeError("Missing extraction hook")
            if at % (size * 20) == 0:
                write_json(root / "status.json", {"stage": "extracting", "completed": at + len(batch), "total": len(rows), "seconds": time.time() - start})
                print("EXTRACT", at + len(batch), "/", len(rows), flush=True)
        for arr in arrays.values():
            arr.flush()
    finally:
        for hook in hooks:
            hook.remove()
    # Written only after all memmaps are complete. Training refuses partial caches.
    write_json(root / "manifest.json", {"complete": True, "model_identity": identity, "adapter": adapter.describe(),
               "layers": layers, "samples": len(rows), "training_data_sha256": sha256(cfg["training_data"]),
               "max_input_tokens": max_length, "hidden_site": "decoder block output before final norm",
               "position": "last nonpadding input token", "normalization": "unit L2 in float32; training multiplies sqrt(d)",
               "data_audit": audit})
    write_json(root / "status.json", {"stage": "complete", "completed": len(rows)})
