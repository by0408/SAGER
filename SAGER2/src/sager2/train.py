import copy
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import read_json, read_jsonl, write_json, sha256, digest
from .probe import ReLUProbe, Probe


def metrics(y, score):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    return {"n": len(y), "accuracy": float(accuracy_score(y, score >= 0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, score >= 0)),
            "AUROC": float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else None}


def predict(net, z, device, batch_size=2048):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(z[at:at + batch_size] if isinstance(z, torch.Tensor) else torch.from_numpy(np.array(z[at:at + batch_size], copy=True)).to(device)).cpu().numpy() for at in range(0, len(z), batch_size)])


def train(cfg):
    root, hidden = Path(cfg["probe_dir"]), Path(cfg["hidden_dir"])
    manifest = read_json(hidden / "manifest.json")
    if not manifest["complete"] or manifest["training_data_sha256"] != sha256(cfg["training_data"]):
        raise ValueError("Incomplete cache or changed training data")
    rows = read_jsonl(cfg["training_data"])
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    split = {part: np.array([i for i, r in enumerate(rows) if r["partition"] == part]) for part in ("train", "val", "test")}
    train_ix, val_ix = split["train"], split["val"]
    device, settings = cfg.get("train_device", "cuda:0"), cfg.get("training", {})
    seeds = settings.get("seeds", [20261009, 20261010, 20261011])
    width = int(settings.get("width", 32))
    protocol = {"hidden_manifest_sha256": sha256(hidden / "manifest.json"), "training_data_sha256": sha256(cfg["training_data"]),
                "training_settings": settings, "train_device": device, "probe_code_sha256": sha256(Path(__file__).parent / "probe.py"),
                "training_code_sha256": sha256(__file__)}
    protocol["fingerprint"] = digest(protocol)
    if root.exists():
        if not (root / "training_protocol.json").exists() or read_json(root / "training_protocol.json") != protocol:
            raise ValueError("Existing probe directory has a different training protocol")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "training_protocol.json", protocol)
    reports, threshold_rows = [], []
    for layer in manifest["layers"]:
        completed_path = root / ("block%d_training.json" % layer)
        state_path = root / ("block%d.pt" % layer)
        if completed_path.exists() and state_path.exists():
            record = read_json(completed_path)
            if record.get("threshold_record", {}).get("checkpoint_sha256") != sha256(state_path):
                raise ValueError("Completed layer checkpoint checksum mismatch")
            reports.append(record)
            threshold_rows.append(record["threshold_record"])
            print("RESUME_COMPLETED_LAYER", layer, flush=True)
            continue
        z = np.load(hidden / ("block%d.npy" % layer), mmap_mode="r")
        if z.shape != (len(rows), manifest["adapter"]["dimension"]):
            raise ValueError("Wrong hidden cache shape")
        # One layer is ~0.95 GB for this dataset. Cache it once rather than
        # repeatedly copying random memmap batches across PCIe every epoch.
        z_device = torch.from_numpy(np.array(z, copy=True)).to(device)
        y_device = torch.from_numpy(y).to(device)
        val_device = torch.tensor(val_ix, device=device, dtype=torch.long)
        best_across, chosen, histories = math.inf, None, []
        started = time.time()
        for seed in seeds:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = ReLUProbe(z.shape[1], width, settings.get("dropout", 0.1)).to(device)
            weights = [p for name, p in net.named_parameters() if name.endswith("weight")]
            biases = [p for name, p in net.named_parameters() if name.endswith("bias")]
            opt = torch.optim.AdamW([{"params": weights, "weight_decay": settings.get("weight_decay", 0.01)},
                                    {"params": biases, "weight_decay": 0.0}], lr=settings.get("lr", 0.001))
            best, best_state, stale, best_epoch = math.inf, None, 0, 0
            for epoch in range(1, int(settings.get("epochs", 120)) + 1):
                write_json(root / "train_status.json", {"stage": "training", "layer": layer, "seed": seed, "epoch": epoch,
                           "completed_layers": len(reports), "target_layers": len(manifest["layers"]), "layer_seconds": time.time() - started})
                net.train()
                order = rng.permutation(train_ix)
                batch_size = int(settings.get("batch_size", 1024))
                for at in range(0, len(order), batch_size):
                    ix = order[at:at + batch_size]
                    batch_indices = torch.tensor(ix, device=device, dtype=torch.long)
                    x, target = z_device[batch_indices], y_device[batch_indices]
                    loss = F.binary_cross_entropy_with_logits(net(x), target)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                score_val = predict(net, z_device[val_device], device)
                val_loss = float(F.binary_cross_entropy_with_logits(torch.from_numpy(score_val), torch.from_numpy(y[val_ix])))
                if not math.isfinite(val_loss):
                    raise ValueError("Nonfinite validation loss")
                histories.append({"seed": seed, "epoch": epoch, "validation_BCE": val_loss})
                if val_loss < best - 1e-5:
                    best, best_state, stale, best_epoch = val_loss, copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()}), 0, epoch
                else:
                    stale += 1
                if stale >= int(settings.get("patience", 12)):
                    break
            if best < best_across:
                best_across, chosen = best, {"state": best_state, "seed": seed, "best_epoch": best_epoch}
        net = ReLUProbe(z.shape[1], width).to(device)
        net.load_state_dict(chosen["state"])
        scores = predict(net, z_device, device)
        probe = Probe(chosen["state"])
        benign = train_ix[y[train_ix] == 1]
        threshold = float(np.mean([probe.score(np.asarray(z[i], dtype=np.float64)) for i in benign]))
        state_path = root / ("block%d.pt" % layer)
        torch.save(chosen["state"], state_path)
        row = {"layer": layer, "benign_mean_logit_threshold": threshold, "benign_training_count": len(benign), "checkpoint_sha256": sha256(state_path)}
        threshold_rows.append(row)
        cohorts = sorted({r.get("cohort", r["partition"]) for r in rows if r["partition"] == "test"})
        cohort_metrics = {}
        for cohort in cohorts:
            ix = np.array([i for i, r in enumerate(rows) if r["partition"] == "test" and r.get("cohort", r["partition"]) == cohort])
            cohort_metrics[cohort] = metrics(y[ix], scores[ix])
        report = {"layer": layer, "seed": chosen["seed"], "best_epoch": chosen["best_epoch"], "best_validation_BCE": best_across,
                  "seconds": time.time() - started, "metrics": {part: metrics(y[ix], scores[ix]) for part, ix in split.items()},
                  "test_cohorts": cohort_metrics, "history": histories, "threshold_record": row}
        reports.append(report)
        write_json(root / ("block%d_training.json" % layer), report)
        print("TRAIN", layer, "validation_BCE", best_across, "test_AUROC", report["metrics"]["test"]["AUROC"], flush=True)
        del z_device, y_device, val_device
    write_json(root / "thresholds.json", threshold_rows)
    write_json(root / "manifest.json", {"complete": True, "model_identity": manifest["model_identity"], "layers": manifest["layers"],
               "dimension": manifest["adapter"]["dimension"], "width": width, "positive_class": "benign=1", "input_scale": "sqrt(d)*unit_L2",
               "training_data_sha256": sha256(cfg["training_data"]), "selection": "minimum validation BCE across seeds, no test/generation selection",
               "threshold": "mean logit of benign TRAINING samples only", "hidden_site": manifest["hidden_site"]})
    write_json(root / "training_summary.json", reports)
    write_json(root / "train_status.json", {"stage": "complete", "completed_layers": len(reports), "target_layers": len(manifest["layers"])})


