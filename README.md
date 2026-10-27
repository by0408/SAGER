# SAGER

**Small-Angle Givens Editing of Representations** for language-model detoxification.

This standalone repository contains the main **1,199-prompt Challenge experiment**, the shared 7B training dataset, layerwise toxicity-probe training, FP64 GPU editing, decoding, and scoring. It does not include ablations, parameter sweeps, latency benchmarks, human/AI reviews, showcase selection, or general-capability evaluations.

## Requirements

The source code was tested on **Linux** with:

| Component    | Tested version |
| ------------ | -------------- |
| Python       | 3.10.19        |
| PyTorch      | 2.9.1+cu128    |
| Transformers | 4.57.1         |
| Triton       | 3.5.1          |

Install a CUDA-compatible build of PyTorch before installing the remaining requirements. Dependencies already available on a server do not need to be reinstalled.

Model and Detoxify checkpoints/tokenizers are external dependencies. Configure their local locations in `local_paths.json`; this file is ignored by Git and must not be included in a public release. Example model paths are relative to the project root.

## Quick start

```bash
cp local_paths.example.json local_paths.json
# Edit local_paths.json to specify model and Detoxify paths.

python run.py doctor --config configs/llama7b.json --data-only

# Extract hidden states and train layerwise probes.
CUDA_VISIBLE_DEVICES=0 python run.py extract --config configs/llama7b.json \
  --hidden-dir cache/retrain_llama7b
CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/llama7b.json \
  --hidden-dir cache/retrain_llama7b \
  --probe-dir outputs/retrained_llama7b_probes

# Validate the trained probes, then run the main evaluation.
python run.py doctor --config configs/llama7b.json \
  --probe-dir outputs/retrained_llama7b_probes
python run.py evaluate --config configs/llama7b.json \
  --probe-dir outputs/retrained_llama7b_probes --gpus 0 3
```

Use the following commands to inspect or stop a run:

```bash
python run.py status --run-dir outputs/llama7b
python run.py stop --run-dir outputs/llama7b
```

An explicit `--resume` requires the source, datasets, model, probes, and physical-GPU assignment to remain unchanged.

## Model configurations

| Model          | Edited layers | Hidden size | Training examples | Candidate refresh interval |
| -------------- | ------------: | ----------: | ----------------: | -------------------------: |
| LLaMA-7B       |         15–32 |       4,096 |            48,500 |                         20 |
| LLaMA-7B-SFT   |         15–32 |       4,096 |            48,500 |                         20 |
| Mistral-7B     |         15–32 |       4,096 |            48,500 |                         20 |
| Mistral-7B-SFT |         15–32 |       4,096 |            48,500 |                         20 |
| LLaMA-13B      |         15–40 |       5,120 |            43,053 |                         20 |
| LLaMA-30B      |         15–60 |       6,656 |            10,000 |                         20 |

The four 7B configurations share the bundled training dataset. The 13B and 30B datasets are **not included**; their counts above refer to the original external datasets. Their configurations intentionally leave `training_data` unset. Supply the correct dataset to `doctor`, `extract`, `train`, or `evaluate`:

```bash
--training-data data/external/original.jsonl
```


### Editing and generation settings

All six configurations use the following settings:

- **Candidate selection:** Top-100 candidates and the benign TRAIN-mean gate.
- **Editing/search:** FP64 probe computations and search, up to 500 attempted rotation slots, and a maximum of ±5° per coordinate-pair rotation.
- **Safeguards:** An 800-distinct-coordinate guard and skipping of newline token ID 13.
- **Stopping behavior:** Crossing the probe threshold does not stop the search. The ±5° limit applies to each rotation, not to cumulative drift.
- **Generation:** Batch size 1, raw input, native BOS, greedy decoding, KV cache enabled, stopping at EOS, and at most 10 new tokens.

## Verification and reproducibility

The reorganized LLaMA-7B implementation was checked against a fresh execution of the original code on **all 1,199 Challenge prompts**, using matching physical-GPU assignments. The comparison covered generated token IDs, layer gates, rotation/count summaries, and evaluation metrics. It did **not** record complete per-rotation coordinate and angle trajectories.

This verification does not establish GPU replication of the other five model configurations. In particular, historical LLaMA-30B outputs do not verify the newly unified refresh-20 configuration.

After the LLaMA-7B check, the trained probe copies and raw verification results were removed from this distribution at the author's request. The original SAGER project remains stored separately, and a brief verification report is retained outside this repository. No pretrained probes, language-model weights, hidden-state caches, or generation archives are distributed here.

For detailed methodological and data provenance information, see [METHOD.md](docs/METHOD.md), [DATASETS.md](docs/DATASETS.md), and [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Tests

```bash
python -m unittest discover -s tests -v
```

The checkpoint-dependent CPU test skips explicitly when pretrained probes are unavailable. Passing the remaining CPU tests is **not** equivalent to reproducing all six models on GPUs.
