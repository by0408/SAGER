# SAGER

Small-Angle Givens Editing of Representations for language-model detoxification.
This standalone project contains only the main1199-prompt Challenge experiment,
the shared7B training dataset, probe training, FP64 GPU editing, decoding and scoring.
See [中文说明](README_zh.md).

All trained probe copies and the verification run's raw results have been removed
at the author's request after full LLaMA-7B alignment. The original SAGER project
is retained separately. A brief verification report is stored outside this project.
No pretrained probes, language-model weights, hidden caches or generation archives
are distributed here.

## Setup and run

The tested source environment is Linux, Python3.10.19, PyTorch2.9.1+cu128,
Transformers4.57.1 and Triton3.5.1. Choose a compatible CUDA PyTorch installation
before installing requirements. Existing server dependencies need no reinstall.

```bash
cp local_paths.example.json local_paths.json
# Set local language-model and Detoxify checkpoint/tokenizer locations.
python run.py doctor --config configs/llama7b.json --data-only
CUDA_VISIBLE_DEVICES=0 python run.py extract --config configs/llama7b.json \
  --hidden-dir cache/retrain_llama7b
CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/llama7b.json \
  --hidden-dir cache/retrain_llama7b --probe-dir outputs/retrained_llama7b_probes
python run.py doctor --config configs/llama7b.json --probe-dir outputs/retrained_llama7b_probes
python run.py evaluate --config configs/llama7b.json --probe-dir outputs/retrained_llama7b_probes --gpus 0 3
```

Without trained probes, data-only preflight validates datasets/configuration only;
full preflight and evaluation refuse to run. A new training run is not assumed to
reproduce the removed original probe bytes. Models and scorer weights remain external.

The13B and30B training datasets are not bundled. Those configurations leave
`training_data` unset; pass `--training-data data/external/original.jsonl` to doctor,
extract, train or evaluate. Their original dataset SHA and expected training counts
are retained. The shared7B dataset is not substituted automatically.

Example model paths are relative to the project root. A user-created
`local_paths.json` may configure local locations; it is ignored by Git and must
not be included in a public release.

`evaluate` starts detached workers, writes atomic paired results and enforces same
physicalGPU pairing. Each explicitly selected GPU holds one full FP16 model replica.
No quantization, offload, eviction of other jobs, or independent-repeat claim is made.

```bash
python run.py status --run-dir outputs/llama7b
python run.py stop --run-dir outputs/llama7b
# Explicit --resume requires unchanged source/data/model/probes/GPU assignment.
```

## Main configurations

| Model | Edit layers | Hidden size | TRAIN | Refresh |
|---|---:|---:|---:|---:|
| LLaMA-7B | 15–32 |4096|48,500|20|
| LLaMA-7B-SFT |15–32|4096|48,500|20|
| Mistral-7B |15–32|4096|48,500|20|
| Mistral-7B-SFT |15–32|4096|48,500|20|
| LLaMA-13B |15–40|5120|43,053|20|
| LLaMA-30B |15–60|6656|10,000|20|

The13B/30B TRAIN counts describe their original external datasets.

Each uses Top100 candidates, the benign TRAIN mean gate, FP64 probes/search, up to
500 attempted rotation slots, ±5° per pair, an800-distinct-coordinate guard, and
newline-token-ID13 skipping. Crossing the threshold does not stop the search.
Per-step angle bounds do not bound cumulative drift. Generation is batch1, raw,
native BOS, greedy, KV-cache, EOS-stopped, max10 new tokens.

LLaMA-7B was checked on all1199 prompts against fresh original code with matching
physicalGPU assignment. This does not rerun the other five models. In particular,
historical30B output is not evidence for its newly unified refresh20 configuration.
The run compared output tokens, layer gates, rotation/count summaries and metrics;
it did not record full per-rotation coordinate/angle trajectories.

See [method](docs/METHOD.md), [data](docs/DATASETS.md), and
[reproducibility](docs/REPRODUCIBILITY.md) for precise scoring/training scope.

## Checks and publication

```bash
python -m unittest discover -s tests -v
```

The checkpoint-dependent CPU test explicitly skips when pretrained probes are absent.
The other CPU tests do not constitute a six-model GPU replication. `models/`,
`cache/`, `outputs/`, checkpoint files, local paths, logs and ZIPs are ignored by Git.
No ablations, sweeps, latency benchmarks, human/AI reviews, showcase selection or
general-capability tests are included. Dataset/model source terms remain applicable;
the author must choose a code license before publication. Nothing was uploaded to GitHub.
