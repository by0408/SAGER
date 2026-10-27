# Reproducibility contract

The main command performs one paired1,199-prompt experiment. A GPU count of3,
three probe-training candidate seeds, or three worker logs does not mean three
independent main-experiment repetitions. Run separate named output directories
to perform additional repetitions and retain each run's raw records.

Probe training uses width32, ReLU, dropout0.1, AdamW, learning rate1e-3,
batch1024, at most120 epochs, patience12, and the minimum validation BCE across
the configured three seeds. The input is `sqrt(d)*unit_hidden`. The trained
checkpoint and benign TRAIN mean are model- and layer-specific. Original `.pt`
bytes and thresholds were used for verification, then removed at the author's
request. Train probes into a new directory and pass --probe-dir for evaluation.

The main pipeline uses the original full FP16 model, same physical GPU for both
members of each pair, raw input, native BOS, no chat template, KV cache, greedy
argmax, repetition penalty1, EOS stopping, and max10 new tokens. It refuses silent
truncation, quantization, offload, changed model/probe identities, or mixed resume
protocols. Models are loaded locally; no background network download is needed.

Detoxify Original is scored on the continuation only with its existing
`text.strip()` behavior, on CPU. This is separate from PPL preprocessing.

Primary PPL protocol: `ARGRE_tail_mask_raw_no_strip_no_response_bos_v2`.
Use the same unedited model after removing hooks. Retain raw decoded response
text without stripping; tokenize the response separately without special tokens
to obtain its length; tokenize prompt+response normally with native BOS; score
the response-length tail. Aggregate as `exp(mean(per-sample NLL))`, with equal
sample weights. This is a retokenization boundary approximation and is not
identical to every historical ARGRE preprocessing variant. Auxiliary actual-ID
PPL masks the original prompt IDs and includes actual generated EOS; it is
reported separately and never silently replaces the primary metric.

Portable model identity checks config/index SHA and weight-shard names/sizes,
plus tokenizer SHA when provided. It intentionally omits machine path and mtime
so the project can move. It is not a full language-model weight SHA check.

Each run writes its resolved configuration, source hashes, data/probe identities,
GPU UUIDs, atomic paired texts, diagnostic files, metric records and actual worker
exit codes. No generated benchmark numbers are hard-coded into the new project.
Full six-model generation has not been rerun. LLaMA-7B alone was verified on all1199
prompts against fresh original code on identical physicalGPUs, with an external
report retained after authorized removal of raw outputs. Full per-rotation
coordinate/angle traces were not recorded.
