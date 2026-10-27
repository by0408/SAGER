# Bundled data

`data/challenge_prompts.jsonl` is the original1,199-prompt Challenge evaluation
file. Every line contains the original raw `prompt`; no dataset download or
hidden-state extraction is required to inspect these texts.

The bundled probe dataset is an actual file, not a link to the old project:

| File | TRAIN | VAL | TEST | Used by |
|---|---:|---:|---:|---|
| `labeled.jsonl` | 48,500 | 224 | 9,242 | 7B models |

Fields preserve text, source ID, label, partition, cohort, source group and
label provenance. The positive label is benign=1. Some original labels are weak
or noisy; preserving a benign label is not a guarantee that every text is safe.
Groups and exact stripped texts are checked for split overlap. The check does
not prove absence of all semantic near-duplicates.

The full training data combines Jigsaw, Civil Comments, ARGRE paired data, and
the previously retained hard Jigsaw cohort. The13B and30B model-specific datasets
and the separate data-directory manifest were removed at the author's request.
Their original partitions contain43,053/224/7,243 and10,000/224/9,242
TRAIN/VAL/TEST rows, respectively. To use those configurations, supply the original
external JSONL with `--training-data`; its SHA and training count are checked.
The shared7B dataset is not an implicit replacement. Each experiment records the
training and Challenge dataset SHA256. Each newly trained probe manifest is also
checked against its own dataset SHA.

Upstream resources:

- RealToxicityPrompts: https://huggingface.co/datasets/allenai/real-toxicity-prompts
- Jigsaw: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
- Civil Comments: https://huggingface.co/datasets/google/civil_comments
- ARGRE: https://github.com/xiaoyisong/ARGRE

These source identifiers describe provenance; this release does not grant a new
license to third-party texts. The author should retain upstream attribution and
review the applicable source distribution terms before public redistribution.
No parameter-selection prompts, capability-task datasets, or hidden tensors are
included.
