"""Prepare labeled text from the audited prior manifest; no Gemma token IDs reused."""
import collections
from pathlib import Path
from .common import read_json, read_jsonl, write_json, sha256, digest


def validate_rows(rows, challenge=None):
    groups, texts = {}, {}
    for row in rows:
        if row["y"] not in (0, 1) or not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Need nonempty text and y=1 benign / y=0 toxic")
        part = row["partition"]
        if part not in ("train", "val", "test"):
            raise ValueError("Partitions must be train, val, test")
        key = digest(row["text"].strip())
        for mapping, value in ((groups, str(row["group"])), (texts, key)):
            if value in mapping and mapping[value] != part:
                raise ValueError("Group or exact text overlaps partitions: " + str(value))
            mapping[value] = part
    audit = {"samples": len(rows), "partition_counts": dict(collections.Counter(r["partition"] for r in rows)),
             "source_counts": dict(collections.Counter(r.get("source", "custom") for r in rows)),
             "class_counts": dict(collections.Counter(str(r["y"]) for r in rows)),
             "split_overlap_check": "groups and exact stripped text", "semantic_duplicates_checked": False}
    if challenge:
        prompts = [r["prompt"].strip() for r in read_jsonl(challenge)]
        train_text = {r["text"].strip() for r in rows if r["partition"] == "train"}
        overlaps = sorted(set(prompts) & train_text)
        audit["challenge_exact_training_overlaps"] = len(overlaps)
        if overlaps:
            raise ValueError("challenge_prompts overlaps training texts")
    for part in ("train", "val", "test"):
        labels = {r["y"] for r in rows if r["partition"] == part}
        if labels != {0, 1}:
            raise ValueError("Each partition needs both classes: " + part)
    return audit


