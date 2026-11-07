"""Read-only provenance checks for the inherited, unchanged benchmark rows."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(values) -> dict[str, int]:
    return dict(sorted(Counter(map(str, values)).items()))


def load_artifact(path: Path, source_to_domain: dict[str, str]) -> tuple[dict, dict, dict]:
    """Load immutable pre-threshold scores, validating alignment and domains."""
    metadata_path = path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    required = {"dev_dv", "dev_labels", "test_probe", "test_expert", "test_labels", "test_dv", "test_groups"}
    if missing := required - arrays.keys():
        raise ValueError(f"Missing score arrays in {path}: {sorted(missing)}")
    for split in ("dev", "test"):
        n = len(arrays[f"{split}_labels"])
        if not n:
            raise ValueError(f"Empty {split} artifact")
        for name, values in arrays.items():
            if not name.startswith(split + "_"):
                continue
            if values.ndim != 1 or len(values) != n:
                raise ValueError(f"Misaligned array {name} in {path}")
            if values.dtype.kind in "biuf" and not np.isfinite(values).all():
                raise ValueError(f"Non-finite values in {name}")
        if not np.isin(arrays[f"{split}_labels"], [0, 1]).all():
            raise ValueError("Expected binary stakes labels")
    if not source_to_domain or any(not isinstance(v, str) or not v for v in source_to_domain.values()):
        raise ValueError("Specify a nonempty source-to-domain mapping")
    sources = arrays["test_groups"].astype(str)
    if missing := set(sources) - source_to_domain.keys():
        raise ValueError(f"Unmapped source groups: {sorted(missing)}")
    arrays["test_domains"] = np.array([source_to_domain[g] for g in sources])
    inventory = {
        "artifact": str(path.resolve()),
        "artifact_sha256": file_sha256(path),
        "metadata_sha256": file_sha256(metadata_path),
        "dev_count": len(arrays["dev_labels"]),
        "test_count": len(sources),
        "test_sources": counts(sources),
        "test_domains": counts(arrays["test_domains"]),
        "source_to_domain": source_to_domain,
        "dataset_policy": "Preserve all inherited rows, labels and dev/test assignments; map group labels only.",
    }
    return arrays, metadata, inventory


def reconstruct_rows(metadata: dict, data_root: Path) -> tuple[dict, list[dict]]:
    """Reproduce the original per-source cap without editing the source files.

    Source paths in GPU metadata can differ from this machine. Match the exact
    filenames under data_root/evals/{dev,test}, and record their current hashes.
    These legacy artifacts have no saved IDs or source-file hashes; reconstructed
    IDs are checked against score labels/group order, not claimed as original IDs.
    """
    config = metadata["config"]
    mix = config["mixed_datasets"]
    result, provenance = {}, []
    for split in ("dev", "test"):
        sources = []
        for source in mix["sources"]:
            path = data_root / "evals" / split / Path(source[split]).name
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            sources.append((source["group"], rows))
            provenance.append(
                {
                    "split": split,
                    "source": source["group"],
                    "filename": path.name,
                    "rows": len(rows),
                    "sha256": file_sha256(path),
                }
            )
        balance = mix.get("balance_strategy", "min_size")
        if balance == "min_size":
            cap = min(len(rows) for _, rows in sources)
        elif balance == "none":
            cap = None
        elif isinstance(balance, int) and not isinstance(balance, bool) and balance > 0:
            cap = balance
        else:
            raise ValueError(f"Unsupported source cap: {balance!r}")
        selected = []
        for source, rows in sources:
            indices = (
                random.Random(config["seed"]).sample(range(len(rows)), cap)
                if cap and len(rows) > cap
                else range(len(rows))
            )
            selected.extend({"source": source, "row_index": i, "record": rows[i]} for i in indices)
        result[split] = selected
    return result, provenance


def audit_inherited_overlap(rows: dict) -> dict:
    """Describe exact inputs, repeated MT notes and shared source content.

    This is a descriptive exact/normalised text check, not a claim to detect all
    semantic duplicates or identify patients. It never filters any examples.
    """

    def norm(text):
        return re.sub("[^a-z0-9]", "", text.lower())

    def input_hash(item):
        return hashlib.sha256(json.dumps(item["record"]["inputs"], sort_keys=True).encode()).hexdigest()

    dev_hashes = {input_hash(r) for r in rows["dev"]}
    mt_test = [r for r in rows["test"] if r["source"] == "mt"]
    mt_dev = [r for r in rows["dev"] if r["source"] == "mt"]
    matches = []
    for item in rows["test"]:
        if item["source"] != "mts":
            continue
        section = norm(item["record"]["section_text"])
        if len(section) < 80:
            continue
        for mt in mt_dev:
            if section in norm(mt["record"]["transcription"]):
                matches.append(
                    {
                        "mts_test_id": str(item["record"]["ids"]),
                        "mt_dev_id": str(mt["record"]["ids"]),
                        "normalised_section_length": len(section),
                    }
                )
    pairs = {}
    for split in ("dev", "test"):
        pairs[split] = {str(r["record"]["ids"]).rsplit("_", 1)[0] for r in rows[split] if r["source"] == "anthropic"}
    return {
        "policy": "Report inherited overlap; remove or relabel no rows.",
        "test_rows_with_exact_input_in_dev": sum(input_hash(r) in dev_hashes for r in rows["test"]),
        "mt_test_rows": len(mt_test),
        "mt_distinct_normalised_test_notes": len({norm(r["record"]["transcription"]) for r in mt_test}),
        "anthropic_preference_pairs_shared_between_dev_test": len(pairs["dev"] & pairs["test"]),
        "mts_test_rows_with_long_note_section_in_mt_dev": len({r["mts_test_id"] for r in matches}),
        "medical_content_match_examples": matches[:10],
    }


def add_row_inventory(arrays: dict, metadata: dict, inventory: dict, data_root: Path) -> dict:
    rows, provenance = reconstruct_rows(metadata, data_root)
    manifests = {}
    for split in ("dev", "test"):
        labels = []
        manifest = []
        for item in rows[split]:
            row = item["record"]
            if row["labels"] not in {"high-stakes", "low-stakes"}:
                raise ValueError("The official balanced source unexpectedly contains an ambiguous label")
            labels.append(int(row["labels"] == "high-stakes"))
            manifest.append(
                {
                    "source": item["source"],
                    "source_id": str(row["ids"]),
                    "source_row_index": item["row_index"],
                    "label": labels[-1],
                }
            )
        if not np.array_equal(labels, arrays[f"{split}_labels"]):
            raise ValueError(f"Reconstructed {split} source labels do not match the saved scores")
        if split == "test" and not np.array_equal([r["source"] for r in manifest], arrays["test_groups"]):
            raise ValueError("Reconstructed source order does not match saved test groups")
        manifests[split] = manifest
    inventory.update(
        {
            "source_files": provenance,
            "row_manifest": manifests,
            "id_provenance": "Reconstructed from metadata and current source files; label and test-group order checked.",
            "inherited_overlap": audit_inherited_overlap(rows),
        }
    )
    return inventory
