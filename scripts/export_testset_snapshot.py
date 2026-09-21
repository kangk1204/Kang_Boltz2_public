#!/usr/bin/env python
"""Export the declared historical cohort, retaining hashes and sampling provenance.

Run from the repository root. This reads existing predictions; it never predicts,
reanalyzes, imputes missing metrics, or treats mutant scores as affinity measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_snapshot(root: Path) -> dict:
    records = []
    sources = {}
    for kind, table, batch in (
        ("WT", "batch_wt_updated.tsv", "batch_testset_wt"),
        ("mutant", "batch_mutants_updated.tsv", "batch_testset_mutants"),
    ):
        table_path = root / "examples/testset" / table
        sources[str(table_path.relative_to(root))] = digest(table_path)
        with table_path.open(newline="") as handle:
            jobs = list(csv.DictReader(handle, delimiter="\t"))
        for job in jobs:
            result_path = root / "outputs" / batch / job["name"] / "analysis/results.json"
            result = json.loads(result_path.read_text())
            models = result["models"]
            best = next(m for m in models if m["index"] == result["best_model_index"])
            params = result["settings"].get("run_params") or {}
            if params.get("samples") != len(models):
                raise ValueError(f"sampling count mismatch: {job['name']}")
            inputs = {}
            for key in ("antigen", "nanobody", "reference"):
                value = job.get(key)
                if value and value != "-":
                    inputs[key] = {"path": value, "sha256": digest(root / value)}
            dockq = best.get("dockq") or {}
            records.append({
                "job": job["name"], "kind": kind, "target_group": job["name"].split("_")[0],
                "notes": job["notes"], "inputs": inputs,
                "result_path": str(result_path.relative_to(root)),
                "result_sha256": digest(result_path), "analysis_created": result["created"],
                "n_models": len(models), "best_model_index": best["index"],
                "sampling": {key: params.get(key) for key in ("samples", "seed", "steps", "recycles", "msa")},
                "iptm_nb": best["boltz_pair_iptm"].get("nanobody_in_antigen_frame"),
                "ipsae": (best.get("ipsae", {}).get("max") or {}).get("ipsae"),
                "dockq": dockq.get("headline", dockq.get("global_dockq")),
            })
    return {
        "schema_version": 1,
        "cohort": "current9_wt180mutants_2026-09-19",
        "scope": "Historical structural-confidence snapshot; not a matched WT-mutant experiment or affinity validation.",
        "excluded_historical_group": "8emz (outside current manifest; not proven invalid)",
        "source_tables_sha256": sources,
        "counts": {"WT": sum(r["kind"] == "WT" for r in records),
                   "mutant": sum(r["kind"] == "mutant" for r in records)},
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    snapshot = export_snapshot(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    print(f"Exported {snapshot['counts']} to {args.out}")


if __name__ == "__main__":
    main()
