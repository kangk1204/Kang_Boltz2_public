"""Ensure handoff tables identify the same actual sequences and stored cohort."""

import csv
import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "examples/testset"


def rows(name):
    with (PANEL / name).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sequence(path):
    return "".join(line.strip() for line in (ROOT / path).read_text().splitlines()
                   if line.strip() and not line.startswith(">"))


class CohortTests(unittest.TestCase):
    def test_snapshot_matches_inputs_and_sampling(self):
        snapshot = json.loads((PANEL / "cohort_snapshot_2026-09-19.json").read_text())
        self.assertEqual(snapshot["counts"], {"WT": 9, "mutant": 180})
        for path, expected in snapshot["source_tables_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)
        for record in snapshot["records"]:
            self.assertEqual(record["n_models"], 3 if record["kind"] == "WT" else 1)
            if record["kind"] == "mutant":
                self.assertIsNone(record["dockq"])
            for source in record["inputs"].values():
                self.assertEqual(hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(),
                                 source["sha256"])

    def test_matched_panel_has_real_wt_and_exact_mutation_counts(self):
        panel = rows("batch_matched_wt_mutants.tsv")
        self.assertEqual(len(panel), 189)
        self.assertEqual(len({row["name"] for row in panel}), len(panel))
        controls = {r["name"].split("_")[0]: r for r in panel if r["notes"].startswith("m0:")}
        self.assertEqual(len(controls), 9)
        for row in panel:
            control = controls[row["name"].split("_")[0]]
            self.assertEqual(row["antigen"], control["antigen"])
            self.assertEqual(row["reference"], "-")
            wt, variant = sequence(control["nanobody"]), sequence(row["nanobody"])
            self.assertEqual(len(wt), len(variant))
            expected = int(re.match(r"m(\d+):", row["notes"])[1])
            self.assertEqual(sum(a != b for a, b in zip(wt, variant, strict=True)), expected,
                             row["name"])


if __name__ == "__main__":
    unittest.main()
