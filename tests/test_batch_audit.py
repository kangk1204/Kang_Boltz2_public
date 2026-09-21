from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import batch_report  # noqa: E402


def runtime() -> dict:
    return {
        "schema_version": 1,
        "python": "3.11.0",
        "boltz_version": "2.0.0",
        "torch_version": "2.5.0",
        "boltz_executable_sha256": "abc123",
        "kernel_mode": "default",
    }


def run_params(**overrides) -> dict:
    value = {
        "samples": 2,
        "seed": 42,
        "steps": 200,
        "recycles": 3,
        "msa": "cache",
        "parallel_samples": 1,
        "devices": 1,
        "runtime": runtime(),
    }
    value.update(overrides)
    return value


def msa_policy() -> dict:
    return {
        "schema_version": 1,
        "available": True,
        "requested_msa_policy": "cache",
        "pairing_policy": "cache-unpaired",
        "constraints_hash": "none",
        "constraints_count": 0,
        "delta_eligible": True,
        "chains": [
            {
                "ids": ["A"],
                "sequence_hash": "ag",
                "length": 100,
                "msa": {
                    "class": "cache-unpaired",
                    "delta_eligible": True,
                    "sequence_hash": "ag",
                    "content_sha256": "ag-msa",
                    "pairing": "unpaired",
                },
                "templates_hash": "none",
                "modifications_hash": "none",
                "properties_hash": "none",
                "cyclic": False,
            },
            {
                "ids": ["B"],
                "sequence_hash": "nb",
                "length": 120,
                "msa": {
                    "class": "cache-unpaired",
                    "delta_eligible": True,
                    "sequence_hash": "nb",
                    "content_sha256": "nb-msa",
                    "pairing": "unpaired",
                },
                "templates_hash": "none",
                "modifications_hash": "none",
                "properties_hash": "none",
                "cyclic": False,
            },
        ],
    }


def input_constraints() -> dict:
    return {
        "schema_version": 1,
        "chain_ids": ["A", "B"],
        "chain_lengths": {"A": 100, "B": 120},
        "antigen_chain_ids": ["A"],
        "nanobody_chain_id": "B",
        "target_only": False,
        "yaml_constraints_hash": "none",
        "yaml_constraints_count": 0,
        "templates_modifications_properties": [
            {"ids": ["A"], "templates_hash": "none", "modifications_hash": "none",
             "properties_hash": "none", "cyclic": False},
            {"ids": ["B"], "templates_hash": "none", "modifications_hash": "none",
             "properties_hash": "none", "cyclic": False},
        ],
    }


def analysis_settings(**overrides) -> dict:
    value = {
        "pae_cutoff": 10,
        "dist_cutoff": 15,
        "contact_cutoff": 8,
        "effective_msa_policy": msa_policy(),
        "input_constraints": input_constraints(),
        "analysis_provenance": {
            "schema_version": 2,
            "sources": {"analyze.py": "sha256:a"},
            "packages": {"numpy": "2.0.0"},
        },
        "ranking_policy": "primary_interface.ipTM, primary_interface.ipSAE, -index",
        "score_policy": "primary",
    }
    value.update(overrides)
    return value


def stat(mean: float, n: int = 2, *, partial: bool = False) -> dict:
    return {
        "n": n,
        "mean": mean,
        "std": 0.1 if n > 1 else None,
        "min": mean - 0.1 if n > 1 else mean,
        "max": mean + 0.1 if n > 1 else mean,
        "expected_n": 2,
        "partial": partial,
    }


def metric_stats(base: float = 0.5, *, partial_metric: str | None = None) -> dict:
    return {
        "iptm_nb": stat(base, partial=partial_metric == "iptm_nb"),
        "ipsae": stat(base + 0.1, partial=partial_metric == "ipsae"),
        "pdockq2": stat(base + 0.2, partial=partial_metric == "pdockq2"),
        "cdr3_plddt": stat(base + 70, partial=partial_metric == "cdr3_plddt"),
        "nb_plddt": stat(base + 80, partial=partial_metric == "nb_plddt"),
    }


def row(name: str, *, n_mut: int = 1, stats_base: float = 0.5, **overrides) -> dict:
    value = {
        "job": name,
        "status": "ok",
        "n_mut": n_mut,
        "group": (("A", "ANTIGEN"),),
        "has_nanobody": True,
        "run_params": run_params(),
        "analysis_settings": analysis_settings(),
        "rank_key": "primary",
        "primary_scope": "pair",
        "primary_source": "boltz_pair,official:max",
        "n_models": 2,
        "metric_stats": metric_stats(stats_base),
        "score_sources": {"iptm": ["boltz_pair"], "ipsae": ["official:max"], "consistent": True},
        "iptm_nb": 0.99,
        "ipsae": 0.99,
    }
    value.update(overrides)
    return value


class BatchAuditTests(unittest.TestCase):
    def test_delta_requires_analysis_provenance_schema2(self):
        ctrl = row("g_WT", n_mut=0)
        mutant = row("g_m1")
        mutant["analysis_settings"] = analysis_settings(analysis_provenance=None)
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("analysis provenance", reason)

    def test_delta_requires_nonempty_runtime_provenance(self):
        ctrl = row("g_WT", n_mut=0)
        mutant = row("g_m1", run_params=run_params(runtime={}))
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("runtime provenance", reason)

    def test_delta_rejects_parallel_samples_and_devices_mismatch(self):
        ctrl = row("g_WT", n_mut=0)
        mutant = row("g_m1", run_params=run_params(parallel_samples=2, devices=2))
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("parallel_samples", reason)
        self.assertIn("devices", reason)

    def test_delta_uses_full_ensemble_means_not_best_preview_values(self):
        ctrl = row("g_WT", n_mut=0, stats_base=0.4, iptm_nb=0.1, ipsae=0.1)
        mutant = row("g_m1", stats_base=0.7, iptm_nb=0.99, ipsae=0.99)
        batch_report.assign_control_deltas([ctrl, mutant])
        self.assertTrue(mutant["delta_ok"])
        self.assertAlmostEqual(mutant["d_iptm"], 0.3)
        self.assertAlmostEqual(mutant["d_ipsae"], 0.3)
        self.assertIn("mean deltas", mutant["delta_reason"])

    def test_partial_ensemble_metric_blocks_that_delta(self):
        ctrl = row("g_WT", n_mut=0)
        mutant = row("g_m1", metric_stats=metric_stats(0.6, partial_metric="ipsae"))
        batch_report.assign_control_deltas([ctrl, mutant])
        self.assertIsNone(mutant["d_ipsae"])
        self.assertIn("partially missing ensemble", mutant["delta_reason"])

    def test_delta_exclusion_cards_skip_delta_ok_success_reason_suffix(self):
        ctrl = row("g_WT", n_mut=0)
        mutant = row("g_m1", stats_base=0.7)
        batch_report.assign_control_deltas([ctrl, mutant])
        jobs = {"g_m1": {"kpis": ""}}
        batch_report.append_delta_exclusion_cards([mutant], jobs)
        self.assertNotIn("Δ 계산 제외", jobs["g_m1"]["kpis"])

    def test_each_group_with_control_gets_delta_card_even_if_first_group_lacks_control(self):
        first = row("first_m1", group=(("X", "ANTIGEN"),))
        ctrl = row("second_WT", n_mut=0, group=(("A", "ANTIGEN"),), stats_base=0.4)
        mutant = row("second_m1", group=(("A", "ANTIGEN"),), stats_base=0.7)
        controls = batch_report.assign_control_deltas([first, ctrl, mutant])
        jobs = {"second_m1": {"kpis": ""}}
        for item in [first, ctrl, mutant]:
            d = item.get("d_ipsae")
            if d is None or item.get("n_mut") == 0 or item["job"] not in jobs:
                continue
            c = controls.get(item.get("group"))
            if c is None or c["job"] == item["job"]:
                continue
            jobs[item["job"]]["kpis"] += (
                f'<div class="kpi"><div class="n">ΔipSAE ({c["job"]})</div>'
                f'<div class="v">{d:+.3f}</div></div>'
            )
        self.assertIn("+0.300", jobs["second_m1"]["kpis"])

    def test_mixed_per_model_score_sources_block_delta(self):
        ctrl = row("g_WT", n_mut=0)
        mutant = row(
            "g_m1",
            score_sources={"iptm": ["boltz_pair"], "ipsae": ["builtin-union", "official:max"], "consistent": False},
        )
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("mixed per-model score sources", reason)

    def test_summary_tsv_expands_metric_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.tsv"
            rows = [row("g_m1", stats_base=0.6)]
            batch_report.write_summary_tsv(
                path,
                rows,
                [("job", "Job", "str", None, True), ("ipsae", "ipSAE", "num", 3, True)],
            )
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertIn("ipsae_n", lines[0])
            self.assertIn("ipsae_mean", lines[0])
            self.assertIn("0.7", lines[1])

    def test_target_only_columns_remove_binder_metrics(self):
        cols = batch_report.select_active_columns([{"job": "target", "has_nanobody": False}])
        keys = {c[0] for c in cols}
        for key in ("iptm_nb", "ipsae", "pdockq2", "nb_plddt", "cdr3_plddt", "iface_pae"):
            self.assertNotIn(key, keys)
        self.assertIn("ptm", keys)

    def test_failed_payload_and_rerun_message_are_escaped(self):
        payload = batch_report.failed_detail_payload("<bad>", "x<y", "<script>")
        self.assertIn("&lt;script&gt;", payload["kpis"])
        self.assertEqual(payload["status"], "x<y")
        self.assertEqual(payload["report"], "")
        source = Path(batch_report.__file__).read_text(encoding="utf-8")
        self.assertIn("esc(j.status)", source)
        self.assertIn("esc(job)", source)

    def test_target_only_tsv_does_not_reintroduce_binder_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.tsv"
            rows = [{"job": "target", "has_nanobody": False, "metric_stats": metric_stats(0.6)}]
            batch_report.write_summary_tsv(path, rows, batch_report.select_active_columns(rows))
            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("ipsae_mean", header)
            self.assertNotIn("iptm_nb_mean", header)
            self.assertIn("ptm", header)

    def test_target_only_has_no_binder_delta_exclusion_card(self):
        jobs = {"target": {"kpis": "pTM"}}
        batch_report.append_delta_exclusion_cards([
            {"job": "target", "has_nanobody": False, "delta_reason": "no matched control"}
        ], jobs)
        self.assertEqual(jobs["target"]["kpis"], "pTM")

    def test_target_only_classification_uses_known_rows_and_keeps_all_unknown_conservative(self):
        target_with_unknown_failures = [
            {"job": "target_ok", "has_nanobody": False},
            {"job": "missing_target", "has_nanobody": None, "status": "미완료"},
            {"job": "broken_target", "has_nanobody": None, "status": "불완전"},
        ]
        keys = {c[0] for c in batch_report.select_active_columns(target_with_unknown_failures)}
        self.assertIn("ptm", keys)
        self.assertNotIn("iptm_nb", keys)
        self.assertNotIn("ipsae", keys)

        all_unknown = [
            {"job": "missing_a", "has_nanobody": None, "status": "미완료"},
            {"job": "broken_b", "has_nanobody": None, "status": "불완전"},
        ]
        unknown_keys = {c[0] for c in batch_report.select_active_columns(all_unknown)}
        self.assertIn("iptm_nb", unknown_keys)
        self.assertIn("ipsae", unknown_keys)

    def test_failed_target_only_rows_keep_target_only_columns_and_guide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            jobs_dir = root / "jobs"
            outdir = root / "report"
            manifest.write_text(json.dumps({
                "batch": "target_fail",
                "created": "now",
                "batch_file": "batch.tsv",
                "jobs": [
                    {"name": "target_missing", "target_only": True},
                    {"name": "target_malformed", "target_only": True},
                    {"name": "target_incomplete", "nanobody_chain": None},
                ],
            }), encoding="utf-8")
            for name, contents in (("target_malformed", "{invalid json"), ("target_incomplete", "{}")):
                analysis = jobs_dir / name / "analysis"
                analysis.mkdir(parents=True)
                (analysis / "results.json").write_text(contents, encoding="utf-8")

            argv = [
                "batch_report",
                "--manifest", str(manifest),
                "--jobs-dir", str(jobs_dir),
                "--outdir", str(outdir),
            ]
            with patch.object(sys, "argv", argv):
                got = batch_report.main()

            self.assertEqual(got, 1)
            header = (outdir / "summary.tsv").read_text(encoding="utf-8").splitlines()[0]
            html = (outdir / "index.html").read_text(encoding="utf-8")
            self.assertIn("ptm", header)
            self.assertNotIn("iptm_nb", header)
            self.assertNotIn("ipsae", header)
            self.assertIn("target-only(나노바디 없음) 배치입니다", html)
            self.assertNotIn("기본 정렬: ipTM(나노바디|항원)", html)
            rows, _ = json.JSONDecoder().raw_decode(html.split("const ROWS = ", 1)[1])
            self.assertEqual({r["job"]: r["status"] for r in rows}, {
                "target_missing": "미완료", "target_malformed": "불완전", "target_incomplete": "불완전",
            })

    def test_failed_job_role_uses_structured_metadata_not_notes(self):
        for job, expected in (
            ({"target_only": True}, False),
            ({"has_nanobody": True}, True),
            ({"nanobody_chain": None}, False),
            ({"nanobody_chain": "B"}, True),
            ({"nanobody_len": 0}, False),
            ({"nanobody_len": 110}, True),
            ({"nanobody_file": "nb.fasta"}, True),
            ({"notes": "not target-only"}, None),
            ({}, None),
        ):
            with self.subTest(job=job):
                self.assertIs(batch_report.job_has_nanobody(job), expected)

    def test_mixed_failed_rows_keep_binder_columns_when_any_job_is_binder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            jobs_dir = root / "jobs"
            outdir = root / "report"
            manifest.write_text(json.dumps({
                "batch": "mixed_fail",
                "created": "now",
                "batch_file": "batch.tsv",
                "jobs": [
                    {"name": "target_missing", "target_only": True},
                    {"name": "binder_missing", "nanobody_len": 110},
                ],
            }), encoding="utf-8")

            argv = [
                "batch_report",
                "--manifest", str(manifest),
                "--jobs-dir", str(jobs_dir),
                "--outdir", str(outdir),
            ]
            with patch.object(sys, "argv", argv):
                got = batch_report.main()

            self.assertEqual(got, 0)
            header = (outdir / "summary.tsv").read_text(encoding="utf-8").splitlines()[0]
            html = (outdir / "index.html").read_text(encoding="utf-8")
            self.assertIn("iptm_nb", header)
            self.assertIn("ipsae", header)
            self.assertIn("ipTM(나노바디|항원)", html)
            self.assertNotIn("target-only(나노바디 없음) 배치입니다", html)

    def test_sort_headers_keep_table_semantics_and_aria(self):
        source = Path(batch_report.__file__).read_text(encoding="utf-8")
        self.assertIn("aria-sort", source)
        self.assertIn("<button type=\"button\" class=\"sortbtn\"", source)
        self.assertNotIn("role=\"button\" aria-sort", source)


if __name__ == "__main__":
    unittest.main()
