from __future__ import annotations

# ruff: noqa: E402, I001

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import batch_report
import make_report

EXECUTION_CONTEXT = {
    "parallel_samples": 1, "devices": 1,
    "runtime": {"schema_version": 1, "python": "3.12.14", "boltz_version": "2.2.1",
                "torch_version": "2.7.0", "boltz_executable_sha256": "a" * 64, "kernel_mode": "disabled"},
}


def policy(binder_hash: str = "nbhash", *, eligible: bool = True, antigen_hash: str = "aghash",
           binder_msa_hash: str = "bindermsa", antigen_msa_hash: str = "antigenmsa") -> dict:
    return {
        "schema_version": 1,
        "available": True,
        "requested_msa_policy": "cache",
        "pairing_policy": "cache-unpaired",
        "constraints_hash": "no-constraints",
        "constraints_count": 0,
        "delta_eligible": eligible,
        "chains": [
            {
                "ids": ["A"],
                "sequence_hash": antigen_hash,
                "length": 100,
                "msa": {
                    "class": "cache-unpaired",
                    "delta_eligible": eligible,
                    "path_name": "antigenmsa.csv",
                    "rows": 1000,
                    "content_sha256": antigen_msa_hash,
                    "pairing": "unpaired-per-sequence-cache",
                    "sequence_hash": antigen_hash,
                },
                "templates_hash": "none",
                "modifications_hash": "none",
                "properties_hash": "none",
                "cyclic": False,
            },
            {
                "ids": ["B"],
                "sequence_hash": binder_hash,
                "length": 120,
                "msa": {
                    "class": "cache-unpaired",
                    "delta_eligible": eligible,
                    "path_name": "bindermsa.csv",
                    "rows": 500,
                    "content_sha256": binder_msa_hash,
                    "pairing": "unpaired-per-sequence-cache",
                    "sequence_hash": binder_hash,
                },
                "templates_hash": "none",
                "modifications_hash": "none",
                "properties_hash": "none",
                "cyclic": False,
            },
        ],
    }


def constraints(*, hotspot: str | None = None) -> dict:
    out = {
        "schema_version": 1,
        "chain_ids": ["A", "B"],
        "chain_lengths": {"A": 100, "B": 120},
        "antigen_chain_ids": ["A"],
        "nanobody_chain_id": "B",
        "target_only": False,
        "yaml_constraints_hash": "no-constraints",
        "yaml_constraints_count": 0,
        "templates_modifications_properties": [
            {"ids": ["A"], "templates_hash": "none", "modifications_hash": "none",
             "properties_hash": "none", "cyclic": False},
            {"ids": ["B"], "templates_hash": "none", "modifications_hash": "none",
             "properties_hash": "none", "cyclic": False},
        ],
    }
    if hotspot:
        out["yaml_constraints_hash"] = f"hotspot:{hotspot}"
        out["yaml_constraints_count"] = 1
    return out


def analysis_settings(*, binder_hash: str = "nbhash", eligible: bool = True,
                      antigen_hash: str = "aghash", hotspot: str | None = None,
                      binder_msa_hash: str = "bindermsa",
                      antigen_msa_hash: str = "antigenmsa") -> dict:
    return {
        "pae_cutoff": 10,
        "analysis_provenance": {"schema_version": 2, "sources": {"metrics.py": "b" * 64},
                                "packages": {"numpy": "2.0"}},
        "dist_cutoff": 15,
        "contact_cutoff": 8,
        "effective_msa_policy": policy(binder_hash, eligible=eligible, antigen_hash=antigen_hash,
                                        binder_msa_hash=binder_msa_hash,
                                        antigen_msa_hash=antigen_msa_hash),
        "input_constraints": constraints(hotspot=hotspot),
        "ranking_policy": "primary_interface.ipTM, primary_interface.ipSAE, -index",
        "score_policy": "primary",
    }


MATCHED_ANALYSIS = analysis_settings()


def model(index: int = 0, *, union: bool = False, samples: int = 3) -> dict:
    ipsae = {
        "max": {"ipsae": 0.51, "pdockq2": 0.42, "pdockq": 0.3, "lis": 0.2},
        "source": "official",
    }
    iptm = {"nanobody_in_antigen_frame": 0.72, "antigen_in_nanobody_frame": 0.61}
    iptm_from_pae = {"nanobody_in_antigen_frame": 0.73, "antigen_in_nanobody_frame": 0.6}
    if union:
        ipsae["union"] = {"ipsae": 0.91}
        iptm_from_pae["nanobody_in_antigen_frame"] = 0.88
    return {
        "index": index,
        "stem": f"job_model_{index}",
        "cif": "",
        "confidence_json": "",
        "pae_npz": "",
        "boltz": {"ptm": 0.4, "confidence_score": 0.5},
        "boltz_pair_iptm": iptm,
        "iptm_from_pae": iptm_from_pae,
        "ipsae": ipsae,
        "interface_8A": {"pae_mean_ab": 4.2, "n_contacts": 8},
        "plddt": {"mean": 80.0, "per_chain": {"B": {"mean": 86.0}}},
        "cdr_mean_plddt": {"CDR3": 75.0},
        "dockq": None,
    }


def results(tmp: Path, *, union: bool = False, samples: int = 1) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    pred = tmp / "predictions"
    pred.mkdir()
    stem = "job_model_0"
    cif = pred / f"{stem}.pdb"
    pae = pred / f"pae_{stem}.npz"
    plddt = pred / f"plddt_{stem}.npz"
    conf = pred / f"confidence_{stem}.json"
    cif.write_text(
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 80.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    np.savez(pae, pae=np.array([[1.0]], dtype=float))
    np.savez(plddt, plddt=np.array([80.0], dtype=float))
    conf.write_text(json.dumps({"confidence_score": 0.5, "ptm": 0.4, "iptm": 0.7}), encoding="utf-8")
    m = model(union=union, samples=samples)
    m.update({"stem": stem, "cif": str(cif), "pae_npz": str(pae), "confidence_json": str(conf)})
    return {
        "run_name": "job",
        "run_dir": str(tmp),
        "predictions_dir": str(pred),
        "created": "2026-09-19T00:00:00+09:00",
        "chains": [{"id": "A", "role": "antigen", "n_res": 10, "sequence": "A" * 10},
                   {"id": "C", "role": "antigen", "n_res": 12, "sequence": "C" * 12},
                   {"id": "B", "role": "nanobody", "n_res": 11, "sequence": "B" * 11}],
        "antigen_chain": "A",
        "antigen_chains": ["A", "C"] if union else ["A"],
        "nanobody_chain": "B",
        "role_source": "metadata",
        "settings": {"reference": None, "run_params": {
            "samples": samples, "seed": 42, "steps": 200, "recycles": 3, "msa": "cache"},
            **MATCHED_ANALYSIS},
        "rank_key": "primary_interface_scores",
        "cdr": {"available": False},
        "models": [m],
        "best_model_index": 0,
        "best_model_stem": "job_model_0",
        "warnings": [],
    }


class ReportTests(unittest.TestCase):
    def test_primary_scores_prefer_antigen_union_for_multichain(self):
        m = model(union=True)
        score = make_report.primary_scores(m, {"antigen_chains": ["A", "C"], "nanobody_chain": "B"})
        self.assertEqual(score["scope"], "antigen-union")
        self.assertEqual(score["iptm"], 0.88)
        self.assertEqual(score["ipsae"], 0.91)
        self.assertIn("union", score["iptm_label"])

    def test_multi_antigen_interface_heading_names_union(self):
        with tempfile.TemporaryDirectory() as d:
            data = results(Path(d), union=True)
            entry = data["models"][0]
            entry["interface_8A"]["residues_a"] = [
                {"chain": "C", "resnum": 1, "resname": "GLY", "n_contacts": 2, "plddt": 80.0}
            ]
            rendered = make_report.build_interface_table(data, entry, "antigen")
            self.assertIn("항원 union", rendered)
            self.assertIn("A,C", rendered)

    def test_delta_exclusion_card_is_attached_to_every_ineligible_job(self):
        rows = [
            {"job": "m1", "delta_reason": "mismatched seed"},
            {"job": "m2", "delta_reason": "missing provenance"},
        ]
        jobs = {"m1": {"kpis": ""}, "m2": {"kpis": ""}}
        batch_report.append_delta_exclusion_cards(rows, jobs)
        self.assertIn("mismatched seed", jobs["m1"]["kpis"])
        self.assertIn("missing provenance", jobs["m2"]["kpis"])

    def test_summary_tsv_preserves_delta_exclusion_reason(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.tsv"
            batch_report.write_summary_tsv(
                path,
                [{"job": "m1", "iptm_nb": 0.5, "delta_reason": "mismatched seed"}],
                [("job", "Job", "str", None, True),
                 ("iptm_nb", "ipTM", "num", 3, True)],
            )
            lines = path.read_text(encoding="utf-8").splitlines()
            record = dict(zip(lines[0].split("\t"), lines[1].split("\t"), strict=True))
            self.assertEqual(record["job"], "m1")
            self.assertEqual(record["iptm_nb"], "0.5")
            self.assertEqual(record["delta_reason"], "mismatched seed")

    def test_chain_role_projection_handles_binder_first_order(self):
        value = constraints()
        value["chain_ids"] = ["B", "A"]
        ok, projected = batch_report._project_input_constraints(value)
        self.assertTrue(ok)
        self.assertEqual(projected["antigen_chain_ids"], ("A",))
        ok_msa, projected_msa = batch_report._project_msa_policy(policy(), value)
        self.assertTrue(ok_msa)
        self.assertEqual([item["ids"] for item in projected_msa["antigen_chains"]], [("A",)])

    def test_result_artifact_validation_rejects_existing_incomplete_result(self):
        with tempfile.TemporaryDirectory() as d:
            data = results(Path(d))
            Path(data["models"][0]["cif"]).unlink()
            with self.assertRaisesRegex(ValueError, "missing readable structure|missing CIF"):
                make_report.validate_results(data, Path(d))

    def test_result_validation_rejects_samples_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = results(root, samples=3)
            with self.assertRaisesRegex(ValueError, "expected 3 model"):
                make_report.validate_results(data, root)

    def test_batch_name_validation_accepts_korean_and_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertTrue(batch_report.safe_job_result_path(root, "첫실행").is_relative_to(root))
            for bad in ("", ".", "..", "../x", "a/b", "/tmp/x"):
                with self.assertRaises(ValueError):
                    batch_report.safe_job_result_path(root, bad)

    def test_batch_collect_uses_primary_scores(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / "out"
            out.mkdir()
            data = results(root, union=True)
            rp = root / "analysis" / "results.json"
            rp.parent.mkdir()
            rp.write_text(json.dumps(data), encoding="utf-8")
            payload = batch_report.collect(rp, {"name": "첫실행"}, "../첫실행/report/index.html", "figures", out, embed=False)
            self.assertEqual(payload["row"]["primary_scope"], "antigen-union")
            self.assertEqual(payload["row"]["iptm_nb"], 0.88)
            self.assertEqual(payload["row"]["ipsae"], 0.91)
            self.assertEqual(payload["row"]["ipsae_src"], "내장 union")
            self.assertFalse(payload["joins"]["cif"])

    def test_no_embed_viewer_message_is_explicit(self):
        source = Path(batch_report.__file__).read_text(encoding="utf-8")
        self.assertIn("--no-embed", source)
        self.assertIn("개별 리포트", source)
        self.assertIn("3D 구조가 포함되지 않았습니다", source)

    def test_delta_requires_matched_sampling_policy(self):
        ctrl = {"status": "ok", "group": ("g",), "job": "g_WT", "iptm_nb": 0.5,
                "ipsae": 0.5, "run_params": {"samples": 3, "seed": 42, "steps": 200,
                                              "recycles": 3, "msa": "cache"},
                "analysis_settings": dict(MATCHED_ANALYSIS),
                "rank_key": "primary_interface_scores", "primary_scope": "pair", "n_models": 3}
        mutant = {"status": "ok", "group": ("g",), "job": "g_m1", "iptm_nb": 0.6,
                  "ipsae": 0.7, "run_params": {"samples": 1, "seed": 42, "steps": 200,
                                                "recycles": 3, "msa": "cache"},
                  "analysis_settings": dict(MATCHED_ANALYSIS),
                  "rank_key": "primary_interface_scores", "primary_scope": "pair", "n_models": 1}
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("samples", reason)

    def test_delta_requires_matched_msa_subsample(self):
        common = {"status": "ok", "group": (("A", "ANTIGEN"),),
                  "run_params": {"samples": 1, "seed": 42, "steps": 200, "recycles": 3,
                                 "msa": "cache", "msa_subsample": 512, **EXECUTION_CONTEXT},
                  "analysis_settings": dict(MATCHED_ANALYSIS),
                  "rank_key": "primary", "primary_scope": "pair",
                  "primary_source": "boltz_pair", "n_models": 1}
        ctrl = dict(common, job="WT")
        mutant = dict(common, job="m1")
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertTrue(ok, reason)

        mutant_params = dict(common["run_params"], msa_subsample=0)
        mutant = dict(common, job="m1", run_params=mutant_params)
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("msa_subsample", reason)

    def test_delta_treats_missing_msa_subsample_as_zero(self):
        ctrl_params = {"samples": 1, "seed": 42, "steps": 200, "recycles": 3,
                       "msa": "cache", **EXECUTION_CONTEXT}
        mutant_params = dict(ctrl_params, msa_subsample=0)
        common = {"status": "ok", "group": (("A", "ANTIGEN"),),
                  "analysis_settings": dict(MATCHED_ANALYSIS),
                  "rank_key": "primary", "primary_scope": "pair",
                  "primary_source": "boltz_pair", "n_models": 1}
        ctrl = dict(common, job="WT", run_params=ctrl_params)
        mutant = dict(common, job="m1", run_params=mutant_params)
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertTrue(ok, reason)

    def test_delta_rejects_invalid_msa_subsample(self):
        common_params = {"samples": 1, "seed": 42, "steps": 200, "recycles": 3,
                         "msa": "cache", **EXECUTION_CONTEXT}
        common = {"status": "ok", "group": (("A", "ANTIGEN"),),
                  "analysis_settings": dict(MATCHED_ANALYSIS),
                  "rank_key": "primary", "primary_scope": "pair",
                  "primary_source": "boltz_pair", "n_models": 1}
        ctrl = dict(common, job="WT", run_params=dict(common_params, msa_subsample=0))
        for value in (True, -1, 1.5, "512"):
            with self.subTest(value=value):
                mutant = dict(common, job="m1", run_params=dict(common_params, msa_subsample=value))
                ok, reason = batch_report.delta_eligibility(mutant, ctrl)
                self.assertFalse(ok)
                self.assertIn("invalid run param: msa_subsample", reason)

    def test_delta_rejects_missing_provenance(self):
        ctrl = {"status": "ok", "group": ("g",), "job": "g_WT", "run_params": {},
                "rank_key": "primary_interface_scores", "n_models": 1}
        mutant = {"status": "ok", "group": ("g",), "job": "g_m1", "run_params": {},
                  "rank_key": "primary_interface_scores", "n_models": 1}
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("missing run params", reason)

    def test_delta_rejects_missing_antigen_sequence_group(self):
        common = {"status": "ok", "run_params": {"samples": 1, "seed": 42, "steps": 200,
                                                 "recycles": 3, "msa": "cache"},
                  "analysis_settings": dict(MATCHED_ANALYSIS),
                  "rank_key": "primary", "primary_scope": "pair",
                  "primary_source": "boltz_pair", "n_models": 1}
        ctrl = dict(common, group=("missing-antigen-sequence", (10,)), job="WT")
        mutant = dict(common, group=("missing-antigen-sequence", (10,)), job="m1")
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("missing antigen sequence", reason)


    def test_delta_permits_real_schema_wt_mutant_with_binder_msa_variation(self):
        ctrl_settings = analysis_settings(binder_hash="wt-binder", binder_msa_hash="wt-binder-msa")
        mut_settings = analysis_settings(binder_hash="mut-binder", binder_msa_hash="mut-binder-msa")
        common = {"status": "ok", "group": (("A", "ANTIGEN"),),
                  "run_params": {"samples": 1, "seed": 42, "steps": 200, "recycles": 3, "msa": "cache", **EXECUTION_CONTEXT},
                  "rank_key": "primary", "primary_scope": "pair",
                  "primary_source": "boltz_pair", "n_models": 1}
        ctrl = dict(common, job="WT", analysis_settings=ctrl_settings)
        mutant = dict(common, job="m1", analysis_settings=mut_settings)
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertTrue(ok, reason)

    def test_delta_false_false_policy_is_blocked(self):
        common = {"status": "ok", "group": (("A", "ANTIGEN"),),
                  "run_params": {"samples": 1, "seed": 42, "steps": 200, "recycles": 3, "msa": "cache", **EXECUTION_CONTEXT},
                  "rank_key": "primary", "primary_scope": "pair",
                  "primary_source": "boltz_pair", "n_models": 1}
        ctrl = dict(common, job="WT", analysis_settings=analysis_settings(eligible=False))
        mutant = dict(common, job="m1", analysis_settings=analysis_settings(eligible=False))
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("MSA delta not eligible", reason)

    def test_delta_rejects_antigen_msa_content_mismatch(self):
        ctrl_settings = analysis_settings(antigen_msa_hash="ag-msa-a")
        mut_settings = analysis_settings(antigen_msa_hash="ag-msa-b")
        common = {"status": "ok", "group": (("A", "ANTIGEN"),),
                  "run_params": {"samples": 1, "seed": 42, "steps": 200, "recycles": 3, "msa": "cache", **EXECUTION_CONTEXT},
                  "rank_key": "primary", "primary_scope": "pair",
                  "primary_source": "boltz_pair", "n_models": 1}
        ctrl = dict(common, job="WT", analysis_settings=ctrl_settings)
        mutant = dict(common, job="m1", analysis_settings=mut_settings)
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("effective_msa_policy", reason)

    def test_delta_rejects_effective_msa_policy_mismatch(self):
        ctrl = {"status": "ok", "group": ("g",), "job": "WT",
                "run_params": {"samples": 1, "seed": 42, "steps": 200, "recycles": 3, "msa": "cache", **EXECUTION_CONTEXT},
                "analysis_settings": dict(MATCHED_ANALYSIS), "rank_key": "primary",
                "primary_scope": "pair", "primary_source": "boltz_pair", "n_models": 1}
        altered = dict(MATCHED_ANALYSIS)
        altered["effective_msa_policy"] = "cache-with-custom-empty-chain"
        mutant = dict(ctrl, job="m1", analysis_settings=altered)
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("MSA delta not eligible", reason)

    def test_delta_rejects_input_constraint_mismatch(self):
        ctrl = {"status": "ok", "group": ("g",), "job": "WT",
                "run_params": {"samples": 1, "seed": 42, "steps": 200, "recycles": 3, "msa": "cache", **EXECUTION_CONTEXT},
                "analysis_settings": dict(MATCHED_ANALYSIS), "rank_key": "primary",
                "primary_scope": "pair", "primary_source": "boltz_pair", "n_models": 1}
        altered = dict(MATCHED_ANALYSIS)
        altered["input_constraints"] = {"target_only": False, "chain_ids": ["A", "B"], "hotspot": "A:10"}
        mutant = dict(ctrl, job="m1", analysis_settings=altered)
        ok, reason = batch_report.delta_eligibility(mutant, ctrl)
        self.assertFalse(ok)
        self.assertIn("antigen chain ids", reason)

    def test_batch_group_uses_antigen_sequences_not_lengths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = results(root / "a")
            b = results(root / "b")
            b["chains"][0]["sequence"] = "Z" * len(a["chains"][0]["sequence"])
            self.assertNotEqual(batch_report.comparison_group(a, {"target_lens": [10, 12]}),
                                batch_report.comparison_group(b, {"target_lens": [10, 12]}))


if __name__ == "__main__":
    unittest.main()
