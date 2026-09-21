import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze

ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "scripts" / "analyze.py"


def _args(**overrides):
    base = {
        "reference": None,
        "dockq_mapping": None,
        "dockq_exe": None,
        "dockq_allowed_mismatches": 0,
        "pae_cutoff": 10.0,
        "dist_cutoff": 15.0,
        "ipsae_script": "scripts/vendor/ipsae_official.py",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _token(chain, resnum, aa="A", icode="", xyz=(0.0, 0.0, 0.0)):
    coord = np.array(xyz, dtype=float)
    return {
        "chain": chain,
        "resnum": resnum,
        "icode": icode,
        "resname": "ALA" if aa == "A" else "GLY",
        "aa": aa,
        "ca": coord,
        "cb": coord,
        "plddt": 80.0,
    }


def _pdb_text(chains):
    lines = []
    serial = 1
    x = 0.0
    for chain, residues in chains.items():
        for resi, resname in enumerate(residues, start=1):
            for atom, dx in (("N", 0.0), ("CA", 1.0), ("C", 1.5)):
                element = atom[0]
                lines.append(
                    f"ATOM  {serial:5d} {atom:>4s} {resname:>3s} {chain:1s}{resi:4d}"
                    f"    {x + dx:8.3f}{0.0:8.3f}{0.0:8.3f}"
                    f"  1.00{80.00:6.2f}           {element:>2s}"
                )
                serial += 1
            x += 3.8
    lines.append("END")
    return "\n".join(lines) + "\n"


def _write_prediction(run_dir, name="job"):
    pred = run_dir / f"boltz_results_{name}" / "predictions" / name
    pred.mkdir(parents=True)
    stem = f"{name}_model_0"
    conf = {
        "confidence_score": 0.5,
        "ptm": 0.6,
        "iptm": 0.7,
        "pair_chains_iptm": {"0": {"0": 0.0, "1": 0.8}, "1": {"0": 0.7, "1": 0.0}},
    }
    (pred / f"confidence_{stem}.json").write_text(json.dumps(conf), encoding="utf-8")
    (pred / f"{stem}.pdb").write_text(
        _pdb_text({"A": ["ALA"], "B": ["GLY"]}), encoding="utf-8"
    )
    np.savez(pred / f"pae_{stem}.npz", pae=np.ones((2, 2), dtype=float))
    np.savez(pred / f"plddt_{stem}.npz", plddt=np.full(2, 80.0, dtype=float))
    return {
        "index": 0,
        "stem": stem,
        "confidence": pred / f"confidence_{stem}.json",
        "cif": pred / f"{stem}.pdb",
        "pae": pred / f"pae_{stem}.npz",
        "plddt": pred / f"plddt_{stem}.npz",
    }


def _run_analyze(run_dir, *extra):
    cmd = [sys.executable, str(ANALYZE), "--run-dir", str(run_dir), "--no-figures", *extra]
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)


class AnalysisAuditTests(unittest.TestCase):
    def test_dockq_uses_only_explicit_allowed_mismatch_count(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "ref.pdb"
            ref.write_text("END\n", encoding="utf-8")
            calls = []
            original = analyze.run_dockq

            def fake_run_dockq(*_args, **kwargs):
                calls.append(kwargs.get("allowed_mismatches"))
                return {"error": "identical corresponding chain required"}

            try:
                analyze.run_dockq = fake_run_dockq
                got = analyze._run_dockq_block(
                    {"cif": ref}, _args(reference=str(ref)), "A", Path(td), ["A"], "B"
                )
            finally:
                analyze.run_dockq = original

            self.assertEqual(calls, [0])
            self.assertIn("error", got)

    def test_dockq_explicit_mismatch_count_is_forwarded_once(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "ref.pdb"
            ref.write_text("END\n", encoding="utf-8")
            calls = []
            original = analyze.run_dockq

            def fake_run_dockq(*_args, **kwargs):
                calls.append(kwargs.get("allowed_mismatches"))
                return {"error": "still failed"}

            try:
                analyze.run_dockq = fake_run_dockq
                analyze._run_dockq_block(
                    {"cif": ref},
                    _args(reference=str(ref), dockq_allowed_mismatches=5),
                    "A",
                    Path(td),
                    ["A"],
                    "B",
                )
            finally:
                analyze.run_dockq = original

            self.assertEqual(calls, [5])

    def test_interface_union_preserves_icode_and_deduplicates_on_stable_residue_key(self):
        tokens = [
            _token("A", 1, "A", "", (0.0, 0.0, 0.0)),
            _token("C", 1, "A", "", (0.0, 0.0, 0.0)),
            _token("B", 10, "G", "A", (1.0, 0.0, 0.0)),
            _token("B", 10, "G", "B", (1.5, 0.0, 0.0)),
        ]
        plddt = np.array([80.0, 81.0, 82.0, 83.0])
        pae = np.ones((4, 4), dtype=float)

        report = analyze.interface_report_union(tokens, plddt, pae, ["A", "C"], "B", cutoff=8.0)

        by_icode = {row["icode"]: row for row in report["residues_b"]}
        self.assertEqual(set(by_icode), {"A", "B"})
        self.assertEqual(by_icode["A"]["aa"], "G")
        self.assertEqual(by_icode["B"]["aa"], "G")
        self.assertEqual(by_icode["A"]["n_contacts"], 2)
        self.assertEqual(by_icode["B"]["n_contacts"], 2)

    def test_token_identity_mismatch_rejects_ordered_icode_difference(self):
        ref = [_token("A", 10, "A", "")]
        model = [_token("A", 10, "A", "B")]

        mismatch = analyze.token_identity_mismatch(ref, model)

        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["first_mismatch_index"], 0)
        self.assertEqual(mismatch["reference"], ["A", 10, "", "A"])
        self.assertEqual(mismatch["model"], ["A", 10, "B", "A"])

    def test_schema_ineligible_input_fails_closed_even_when_dimensions_match(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "ligandcase"
            run.mkdir()
            model = _write_prediction(run, "ligandcase")
            yaml_path = run / "input.yaml"
            yaml_path.write_text(
                "version: 1\n"
                "sequences:\n"
                "  - protein: {id: A, sequence: A, msa: empty}\n"
                "  - ligand: {id: L, ccd: ATP}\n"
                "  - protein: {id: B, sequence: G, msa: empty}\n",
                encoding="utf-8",
            )
            provenance = analyze.yaml_input_provenance(yaml_path, "empty")
            tokens = [_token("A", 1, "A"), _token("B", 1, "G")]

            entry = analyze.analyze_model(
                model,
                tokens,
                ("A", "B", ["A"]),
                _args(),
                {"available": False},
                run / "analysis",
                input_provenance=provenance,
            )

            self.assertTrue(entry["token_dims_ok"])
            self.assertFalse(entry["pae_metrics_applicable"])
            self.assertEqual(entry["ipsae"]["source"], "not-run")
            self.assertEqual(entry["chain_pairs"], [])
            self.assertEqual(entry["plddt"]["per_chain"], {})
            self.assertIn("ligand", "; ".join(entry["warnings"]))

    def test_analyze_model_reordered_tokens_fail_closed_before_indexed_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "reordered"
            run.mkdir()
            model = _write_prediction(run, "reordered")
            reference_tokens = [_token("A", 1, "A"), _token("B", 1, "G")]
            reordered_tokens = [_token("B", 1, "G"), _token("A", 1, "A")]
            original = analyze.read_structure_tokens

            try:
                analyze.read_structure_tokens = lambda _path: reordered_tokens
                entry = analyze.analyze_model(
                    model,
                    reference_tokens,
                    ("A", "B", ["A"]),
                    _args(),
                    {"available": False},
                    run / "analysis",
                    input_provenance={"available": True, "pae_metrics_eligible": True},
                )
            finally:
                analyze.read_structure_tokens = original

            self.assertTrue(entry["token_dims_ok"])
            self.assertFalse(entry["token_identity_ok"])
            self.assertFalse(entry["pae_metrics_applicable"])
            self.assertEqual(entry["plddt"]["per_chain"], {})
            self.assertEqual(entry["ipsae"]["source"], "not-run")
            self.assertEqual(entry["chain_pairs"], [])

    def test_results_declare_validation_limitations_without_numeric_claims(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "target"
            run.mkdir()
            _write_prediction(run, "target")

            proc = _run_analyze(run, "--no-nanobody")

            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            results = json.loads((run / "analysis" / "results.json").read_text(encoding="utf-8"))
            limits = results["validation_limitations"]
            self.assertEqual(limits["schema_version"], 1)
            self.assertIn("experimental binding affinity", limits["not_validated"])
            self.assertIn("not emitted as statistics", limits["policy"])

    def test_pae_title_does_not_overlap_single_chain_label(self):
        from unittest.mock import patch

        import figures

        with tempfile.TemporaryDirectory() as td:
            with patch.object(figures.plt, "close"):
                figures.pae_heatmap(np.zeros((99, 99)), np.array(["A"] * 99),
                                    Path(td) / "pae.png", chain_labels={"A": "antigen"})
                fig = figures.plt.gcf()
            try:
                fig.canvas.draw()
                ax = fig.axes[0]
                renderer = fig.canvas.get_renderer()
                title = ax.title.get_window_extent(renderer)
                top_label = ax.texts[0].get_window_extent(renderer)
                self.assertGreater(title.y0, top_label.y1)
            finally:
                figures.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
