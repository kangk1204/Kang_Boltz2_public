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
import metrics
import msa_cache

ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "scripts" / "analyze.py"


def _pdb_text(chains):
    lines = []
    serial = 1
    x = 10.0
    for chain, residues in chains.items():
        for resi, resname in enumerate(residues, start=1):
            for atom, dx in (("N", 0.0), ("CA", 1.0), ("C", 1.5)):
                element = atom[0]
                lines.append(
                    f"ATOM  {serial:5d} {atom:>4s} {resname:>3s} {chain:1s}{resi:4d}"
                    f"    {x + dx:8.3f}{13.207:8.3f}{10.000:8.3f}"
                    f"  1.00{80.00:6.2f}           {element:>2s}"
                )
                serial += 1
            x += 3.8
    lines.append("END")
    return "\n".join(lines) + "\n"


def _write_fake_prediction(run_dir, name="job", chains=None, samples=1):
    chains = chains or {"A": ["ALA"]}
    pred = run_dir / f"boltz_results_{name}" / "predictions" / name
    pred.mkdir(parents=True)
    n_tokens = sum(len(v) for v in chains.values())
    for idx in range(samples):
        stem = f"{name}_model_{idx}"
        conf = {
            "confidence_score": 0.5 + idx * 0.01,
            "ptm": 0.6 + idx * 0.01,
            "iptm": 0.7 + idx * 0.01,
            "pair_chains_iptm": {
                str(i): {str(j): 0.8 - abs(i - j) * 0.1 for j in range(len(chains))}
                for i in range(len(chains))
            },
        }
        (pred / f"confidence_{stem}.json").write_text(json.dumps(conf), encoding="utf-8")
        (pred / f"{stem}.pdb").write_text(_pdb_text(chains), encoding="utf-8")
        np.savez(pred / f"pae_{stem}.npz", pae=np.ones((n_tokens, n_tokens), dtype=float))
        np.savez(pred / f"plddt_{stem}.npz", plddt=np.full(n_tokens, 80.0, dtype=float))
    return pred


def _run_analyze(run_dir, *extra):
    cmd = [sys.executable, str(ANALYZE), "--run-dir", str(run_dir), "--no-figures", *extra]
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)


def _write_yaml(path, msa=None, hotspot=None):
    lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        "      id: A",
        "      sequence: A",
    ]
    if msa is not None:
        lines.append(f"      msa: {msa}")
    lines += [
        "  - protein:",
        "      id: B",
        "      sequence: G",
    ]
    if msa is not None:
        lines.append(f"      msa: {msa}")
    if hotspot:
        lines += [
            "constraints:",
            "  - pocket:",
            "      binder: B",
            f"      contacts: [[A, {hotspot}]]",
            "      max_distance: 6.0",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _args(**overrides):
    base = {
        "no_nanobody": False,
        "nanobody_chain": None,
        "antigen_chain": None,
        "metadata": None,
        "yaml": None,
        "boltz_dir": None,
        "reference": None,
        "dockq_mapping": None,
        "dockq_allowed_mismatches": 0,
        "pae_cutoff": 10.0,
        "dist_cutoff": 15.0,
        "ipsae_script": "scripts/vendor/ipsae_official.py",
        "no_figures": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class AnalysisContractTests(unittest.TestCase):
    def test_model_ranking_keeps_finite_zero_ahead_of_missing_primary_score(self):
        missing = {"index": 0, "primary_interface": {"iptm": None, "ipsae": 0.99}}
        observed = {"index": 1, "primary_interface": {"iptm": 0.0, "ipsae": 0.0}}
        self.assertGreater(analyze.model_rank_key(observed, False),
                           analyze.model_rank_key(missing, False))

    def test_primary_interface_scores_pair_uses_native_pair_scores(self):
        model = {
            "antigen_chains": ["A"],
            "boltz_pair_iptm": {"nanobody_in_antigen_frame": 0.81},
            "iptm_from_pae": {"nanobody_in_antigen_frame": 0.71},
            "ipsae": {"max": {"ipsae": 0.62}, "union": {"ipsae": 0.44}, "source": "official"},
        }

        got = metrics.primary_interface_scores(model)

        self.assertEqual(got["scope"], "pair")
        self.assertEqual(got["iptm_source"], "boltz_pair_iptm")
        self.assertAlmostEqual(got["iptm"], 0.81)
        self.assertEqual(got["ipsae_source"], "official:max")
        self.assertAlmostEqual(got["ipsae"], 0.62)

    def test_primary_interface_scores_multichain_uses_union_scores(self):
        model = {
            "antigen_chains": ["A", "C", "D"],
            "boltz_pair_iptm": {"nanobody_in_antigen_frame": 0.91},
            "iptm_from_pae": {"nanobody_in_antigen_frame": 0.73},
            "ipsae": {"max": {"ipsae": 0.52}, "union": {"ipsae": 0.67}, "source": "official"},
        }

        got = metrics.primary_interface_scores(model)

        self.assertEqual(got["scope"], "antigen-union")
        self.assertEqual(got["iptm_source"], "pae_union_iptm")
        self.assertAlmostEqual(got["iptm"], 0.73)
        self.assertEqual(got["ipsae_source"], "builtin-union")
        self.assertAlmostEqual(got["ipsae"], 0.67)

    def test_primary_interface_scores_labels_fallback_sources(self):
        model = {
            "antigen_chains": ["A"],
            "boltz_pair_iptm": {"nanobody_in_antigen_frame": None},
            "iptm_from_pae": {"nanobody_in_antigen_frame": 0.71},
            "ipsae": {"max": {"ipsae": 0.62}, "source": "builtin-fallback"},
        }

        got = metrics.primary_interface_scores(model)

        self.assertEqual(got["iptm_source"], "pae_pair_iptm")
        self.assertIn("PAE pair", got["iptm_label"])
        self.assertEqual(got["ipsae_source"], "builtin-fallback:max")
        self.assertIn("built-in fallback", got["ipsae_label"])

    def test_ambiguous_roles_require_explicit_choice(self):
        tokens = [
            {"chain": "A", "aa": "M"},
            {"chain": "A", "aa": "A"},
            {"chain": "B", "aa": "G"},
            {"chain": "B", "aa": "S"},
        ]

        with self.assertRaises(SystemExit) as cm:
            analyze.resolve_roles(tokens, None, _args())

        self.assertIn("--nanobody-chain", str(cm.exception))

    def test_nanobody_autodetect_tie_fails(self):
        seq = "C" + "A" * 30 + "WGQGT" + "A" * 60 + "C"
        tokens = (
            [{"chain": "A", "aa": aa} for aa in seq]
            + [{"chain": "B", "aa": aa} for aa in seq]
        )

        self.assertIsNone(analyze.detect_nanobody_chain(tokens))

    def test_restore_settings_rebases_local_paths_and_rejects_source_absolute(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_run = root / "old" / "run1"
            new_run = root / "new" / "run1"
            old_run.mkdir(parents=True)
            new_run.mkdir(parents=True)
            (new_run / "boltz_results_run1").mkdir()
            (new_run / "reference.cif").write_text("data_ref\n", encoding="utf-8")
            saved_results = new_run / "analysis" / "results.json"
            saved_results.parent.mkdir()
            saved_results.write_text(json.dumps({
                "run_dir": str(old_run),
                "settings": {
                    "boltz_dir": str(old_run / "boltz_results_run1"),
                    "reference": str(old_run / "reference.cif"),
                    "metadata": None,
                    "yaml": None,
                    "no_nanobody": True,
                    "nanobody_chain": None,
                    "antigen_chain": "A",
                    "dockq_mapping": "A:A",
                    "dockq_allowed_mismatches": 5,
                    "pae_cutoff": 9.0,
                    "dist_cutoff": 14.0,
                    "no_figures": True,
                },
            }), encoding="utf-8")

            args = _args(restore_settings=saved_results)
            analyze.apply_restored_settings(args, new_run, {"--restore-settings"})

            self.assertEqual(args.boltz_dir, new_run / "boltz_results_run1")
            self.assertEqual(args.reference, str(new_run / "reference.cif"))
            self.assertTrue(args.no_nanobody)
            self.assertEqual(args.antigen_chain, "A")
            self.assertEqual(args.dockq_mapping, "A:A")
            self.assertEqual(args.dockq_allowed_mismatches, 5)
            self.assertEqual(args.pae_cutoff, 9.0)
            self.assertEqual(args.dist_cutoff, 14.0)
            self.assertTrue(args.no_figures)

    def test_restore_settings_missing_saved_reference_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_run = root / "old" / "run1"
            new_run = root / "new" / "run1"
            old_run.mkdir(parents=True)
            new_run.mkdir(parents=True)
            saved_results = new_run / "results.json"
            saved_results.write_text(json.dumps({
                "run_dir": str(old_run),
                "settings": {"reference": str(old_run / "reference.cif")},
            }), encoding="utf-8")

            with self.assertRaises(SystemExit) as cm:
                analyze.apply_restored_settings(
                    _args(restore_settings=saved_results), new_run, {"--restore-settings"}
                )

            self.assertIn("reference", str(cm.exception))

    def test_chain_roles_mark_all_antigen_chains(self):
        tokens = [
            {"chain": "A", "aa": "M"},
            {"chain": "C", "aa": "G"},
            {"chain": "D", "aa": "S"},
            {"chain": "B", "aa": "Q"},
        ]
        antigen_chains = ["A", "C", "D"]
        nanobody_chain = "B"

        roles = analyze.role_map(antigen_chains, nanobody_chain)
        chains = [
            {"id": ch, "role": roles.get(ch, "other")}
            for ch in metrics.chain_lengths(tokens)
        ]

        self.assertEqual(
            {c["id"]: c["role"] for c in chains},
            {"A": "antigen", "C": "antigen", "D": "antigen", "B": "nanobody"},
        )

    def test_antigen_chains_option_excludes_vl_from_antigen(self):
        # Fv(VH+VL): 항원 A, VH B, VL C. --antigen-chains A 로 VL 이 항원에서 빠져야 한다.
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "fv"
            run.mkdir()
            _write_fake_prediction(run, name="fv",
                                   chains={"A": ["GLY", "GLY"], "B": ["ALA", "ALA"],
                                           "C": ["SER", "SER"]})
            ok = _run_analyze(run, "--nanobody-chain", "B", "--antigen-chain", "A",
                              "--antigen-chains", "A")
            self.assertEqual(ok.returncode, 0, ok.stderr)
            r = json.loads((run / "analysis" / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(r["nanobody_chain"], "B")
            self.assertEqual(r["antigen_chains"], ["A"])
            self.assertEqual(r["settings"]["antigen_chains"], ["A"])
            constraints = r["settings"]["input_constraints"]
            self.assertEqual(constraints["antigen_chain_ids"], ["A"])
            self.assertEqual(constraints["nanobody_chain_id"], "B")
            roles = {c["id"]: c["role"] for c in r["chains"]}
            self.assertEqual(roles["C"], "other")

            # 옵션이 없으면 VL(C)도 항원 집합에 들어간다(기존 동작)
            plain = _run_analyze(run, "--nanobody-chain", "B", "--antigen-chain", "A")
            self.assertEqual(plain.returncode, 0, plain.stderr)
            r2 = json.loads((run / "analysis" / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(r2["antigen_chains"], ["A", "C"])

    def test_process_analyzer_rejects_partial_outputs_from_run_params(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "partial"
            run.mkdir()
            pred = run / "boltz_results_partial" / "predictions" / "partial"
            pred.mkdir(parents=True)
            (pred / "confidence_partial_model_0.json").write_text("{}", encoding="utf-8")
            (run / ".run_params.json").write_text(json.dumps({"samples": 3}), encoding="utf-8")

            proc = _run_analyze(run, "--no-nanobody")

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("expected 3 model", proc.stderr + proc.stdout)

    def test_process_restore_target_only_copy_ignores_external_active_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = root / "old" / "target_model"
            new = root / "new" / "target_model"
            old.mkdir(parents=True)
            new.mkdir(parents=True)
            _write_fake_prediction(old, name="target_model", chains={"A": ["ALA"]})
            _write_fake_prediction(new, name="target_model", chains={"A": ["ALA"]})
            (new / ".active_boltz_dir").write_text(str(old / "boltz_results_target_model"), encoding="utf-8")
            saved = new / "analysis" / "results.json"
            saved.parent.mkdir()
            saved.write_text(json.dumps({
                "run_dir": str(old),
                "antigen_chain": "A",
                "antigen_chains": ["A"],
                "nanobody_chain": None,
                "settings": {"analysis_contract_version": 1},
            }), encoding="utf-8")

            proc = _run_analyze(new, "--restore-settings", str(saved))

            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            result = json.loads((new / "analysis" / "results.json").read_text(encoding="utf-8"))
            self.assertIsNone(result["nanobody_chain"])
            self.assertEqual(result["role_source"], "target-only(metadata)")
            self.assertIn(str(new), result["predictions_dir"])

    def test_process_reference_snapshot_is_stable_across_restore_runs(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "refcase"
            run.mkdir()
            _write_fake_prediction(run, name="refcase", chains={"A": ["ALA"]})
            ref = run / "reference.pdb"
            ref.write_text(_pdb_text({"A": ["ALA"]}), encoding="utf-8")

            first = _run_analyze(run, "--no-nanobody", "--reference", str(ref))
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            saved = run / "analysis" / "results.json"
            second = _run_analyze(run, "--restore-settings", str(saved))
            self.assertEqual(second.returncode, 0, second.stderr + second.stdout)

            inputs = sorted(p.name for p in (run / "analysis" / "inputs").iterdir())
            self.assertEqual(inputs.count("reference.pdb"), 1)
            self.assertFalse(any(name.startswith("reference_reference") for name in inputs))

    def test_process_explicit_nanobody_overrides_saved_target_only_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "override"
            run.mkdir()
            _write_fake_prediction(run, name="override", chains={"A": ["ALA"], "B": ["GLY"]})
            meta = run / "meta.json"
            meta.write_text(json.dumps({
                "target_only": True,
                "nanobody_chain": None,
                "antigen_chain": "A",
                "chains": [{"id": "A", "role": "target"}, {"id": "B", "role": "target"}],
            }), encoding="utf-8")
            saved = run / "analysis" / "results.json"
            saved.parent.mkdir()
            saved.write_text(json.dumps({
                "run_dir": str(run),
                "nanobody_chain": None,
                "antigen_chain": "A",
                "antigen_chains": ["A"],
                "settings": {
                    "analysis_contract_version": 1,
                    "metadata_snapshot": str(meta),
                    "no_nanobody": True,
                    "nanobody_chain": None,
                    "antigen_chain": "A",
                },
            }), encoding="utf-8")

            proc = _run_analyze(run, "--restore-settings", str(saved), "--nanobody-chain", "B")

            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            result = json.loads((run / "analysis" / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["nanobody_chain"], "B")
            self.assertFalse(result["settings"]["target_only"])

    def test_process_yaml_provenance_detects_hotspot_difference(self):
        with tempfile.TemporaryDirectory() as td:
            run1 = Path(td) / "hot1"
            run2 = Path(td) / "hot2"
            for run, hotspot in ((run1, 1), (run2, 2)):
                run.mkdir()
                _write_fake_prediction(run, name=run.name, chains={"A": ["ALA"], "B": ["GLY"]})
                yml = run / "input.yaml"
                _write_yaml(yml, hotspot=hotspot)
                proc = _run_analyze(run, "--yaml", str(yml), "--nanobody-chain", "B")
                self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)

            r1 = json.loads((run1 / "analysis" / "results.json").read_text(encoding="utf-8"))
            r2 = json.loads((run2 / "analysis" / "results.json").read_text(encoding="utf-8"))

            self.assertNotEqual(
                r1["settings"]["input_constraints"]["yaml_constraints_hash"],
                r2["settings"]["input_constraints"]["yaml_constraints_hash"],
            )
            self.assertNotIn("role_source", r1["settings"]["input_constraints"])

    def test_process_yaml_provenance_declared_cache_but_actual_empty(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "msacase"
            run.mkdir()
            _write_fake_prediction(run, name="msacase", chains={"A": ["ALA"], "B": ["GLY"]})
            (run / ".run_params.json").write_text(json.dumps({"samples": 1, "msa": "cache"}),
                                                  encoding="utf-8")
            yml = run / "input.yaml"
            _write_yaml(yml, msa="empty")

            proc = _run_analyze(run, "--yaml", str(yml), "--nanobody-chain", "B")

            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            result = json.loads((run / "analysis" / "results.json").read_text(encoding="utf-8"))
            prov = result["settings"]["effective_msa_policy"]
            self.assertEqual(prov["requested_msa_policy"], "cache")
            self.assertEqual(prov["msa_classes"], {"A": "empty", "B": "empty"})
            self.assertFalse(prov["delta_eligible"])
            self.assertTrue(any("requested MSA policy" in r for r in prov["delta_ineligible_reasons"]))


class ProvenanceBoundaryTests(unittest.TestCase):
    def test_relative_user_msa_is_resolved_from_yaml_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "config"
            config.mkdir()
            msa = config / "custom.csv"
            msa.write_text("key,sequence\n-1,AAAA\n", encoding="utf-8")
            source = config / "input.yaml"
            source.write_text(
                "version: 1\nsequences:\n- protein:\n"
                "    id: A\n    sequence: AAAA\n    msa: custom.csv\n",
                encoding="utf-8",
            )

            output, rows = msa_cache.rewrite_yaml(
                source, root / "cache", msa_cache.DEFAULT_URL, "greedy", False
            )

            self.assertEqual(rows, [("A", "사용자 MSA 보존", 0)])
            self.assertIn(f"msa: {msa.resolve()}", output.read_text(encoding="utf-8"))

    def test_cache_filename_does_not_prove_unpaired_msa(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "0123456789abcdef.csv"
            path.write_text("key,sequence\n1,AAAA\n")
            self.assertFalse(analyze._classify_msa(str(path), "AAAA", "cache")["delta_eligible"])
            path.write_text("key,sequence\n-1,AAAA\n")
            actual = analyze._classify_msa(str(path), "AAAA", "cache")
            self.assertTrue(actual["delta_eligible"])
            self.assertEqual(actual["class"], "cache-unpaired")
            self.assertIn("content_sha256", actual)

    def test_top_level_template_comparison_is_explicitly_unvalidated(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "input.yaml"
            path.write_text("sequences:\n- protein: {id: A, sequence: AAAA, msa: empty}\n"
                            "templates:\n- cif: external_template.cif\n")
            actual = analyze.yaml_input_provenance(path, "empty")
            self.assertFalse(actual["delta_eligible"])
            self.assertTrue(actual["advanced_input_hash"])


if __name__ == "__main__":
    unittest.main()
