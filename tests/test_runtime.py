import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_SH = ROOT / "run.sh"
BOLTZ_PY = Path(os.environ.get("BOLTZ_TEST_PY", "/home/keunsoo/miniforge3/envs/ncf-boltz2/bin/python"))
try:
    import gemmi  # noqa: F401

    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False

sys.path.insert(0, str(ROOT / "scripts"))
import runtime_state  # noqa: E402


class RuntimeStateTests(unittest.TestCase):
    def _write_model(self, pred_dir: Path, stem: str, idx: int, n_tokens: int = 3):
        model_stem = f"{stem}_model_{idx}"
        (pred_dir / f"{model_stem}.pdb").write_text(
            "\n".join(
                [
                    "ATOM      1  N   GLY A   1      11.104  13.207   9.100  1.00 20.00           N",
                    "ATOM      2  CA  GLY A   1      12.104  13.207   9.100  1.00 20.00           C",
                    "ATOM      3  C   GLY A   1      12.604  14.607   9.100  1.00 20.00           C",
                    "END",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (pred_dir / f"confidence_{model_stem}.json").write_text(
            json.dumps(
                {
                    "confidence_score": 0.5,
                    "ptm": 0.6,
                    "iptm": 0.7,
                    "pair_chains_iptm": {"0": {"1": 0.8}},
                }
            ),
            encoding="utf-8",
        )
        np.savez(pred_dir / f"pae_{model_stem}.npz", pae=np.zeros((n_tokens, n_tokens)))
        np.savez(pred_dir / f"plddt_{model_stem}.npz", plddt=np.ones((n_tokens,)))

    def test_validate_run_name_allows_korean_but_blocks_traversal(self):
        self.assertEqual(runtime_state.validate_run_name("첫실행_01"), "첫실행_01")
        for bad in ("", ".", "..", "../escape", "a/b", r"a\\b", "/tmp/x", "bad\x00name"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runtime_state.validate_run_name(bad)

    def test_output_path_must_stay_under_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "root"
            root.mkdir()
            resolved = runtime_state.confined_child(root, "첫실행")
            self.assertEqual(resolved, root.resolve() / "첫실행")
            with self.assertRaises(ValueError):
                runtime_state.confined_child(root, "../escape")

    def test_runtime_provenance_tracks_executable_content_and_kernel_mode(self):
        with tempfile.TemporaryDirectory() as d:
            executable = Path(d) / "boltz"
            executable.write_text("version one\n", encoding="utf-8")
            first = runtime_state.runtime_provenance(executable, "enabled")
            executable.write_text("version two\n", encoding="utf-8")
            second = runtime_state.runtime_provenance(executable, "disabled")

            self.assertNotEqual(first["boltz_executable_sha256"], second["boltz_executable_sha256"])
            self.assertEqual(first["kernel_mode"], "enabled")
            self.assertEqual(second["kernel_mode"], "disabled")
            self.assertTrue(first["python"])

    def test_sha256_verification_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "asset.js"
            path.write_text("trusted", encoding="utf-8")
            digest = runtime_state.file_sha256(path)
            self.assertEqual(runtime_state.main(["verify-sha256", str(path), digest]), 0)
            self.assertEqual(runtime_state.main(["verify-sha256", str(path), "0" * 64]), 2)

    def test_prediction_validation_requires_requested_models(self):
        if not HAS_GEMMI:
            self.skipTest("gemmi is required for structure validation")
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d)
            self._write_model(pred, "case", 0)
            self._write_model(pred, "case", 1)
            runtime_state.validate_prediction_outputs(pred, expected_samples=2)
            with self.assertRaisesRegex(ValueError, "expected 3 model"):
                runtime_state.validate_prediction_outputs(pred, expected_samples=3)

    def test_prediction_validation_rejects_shape_mismatch(self):
        if not HAS_GEMMI:
            self.skipTest("gemmi is required for structure validation")
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d)
            self._write_model(pred, "case", 0, n_tokens=3)
            np.savez(pred / "plddt_case_model_0.npz", plddt=np.ones((2,)))
            with self.assertRaisesRegex(ValueError, "plddt length"):
                runtime_state.validate_prediction_outputs(pred, expected_samples=1)

    def test_prediction_validation_rejects_empty_confidence_and_fake_structure(self):
        if not HAS_GEMMI:
            self.skipTest("gemmi is required for structure validation")
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d)
            self._write_model(pred, "case", 0)
            (pred / "confidence_case_model_0.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "confidence_score"):
                runtime_state.validate_prediction_outputs(pred, expected_samples=1)

            self._write_model(pred, "case", 0)
            (pred / "case_model_0.pdb").write_text("not a structure\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot parse structure|contains no atoms"):
                runtime_state.validate_prediction_outputs(pred, expected_samples=1)

    def test_prediction_validation_rejects_zero_token_arrays(self):
        if not HAS_GEMMI:
            self.skipTest("gemmi is required for structure validation")
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d)
            self._write_model(pred, "case", 0)
            np.savez(pred / "pae_case_model_0.npz", pae=np.zeros((0, 0)))
            np.savez(pred / "plddt_case_model_0.npz", plddt=np.zeros((0,)))
            with self.assertRaisesRegex(ValueError, "at least one token"):
                runtime_state.validate_prediction_outputs(pred, expected_samples=1)

    def test_cli_validate_predictions(self):
        if not HAS_GEMMI and not BOLTZ_PY.exists():
            self.skipTest("gemmi is required for structure validation")
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d)
            self._write_model(pred, "case", 0)
            py_exe = str(BOLTZ_PY if BOLTZ_PY.exists() else Path(sys.executable))
            got = subprocess.run(
                [
                    py_exe,
                    str(ROOT / "scripts" / "runtime_state.py"),
                    "validate-predictions",
                    str(pred),
                    "--expected-samples",
                    "1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(got.returncode, 0, got.stderr)
            self.assertIn("OK", got.stdout)


class RunShellValidationTests(unittest.TestCase):
    def _make_fake_env(self, tmp: Path, boltz_body: str):
        env = tmp / "env"
        (env / "bin").mkdir(parents=True)
        py = env / "bin" / "python"
        py.symlink_to(BOLTZ_PY if BOLTZ_PY.exists() else Path(sys.executable))
        boltz = env / "bin" / "boltz"
        boltz.write_text(boltz_body, encoding="utf-8")
        boltz.chmod(0o755)
        return env

    def _make_env_with_python_wrapper(self, tmp: Path, python_body: str):
        env = tmp / "env"
        (env / "bin").mkdir(parents=True)
        py = env / "bin" / "python"
        py.write_text(python_body, encoding="utf-8")
        py.chmod(0o755)
        boltz = env / "bin" / "boltz"
        boltz.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        boltz.chmod(0o755)
        return env

    def _run_env(self, tmp: Path, extra=None):
        env = {
            "BOLTZ_ENV": str(tmp / "env"),
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp),
            "RETRY_SLEEP": "0",
        }
        if extra:
            env.update(extra)
        return env

    def _cleanup_run_dirs(self, name: str):
        for path in (
            ROOT / "inputs" / "generated" / f"batch_{name}",
            ROOT / "outputs" / f"batch_{name}",
            ROOT / "inputs" / "generated" / f"{name}.yaml",
            ROOT / "inputs" / "generated" / f"{name}.meta.json",
            ROOT / "outputs" / name,
        ):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    def test_jobs_filter_is_rejected_before_batch_report(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._make_fake_env(tmp, "#!/usr/bin/env bash\nexit 0\n")
            name = f"rt_{tmp.name}"
            self._cleanup_run_dirs(name)
            batch = tmp / "batch.tsv"
            antigen = "A" * 40
            nanobody = "Q" * 110
            batch.write_text(f"name\tantigen\tnanobody\njob1\t{antigen}\t{nanobody}\n", encoding="utf-8")
            sentinel = ROOT / "outputs" / f"batch_{name}" / "job1" / ".completed"
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("keep", encoding="utf-8")
            try:
                got = subprocess.run(
                    [
                        "bash",
                        str(RUN_SH),
                        "--batch",
                        str(batch),
                        "--name",
                        name,
                        "--jobs",
                        "missing",
                        "--msa",
                        "empty",
                    ],
                    cwd=ROOT,
                    env=self._run_env(tmp),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(got.returncode, 0)
                self.assertIn("--jobs", got.stderr + got.stdout)
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            finally:
                self._cleanup_run_dirs(name)

    def test_partial_batch_failure_is_nonzero_and_subset_total_is_selected_jobs(self):
        if not BOLTZ_PY.exists():
            self.skipTest("Boltz test Python not available")
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fail_flag = tmp / "fail-job2"
            fake = f"""#!/usr/bin/env bash
set -eu
yaml="$2"
out=""
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == "--out_dir" ]]; then out="$2"; shift 2; continue; fi
  shift
done
stem="$(basename "$yaml" .yaml)"
stem="$(basename "$stem" .cached)"
if [[ -f "{fail_flag}" && "$stem" == "job2" ]]; then exit 9; fi
pred="$out/boltz_results_${{stem}}/predictions/${{stem}}"
mkdir -p "$pred"
cp "{tmp}/model.pdb" "$pred/${{stem}}_model_0.pdb"
cp "{tmp}/pae.npz" "$pred/pae_${{stem}}_model_0.npz"
cp "{tmp}/plddt.npz" "$pred/plddt_${{stem}}_model_0.npz"
printf '{{"confidence_score":0.5,"ptm":0.6,"iptm":0.7,"pair_chains_iptm":{{"0":{{"1":0.8}},"1":{{"0":0.8}}}}}}\n' > "$pred/confidence_${{stem}}_model_0.json"
"""
            self._make_fake_env(tmp, fake)
            (tmp / "model.pdb").write_text(
                "ATOM      1  N   GLY A   1      10.000  13.000  10.000  1.00 80.00           N\n"
                "ATOM      2  CA  GLY A   1      11.000  13.000  10.000  1.00 80.00           C\n"
                "ATOM      3  C   GLY A   1      12.000  13.000  10.000  1.00 80.00           C\n"
                "ATOM      4  N   GLY B   1      13.000  13.000  10.000  1.00 80.00           N\n"
                "ATOM      5  CA  GLY B   1      14.000  13.000  10.000  1.00 80.00           C\n"
                "ATOM      6  C   GLY B   1      15.000  13.000  10.000  1.00 80.00           C\n"
                "END\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    str(BOLTZ_PY),
                    "-c",
                    (
                        f"import numpy as np; np.savez('{tmp / 'pae.npz'}', pae=np.ones((2,2))); "
                        f"np.savez('{tmp / 'plddt.npz'}', plddt=np.ones((2,)))"
                    ),
                ],
                check=True,
            )
            batch = tmp / "batch.tsv"
            antigen = "A" * 40
            nanobody = "G" * 110
            from msa_cache import DEFAULT_URL, cache_path

            msa_dir = tmp / "msa"
            msa_dir.mkdir()
            for sequence in (antigen, nanobody):
                cache_path(msa_dir, sequence, DEFAULT_URL, "greedy").write_text(
                    f"key,sequence\n-1,{sequence}\n"
                )
            batch_env = self._run_env(tmp, {"MSA_CACHE_DIR": str(msa_dir)})
            batch.write_text(
                f"name\tantigen\tnanobody\njob1\t{antigen}\t{nanobody}\n"
                f"job2\t{antigen}\t{nanobody}\n",
                encoding="utf-8",
            )
            name = f"rt_{tmp.name}"
            self._cleanup_run_dirs(name)
            base = [
                "bash", str(RUN_SH), "--batch", str(batch), "--name", name,
                "--samples", "1", "--msa", "cache", "--no-kernels",
            ]
            try:
                first = subprocess.run(
                    base, cwd=ROOT, env=batch_env, text=True,
                    capture_output=True, check=False,
                )
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                repeated = subprocess.run(
                    [*base, "--jobs", "job1"], cwd=ROOT, env=batch_env,
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(repeated.returncode, 0, repeated.stderr + repeated.stdout)
                self.assertIn("[SKIP] job1", repeated.stdout)
                msa_file = cache_path(msa_dir, antigen, DEFAULT_URL, "greedy")
                msa_file.write_text(msa_file.read_text() + f"1,{antigen[:-1]}G\n")
                changed_msa = subprocess.run(
                    [*base, "--jobs", "job1"], cwd=ROOT, env=batch_env,
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(changed_msa.returncode, 0, changed_msa.stderr + changed_msa.stdout)
                self.assertNotIn("[SKIP] job1", changed_msa.stdout)
                changed_analysis = subprocess.run(
                    [*base, "--jobs", "job1", "--dockq-mismatches", "1"],
                    cwd=ROOT, env=batch_env, text=True, capture_output=True, check=False,
                )
                self.assertEqual(changed_analysis.returncode, 0, changed_analysis.stderr + changed_analysis.stdout)
                self.assertNotIn("[SKIP] job1", changed_analysis.stdout)
                (ROOT / "outputs" / f"batch_{name}" / "job2" / ".completed").unlink()
                fail_flag.write_text("fail\n", encoding="utf-8")

                second = subprocess.run(
                    [*base, "--jobs", " job2 "], cwd=ROOT, env=batch_env,
                    text=True, capture_output=True, check=False,
                )

                output = second.stderr + second.stdout
                self.assertNotEqual(second.returncode, 0, output)
                self.assertIn("실패 1", output)
                self.assertIn("선택 1 / manifest 2", output)
            finally:
                self._cleanup_run_dirs(name)

    def test_single_no_embed_rejected_before_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            boltz_log = tmp / "boltz.called"
            self._make_fake_env(
                tmp,
                f"#!/usr/bin/env bash\ntouch {boltz_log}\nexit 0\n",
            )
            yaml = tmp / "in.yaml"
            yaml.write_text("sequences: []\n", encoding="utf-8")
            got = subprocess.run(
                [
                    "bash",
                    str(RUN_SH),
                    "--yaml",
                    str(yaml),
                    "--name",
                    f"rt_{tmp.name}",
                    "--no-embed",
                    "--msa",
                    "empty",
                ],
                cwd=ROOT,
                env=self._run_env(tmp),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(got.returncode, 0)
            self.assertFalse(boltz_log.exists())
            self.assertIn("--no-embed", got.stderr + got.stdout)

    def test_invalid_explicit_boltz_env_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            bad_env = tmp / "not_boltz"
            bad_env.mkdir()
            got = subprocess.run(
                ["bash", str(RUN_SH), "--doctor"],
                cwd=ROOT,
                env={"BOLTZ_ENV": str(bad_env), "PATH": "/usr/bin:/bin", "HOME": str(tmp)},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(got.returncode, 0)
            self.assertIn("BOLTZ_ENV", got.stderr + got.stdout)

    def test_partial_prediction_is_cleared_and_retried(self):
        if not BOLTZ_PY.exists():
            self.skipTest("Boltz test Python not available")
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            calls = tmp / "calls"
            fake = f"""#!/usr/bin/env bash
set -eu
yaml="$2"
out=""
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == "--out_dir" ]]; then out="$2"; shift 2; continue; fi
  shift
done
stem="$(basename "$yaml" .yaml)"
stem="$(basename "$stem" .cached)"
n=1
if [[ -f "{calls}" ]]; then n=$(cat "{calls}"); n=$((n+1)); fi
printf "%s" "$n" > "{calls}"
pred="$out/boltz_results_${{stem}}/predictions/${{stem}}"
mkdir -p "$pred"
models=1
if [[ "$n" -ge 2 ]]; then models=2; fi
for i in $(seq 0 $((models-1))); do
  cp "{tmp}/model.pdb" "$pred/${{stem}}_model_${{i}}.pdb"
  cp "{tmp}/pae.npz" "$pred/pae_${{stem}}_model_${{i}}.npz"
  cp "{tmp}/plddt.npz" "$pred/plddt_${{stem}}_model_${{i}}.npz"
  printf '{{"confidence_score":0.5,"ptm":0.6,"iptm":0.7}}\\n' > "$pred/confidence_${{stem}}_model_${{i}}.json"
done
exit 0
"""
            self._make_fake_env(tmp, fake)
            (tmp / "model.pdb").write_text(
                "ATOM      1  N   GLY A   1      11.104  13.207   9.100  1.00 20.00           N\nEND\n",
                encoding="utf-8",
            )
            subprocess.run(
                [str(BOLTZ_PY), "-c", f"import numpy as np; np.savez('{tmp / 'pae.npz'}', pae=np.zeros((1,1))); np.savez('{tmp / 'plddt.npz'}', plddt=np.ones((1,)))"],
                check=True,
            )
            name = f"rt_{tmp.name}"
            self._cleanup_run_dirs(name)
            yaml = tmp / f"{name}.yaml"
            yaml.write_text("version: 1\nsequences: []\n", encoding="utf-8")
            try:
                got = subprocess.run(
                    [
                        "bash",
                        str(RUN_SH),
                        "--yaml",
                        str(yaml),
                        "--name",
                        name,
                        "--samples",
                        "2",
                        "--msa",
                        "empty",
                        "--nanobody-chain",
                        "Z",  # Force analysis failure after prediction retry, not input preflight.
                    ],
                    cwd=ROOT,
                    env=self._run_env(tmp),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(got.returncode, 0)
                self.assertEqual(calls.read_text(encoding="utf-8"), "2")
                self.assertFalse((ROOT / "outputs" / name / ".completed").exists())
                params = json.loads(
                    (ROOT / "outputs" / name / ".run_params.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    params["runtime"]["boltz_executable_sha256"],
                    runtime_state.file_sha256(tmp / "env" / "bin" / "boltz"),
                )
                pred = ROOT / "outputs" / name / f"boltz_results_{name}" / "predictions" / name
                self.assertTrue((pred / f"confidence_{name}_model_1.json").exists())
            finally:
                self._cleanup_run_dirs(name)

    def test_report_only_replays_saved_settings_with_only_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            arg_log = tmp / "analyze_args.json"
            run_dir = tmp / "run"
            (run_dir / "analysis").mkdir(parents=True)
            saved_reference = tmp / "old_reference.cif"
            new_reference = tmp / "new_reference.cif"
            saved_reference.write_text("old\n", encoding="utf-8")
            new_reference.write_text("new\n", encoding="utf-8")
            (run_dir / "analysis" / "results.json").write_text(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "models": [],
                        "settings": {
                            "analysis_contract_version": 1,
                            "run_dir": str(run_dir),
                            "reference": str(saved_reference),
                            "dockq_mapping": "OLD:OLD",
                            "dockq_allowed_mismatches": 5,
                            "no_nanobody": True,
                            "nanobody_chain": None,
                            "antigen_chain": "A",
                        },
                    }
                ),
                encoding="utf-8",
            )
            python_body = f"""#!/usr/bin/env bash
set -eu
if [[ "${{1:-}}" == *"/scripts/analyze.py" ]]; then
  "{BOLTZ_PY}" - "$@" <<'PY'
import json, sys
open("{arg_log}", "w", encoding="utf-8").write(json.dumps(sys.argv[1:], ensure_ascii=False))
PY
  exit 0
fi
if [[ "${{1:-}}" == *"/scripts/make_report.py" ]]; then
  exit 0
fi
exec "{BOLTZ_PY}" "$@"
"""
            self._make_env_with_python_wrapper(tmp, python_body)
            got = subprocess.run(
                [
                    "bash",
                    str(RUN_SH),
                    "--report-only",
                    str(run_dir),
                    "--reference",
                    str(new_reference),
                    "--dockq-mapping",
                    "AB:CD",
                    "--dockq-mismatches",
                    "0",
                    "--nanobody-chain",
                    "B",
                    "--antigen-chain",
                    "A",
                ],
                cwd=ROOT,
                env=self._run_env(tmp),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(got.returncode, 0, got.stderr + got.stdout)
            args = json.loads(arg_log.read_text(encoding="utf-8"))
            self.assertIn("--restore-settings", args)
            self.assertEqual(args[args.index("--restore-settings") + 1], str(run_dir / "analysis" / "results.json"))
            self.assertIn("--reference", args)
            self.assertEqual(args[args.index("--reference") + 1], str(new_reference))
            self.assertNotIn(str(saved_reference), args)
            self.assertEqual(args[args.index("--dockq-mapping") + 1], "AB:CD")
            self.assertEqual(args[args.index("--dockq-allowed-mismatches") + 1], "0")
            self.assertEqual(args[args.index("--nanobody-chain") + 1], "B")
            self.assertEqual(args[args.index("--antigen-chain") + 1], "A")
            self.assertNotIn("--no-nanobody", args)

    def test_list_uses_primary_interface_scores(self):
        name = "rt_primary_interface"
        out = ROOT / "outputs" / name
        self._cleanup_run_dirs(name)
        try:
            (out / "analysis").mkdir(parents=True)
            (out / "analysis" / "results.json").write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "index": 0,
                                "primary_interface": {
                                    "iptm": 0.9,
                                    "ipsae": 0.8,
                                    "iptm_source": "pae_union_iptm",
                                    "ipsae_source": "builtin-union",
                                },
                                "boltz_pair_iptm": {"nanobody_in_antigen_frame": 0.1},
                                "ipsae": {"max": {"ipsae": 0.2}},
                                "boltz": {"ptm": 0.3},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            got = subprocess.run(
                ["bash", str(RUN_SH), "--list"],
                cwd=ROOT,
                env={"BOLTZ_ENV": "/home/keunsoo/miniforge3/envs/ncf-boltz2", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(got.returncode, 0, got.stderr)
            self.assertIn("ipTM/pTM", got.stdout)
            line = next(line for line in got.stdout.splitlines() if name in line)
            self.assertIn("0.900", line)
            self.assertIn("0.800", line)
            self.assertNotIn("0.100", line)
            self.assertNotIn("0.200", line)
        finally:
            self._cleanup_run_dirs(name)


if __name__ == "__main__":
    unittest.main()
