"""Batch-level GPU OOM recovery regressions."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import test_runtime as runtime_tests


class BatchOomRecoveryTests(unittest.TestCase):
    def _write_two_chain_prediction_fixtures(self, tmp: Path):
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
        np.savez(tmp / "pae.npz", pae=np.ones((2, 2)))
        np.savez(tmp / "plddt.npz", plddt=np.ones((2,)))

    def _calls(self, calls: Path):
        if not calls.exists():
            return []
        return [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]

    def _run(self, base, tmp: Path, *, extra=()):
        fake_path = tmp / "fake_path"
        fake_path.mkdir(exist_ok=True)
        nvidia_smi = fake_path / "nvidia-smi"
        nvidia_smi.write_text("#!/usr/bin/env bash\necho 'No devices'\nexit 1\n", encoding="utf-8")
        nvidia_smi.chmod(0o755)
        helpers = runtime_tests.RunShellValidationTests()
        env = helpers._run_env(tmp, {"PATH": f"{fake_path}:/usr/bin:/bin"})
        return subprocess.run(
            [*base, *extra],
            cwd=runtime_tests.ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_batch_oom_fallback_completion_tracks_final_settings(self):
        if not runtime_tests.BOLTZ_PY.exists():
            self.skipTest("Boltz test Python not available")
        helpers = runtime_tests.RunShellValidationTests()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            calls = tmp / "calls.jsonl"
            self._write_two_chain_prediction_fixtures(tmp)
            fake = f'''#!{runtime_tests.BOLTZ_PY}
import json
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
yaml = Path(args[1])
stem = yaml.stem.removesuffix(".cached")
with Path({str(calls)!r}).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"stem": stem, "args": args}}) + "\\n")

if stem == "job1" and "--subsample_msa" not in args:
    print("| WARNING: ran out of memory, skipping batch", flush=True)
    raise SystemExit(0)

out = Path(args[args.index("--out_dir") + 1])
pred = out / ("boltz_results_" + stem) / "predictions" / stem
pred.mkdir(parents=True, exist_ok=True)
count = int(args[args.index("--diffusion_samples") + 1])
for i in range(count):
    model_stem = f"{{stem}}_model_{{i}}"
    shutil.copyfile({str(tmp / "model.pdb")!r}, pred / f"{{model_stem}}.pdb")
    shutil.copyfile({str(tmp / "pae.npz")!r}, pred / f"pae_{{model_stem}}.npz")
    shutil.copyfile({str(tmp / "plddt.npz")!r}, pred / f"plddt_{{model_stem}}.npz")
    (pred / f"confidence_{{model_stem}}.json").write_text(
        '{{"confidence_score":0.5,"ptm":0.6,"iptm":0.7,'
        '"pair_chains_iptm":{{"0":{{"1":0.8}},"1":{{"0":0.8}}}}}}\\n',
        encoding="utf-8",
    )
'''
            helpers._make_fake_env(tmp, fake)
            batch = tmp / "batch.tsv"
            antigen = "A" * 40
            nanobody = "G" * 110
            batch.write_text(
                f"name\tantigen\tnanobody\njob1\t{antigen}\t{nanobody}\n"
                f"job2\t{antigen}\t{nanobody}\n",
                encoding="utf-8",
            )
            name = f"oom_batch_{tmp.name}"
            helpers._cleanup_run_dirs(name)
            base = [
                "bash",
                str(runtime_tests.RUN_SH),
                "--batch",
                str(batch),
                "--name",
                name,
                "--samples",
                "3",
                "--msa",
                "empty",
                "--no-kernels",
            ]
            jobs_dir = runtime_tests.ROOT / "outputs" / f"batch_{name}"
            try:
                first = self._run(base, tmp)
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                first_calls = self._calls(calls)
                self.assertEqual([call["stem"] for call in first_calls], ["job1", "job1", "job2"])
                self.assertNotIn("--subsample_msa", first_calls[0]["args"])
                self.assertIn("--subsample_msa", first_calls[1]["args"])
                self.assertNotIn("--subsample_msa", first_calls[2]["args"])
                for call in first_calls:
                    args = call["args"]
                    self.assertEqual(args[args.index("--diffusion_samples") + 1], "3")

                job1_params = json.loads((jobs_dir / "job1" / ".run_params.json").read_text())
                job2_params = json.loads((jobs_dir / "job2" / ".run_params.json").read_text())
                self.assertEqual(job1_params["msa_subsample"], 512)
                self.assertEqual(job1_params["parallel_samples"], 1)
                self.assertTrue(job1_params["oom_retry_applied"])
                self.assertEqual(job2_params["msa_subsample"], 0)
                self.assertEqual(job2_params["parallel_samples"], 1)
                self.assertFalse(job2_params["oom_retry_applied"])

                repeated_default = self._run(base, tmp)
                self.assertEqual(
                    repeated_default.returncode,
                    0,
                    repeated_default.stderr + repeated_default.stdout,
                )
                repeated_calls = self._calls(calls)
                self.assertEqual(
                    [call["stem"] for call in repeated_calls[len(first_calls):]],
                    ["job1", "job1"],
                )
                self.assertIn("[SKIP] job2", repeated_default.stdout)

                cached_job1 = self._run(
                    base,
                    tmp,
                    extra=("--msa-subsample", "512", "--parallel-samples", "1", "--jobs", "job1"),
                )
                self.assertEqual(cached_job1.returncode, 0, cached_job1.stderr + cached_job1.stdout)
                self.assertEqual(len(self._calls(calls)), len(repeated_calls))
                self.assertIn("[SKIP] job1", cached_job1.stdout)

                strict = self._run(base, tmp, extra=("--no-oom-retry", "--jobs", "job1"))
                self.assertNotEqual(strict.returncode, 0, strict.stderr + strict.stdout)
                strict_calls = self._calls(calls)
                self.assertEqual([call["stem"] for call in strict_calls[len(repeated_calls):]], ["job1"])
                self.assertIn("실패 1", strict.stderr + strict.stdout)
            finally:
                helpers._cleanup_run_dirs(name)


if __name__ == "__main__":
    unittest.main()
