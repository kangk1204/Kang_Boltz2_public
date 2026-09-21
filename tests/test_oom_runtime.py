"""Exercise GPU OOM recovery without allocating GPU memory."""

import json
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import test_runtime as runtime_tests


class OomRecoveryTests(unittest.TestCase):
    @contextmanager
    def run_case(self, *, exit_code=0, always_fail=False, extra=()):
        helpers = runtime_tests.RunShellValidationTests()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            fixture = tmp / "fixture"
            fixture.mkdir()
            runtime_tests.RuntimeStateTests()._write_model(fixture, "fixture", 0, n_tokens=1)
            calls = tmp / "calls.jsonl"
            py = runtime_tests.BOLTZ_PY
            if not py.exists():
                self.skipTest("Boltz test Python not available")
            fake = f'''#!{py}
import json, shutil, sys
from pathlib import Path
args = sys.argv[1:]
with Path({str(calls)!r}).open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
if "--subsample_msa" not in args or {always_fail!r}:
    print("| WARNING: ran out of memory, skipping batch", flush=True)
    raise SystemExit({exit_code})
out = Path(args[args.index("--out_dir") + 1])
stem = Path(args[1]).stem
pred = out / ("boltz_results_" + stem) / "predictions" / stem
pred.mkdir(parents=True, exist_ok=True)
count = int(args[args.index("--diffusion_samples") + 1])
for i in range(count):
    for source in Path({str(fixture)!r}).iterdir():
        name = source.name.replace("fixture_model_0", stem + "_model_" + str(i))
        shutil.copyfile(source, pred / name)
'''
            helpers._make_fake_env(tmp, fake)
            yaml = tmp / "input.yaml"
            yaml.write_text("version: 1\nsequences:\n  - protein: {id: A, sequence: G}\n")
            name = f"oom_{tmp.name}"
            out = runtime_tests.ROOT / "outputs" / name
            out.mkdir(parents=True)
            (out / ".completed").write_text("stale completion from previous run")
            try:
                result = subprocess.run(
                    ["bash", str(runtime_tests.RUN_SH), "--yaml", str(yaml), "--name", name,
                     "--target-only", "--samples", "3", "--parallel-samples", "2", *extra],
                    cwd=runtime_tests.ROOT, env=helpers._run_env(tmp),
                    text=True, capture_output=True, check=False,
                )
                arguments = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
                yield result, arguments, out
            finally:
                helpers._cleanup_run_dirs(name)

    def test_zero_exit_oom_retries_with_lower_memory_and_preserves_samples(self):
        with self.run_case() as (result, calls, out):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(calls), 2)
            self.assertNotIn("--subsample_msa", calls[0])
            self.assertIn("--subsample_msa", calls[1])
            self.assertEqual(calls[1][calls[1].index("--num_subsampled_msa") + 1], "512")
            self.assertEqual(calls[1][calls[1].index("--max_parallel_samples") + 1], "1")
            for call in calls:
                self.assertEqual(call[call.index("--diffusion_samples") + 1], "3")
            params = json.loads((out / ".run_params.json").read_text())
            self.assertEqual(params["msa_subsample"], 512)
            self.assertEqual(params["requested_msa_subsample"], 0)
            self.assertEqual(params["parallel_samples"], 1)
            self.assertTrue(params["oom_retry_applied"])
            saved = json.loads((out / "analysis/results.json").read_text())
            self.assertEqual(saved["settings"]["run_params"]["msa_subsample"], 512)
            self.assertTrue((out / ".completed").is_file())
            self.assertTrue((out / "report/index.html").is_file())
            self.assertIn("out of memory", (out / "prediction_attempt_1.log").read_text())
            self.assertTrue((out / "prediction_attempt_2.log").is_file())

    def test_nonzero_exit_oom_is_also_recovered(self):
        with self.run_case(exit_code=9) as (result, calls, _out):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(calls), 2)
            self.assertIn("--subsample_msa", calls[1])

    def test_repeated_oom_fails_without_completion(self):
        with self.run_case(always_fail=True) as (result, calls, out):
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(calls), 2)
            self.assertIn("GPU 메모리 부족", result.stdout + result.stderr)
            self.assertFalse((out / ".completed").exists())
            self.assertFalse((out / "report/index.html").exists())

    def test_strict_mode_stops_without_changing_settings(self):
        with self.run_case(extra=("--no-oom-retry",)) as (result, calls, out):
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("--subsample_msa", calls[0])
            self.assertFalse((out / ".completed").exists())

    def test_explicit_subsample_is_recorded_without_automatic_change(self):
        with self.run_case(extra=("--msa-subsample", "256")) as (result, calls, out):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(calls), 1)
            params = json.loads((out / ".run_params.json").read_text())
            self.assertEqual(params["msa_subsample"], 256)
            self.assertEqual(params["requested_msa_subsample"], 256)
            self.assertFalse(params["oom_retry_applied"])

    def test_recovery_does_not_increase_an_explicit_msa_limit(self):
        with self.run_case(always_fail=True, extra=("--msa-subsample", "128")) as (result, calls, out):
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(calls), 2)
            for call in calls:
                self.assertEqual(call[call.index("--num_subsampled_msa") + 1], "128")
            self.assertEqual(calls[1][calls[1].index("--max_parallel_samples") + 1], "1")
            self.assertFalse((out / ".completed").exists())

    def test_already_reduced_oom_stops_without_an_identical_retry(self):
        with self.run_case(always_fail=True, extra=("--msa-subsample", "512", "--parallel-samples", "1")) as case:
            result, calls, out = case
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(calls), 1)
            self.assertFalse((out / ".completed").exists())


if __name__ == "__main__":
    unittest.main()
