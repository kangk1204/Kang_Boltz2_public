import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_runtime as runtime_tests


class ServeRuntimeTests(unittest.TestCase):
    def run_case(self, tmp, requested):
        helpers = runtime_tests.RunShellValidationTests()
        py = runtime_tests.BOLTZ_PY
        if not py.exists():
            self.skipTest("Boltz test Python not available")
        marker = tmp / "server.json"
        wrapper = f'''#!{py}
import json, os, sys
from pathlib import Path
if sys.argv[1:3] == ["-m", "http.server"]:
    Path({str(marker)!r}).write_text(json.dumps({{"cwd": os.getcwd(), "args": sys.argv[1:]}}))
else:
    os.execv({str(py)!r}, [{str(py)!r}, *sys.argv[1:]])
'''
        helpers._make_env_with_python_wrapper(tmp, wrapper)
        result = subprocess.run(
            ["bash", str(runtime_tests.RUN_SH), "--serve", str(requested)],
            cwd=runtime_tests.ROOT, env=helpers._run_env(tmp), text=True, capture_output=True, check=False,
        )
        return result, marker

    def test_missing_path_does_not_serve_existing_parent_index(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "index.html").write_text("unrelated parent report")
            requested = tmp / "missing-report"
            result, marker = self.run_case(tmp, requested)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(str(requested), result.stderr)
            self.assertIn("--list", result.stderr)
            self.assertFalse(marker.exists())

    def test_report_directory_and_index_file_serve_the_requested_directory(self):
        for index_file in (False, True):
            with self.subTest(index_file=index_file), tempfile.TemporaryDirectory() as directory:
                tmp = Path(directory)
                report = tmp / "report with spaces"
                report.mkdir()
                index = report / "index.html"
                index.write_text("report")
                result, marker = self.run_case(tmp, index if index_file else report)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                server = json.loads(marker.read_text())
                self.assertEqual(server["cwd"], str(report))
                self.assertEqual(server["args"], ["-m", "http.server", "8765", "--bind", "127.0.0.1"])

    def test_existing_directory_without_index_explains_how_to_find_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            report = tmp / "unfinished" / "report"
            report.mkdir(parents=True)
            result, marker = self.run_case(tmp, report)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(str(report / "index.html"), result.stderr)
            self.assertIn("--list", result.stderr)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
