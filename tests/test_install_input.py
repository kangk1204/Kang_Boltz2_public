from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from runtime_state import ensure_path_within, validate_run_name  # noqa: E402


class RuntimeNameTests(unittest.TestCase):
    def test_validate_run_name_allows_korean(self) -> None:
        self.assertEqual(validate_run_name("첫실행"), "첫실행")

    def test_validate_run_name_rejects_path_syntax(self) -> None:
        for bad in ("../escape", "a/b", r"a\b", ".", "..", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_run_name(bad)

    def test_ensure_path_within_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            outside = Path(td) / "outside"
            root.mkdir()
            outside.mkdir()
            link = root / "link"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                ensure_path_within(link / "x.yaml", root)


class PrepareInputTests(unittest.TestCase):
    def run_prepare_input(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "prepare_input.py"), *args],
            cwd=str(cwd or ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

    def test_prepare_input_writes_korean_name_under_outdir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            target = tmp / "target.fasta"
            nanobody = tmp / "nanobody.fasta"
            outdir = tmp / "generated"
            target.write_text(">ag\n" + "A" * 30 + "\n", encoding="utf-8")
            nanobody.write_text(">nb\n" + "C" * 30 + "\n", encoding="utf-8")
            proc = self.run_prepare_input(
                "--target", str(target),
                "--nanobody", str(nanobody),
                "--name", "첫실행",
                "--outdir", str(outdir),
                "--msa", "empty",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertTrue((outdir / "첫실행.yaml").is_file())
            self.assertTrue((outdir / "첫실행.meta.json").is_file())

    def test_prepare_input_rejects_traversal_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            target = tmp / "target.fasta"
            nanobody = tmp / "nanobody.fasta"
            target.write_text(">ag\n" + "A" * 30 + "\n", encoding="utf-8")
            nanobody.write_text(">nb\n" + "C" * 30 + "\n", encoding="utf-8")
            proc = self.run_prepare_input(
                "--target", str(target),
                "--nanobody", str(nanobody),
                "--name", "../escape",
                "--outdir", str(tmp / "generated"),
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("--name", proc.stderr + proc.stdout)
            self.assertFalse((tmp / "escape.yaml").exists())


class PrepareBatchTests(unittest.TestCase):
    def run_prepare_batch(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "prepare_batch.py"), *args],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

    def test_prepare_batch_rejects_bad_batch_name_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            table = tmp / "batch.tsv"
            table.write_text("name\tagenten\tnanobody\n", encoding="utf-8")
            proc = self.run_prepare_batch("--batch", str(table), "--name", "../batch")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("batch name", proc.stderr + proc.stdout)

    def test_prepare_batch_rejects_dot_row_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ag = tmp / "ag.fasta"
            nb = tmp / "nb.fasta"
            ag.write_text(">ag\n" + "A" * 30 + "\n", encoding="utf-8")
            nb.write_text(">nb\n" + "C" * 30 + "\n", encoding="utf-8")
            table = tmp / "batch.tsv"
            with table.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                writer.writerow(["name", "antigen", "nanobody"])
                writer.writerow([".", str(ag), str(nb)])
            proc = self.run_prepare_batch("--batch", str(table), "--outdir", str(tmp / "out"))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("invalid job name", proc.stderr + proc.stdout)

    def test_prepare_batch_hotspot_dash_means_none(self) -> None:
        # 표의 hotspot 칸에 '-' (없음 관례) 를 넣어도 범위로 해석되어 실패하지 않아야 한다.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ag = tmp / "ag.fasta"
            nb = tmp / "nb.fasta"
            ag.write_text(">ag\n" + "A" * 30 + "\n", encoding="utf-8")
            nb.write_text(">nb\n" + "C" * 30 + "\n", encoding="utf-8")
            table = tmp / "batch.tsv"
            with table.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                writer.writerow(["name", "antigen", "nanobody", "reference", "hotspot", "notes"])
                writer.writerow(["j1", str(ag), str(nb), "-", "-", "no hotspot"])
            proc = self.run_prepare_batch("--batch", str(table), "--outdir", str(tmp / "out"))
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)


class CdrDiagnosticsTests(unittest.TestCase):
    def test_get_cdr_positions_distinguishes_missing_anarci(self) -> None:
        import make_cdr_library as M

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "anarci":
                raise ImportError("no anarci")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(M.metrics, "find_hmmer_dir", return_value=None), \
             mock.patch("builtins.__import__", side_effect=fake_import), \
             self.assertRaises(SystemExit) as cm:
            M.get_cdr_positions("A" * 120)
        self.assertIn("ANARCI Python package", str(cm.exception))

    def test_get_cdr_positions_distinguishes_missing_hmmscan(self) -> None:
        import make_cdr_library as M

        fake_anarci = types.ModuleType("anarci")
        with mock.patch.dict(sys.modules, {"anarci": fake_anarci}), \
             mock.patch.object(M.metrics, "find_hmmer_dir", return_value=None), \
             mock.patch.object(M.metrics, "find_executable", return_value=None), \
             self.assertRaises(SystemExit) as cm:
            M.get_cdr_positions("A" * 120)
        self.assertIn("hmmscan", str(cm.exception))


class SetupContractTests(unittest.TestCase):
    def test_setup_pins_cu128_and_has_no_generic_torch_fallback(self) -> None:
        setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
        constraints = (ROOT / "requirements" / "boltz2-cu128-constraints.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("TORCH_VERSION=\"${TORCH_VERSION:-2.7.0+cu128}\"", setup)
        self.assertIn("--reuse-env", setup)
        self.assertIn("-c \"$CONSTRAINTS\"", setup)
        self.assertNotIn("기본 PyPI torch", setup)
        self.assertIn("torch==2.7.0+cu128", constraints)


if __name__ == "__main__":
    unittest.main()
