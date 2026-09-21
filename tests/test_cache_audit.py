"""Regression cases for source-aware batch reuse and whole-job completion."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_batch  # noqa: E402
import runtime_state  # noqa: E402


class CacheAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ag = self.root / "antigen.fasta"
        self.nb = self.root / "binder.fasta"
        self.ref = self.root / "reference.cif"
        self.ag.write_text(">ag\n" + "A" * 40 + "\n")
        self.nb.write_text(">nb\n" + "Q" * 110 + "\n")
        self.ref.write_text("reference version one\n")
        self.batch = self.root / "batch.tsv"
        self.batch.write_text("name\tantigen\tnanobody\treference\n"
                              "job\tantigen.fasta\tbinder.fasta\treference.cif\n")
        self.out = self.root / "generated"
        self.manifest = self.out / "batch_batch" / "manifest.json"

    def prepare(self):
        argv = ["prepare_batch", "--batch", str(self.batch), "--outdir", str(self.out),
                "--msa", "empty"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return prepare_batch.main()

    def reusable(self):
        return prepare_batch.manifest_reusable(self.manifest, self.batch, "empty")

    def test_source_content_change_invalidates_even_when_mtime_is_preserved(self):
        for source in (self.ag, self.nb, self.ref):
            with self.subTest(source=source.name):
                self.assertEqual(self.prepare(), 0)
                self.assertTrue(self.reusable())
                stat = source.stat()
                original = source.read_text()
                source.write_text(original.replace("A", "G") if source == self.ag else original + "G")
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertFalse(self.reusable())
                source.write_text(original)

    def test_missing_or_corrupt_generated_files_and_legacy_manifest_invalidate(self):
        for key in ("yaml", "meta"):
            self.assertEqual(self.prepare(), 0)
            job = json.loads(self.manifest.read_text())["jobs"][0]
            generated = Path(job[key])
            generated.write_text("corrupt")
            self.assertFalse(self.reusable())
            generated.unlink()
            self.assertFalse(self.reusable())
        self.assertEqual(self.prepare(), 0)
        m = json.loads(self.manifest.read_text())
        m.pop("schema_version")
        self.manifest.write_text(json.dumps(m))
        self.assertFalse(self.reusable())

    def test_reference_must_be_file_and_missing_reference_fails_before_overwrite(self):
        self.assertEqual(self.prepare(), 0)
        saved = self.manifest.read_bytes()
        self.ref.unlink()
        self.assertEqual(self.prepare(), 1)
        self.assertEqual(self.manifest.read_bytes(), saved)
        with self.assertRaisesRegex(ValueError, "reference file not found"):
            prepare_batch.resolve_reference("A" * 40, [self.root])

    def test_completion_key_tracks_reference_metadata_and_analysis_settings(self):
        self.assertEqual(self.prepare(), 0)
        meta = json.loads(self.manifest.read_text())["jobs"][0]["meta"]
        settings = {"dockq_allowed_mismatches": 0, "antigen_chains": "A"}
        base = runtime_state.job_fingerprint("prediction", meta, str(self.ref), settings)
        self.assertEqual(base, runtime_state.job_fingerprint("prediction", meta, str(self.ref), settings))
        for changed in ({**settings, "dockq_allowed_mismatches": 5},
                        {**settings, "antigen_chains": "A,C"}):
            self.assertNotEqual(base, runtime_state.job_fingerprint("prediction", meta, str(self.ref), changed))
        self.ref.write_text("reference version two")
        self.assertNotEqual(base, runtime_state.job_fingerprint("prediction", meta, str(self.ref), settings))
        self.ref.unlink()
        with self.assertRaises(FileNotFoundError):
            runtime_state.job_fingerprint("prediction", meta, str(self.ref), settings)

    def test_analysis_provenance_hashes_custom_ipsae_implementation(self):
        custom = self.root / "ipsae.py"
        custom.write_text("# first")
        first = runtime_state.analysis_provenance(custom)
        custom.write_text("# second")
        second = runtime_state.analysis_provenance(custom)
        self.assertNotEqual(first["sources"]["ipsae_script"], second["sources"]["ipsae_script"])
        self.assertIn("gemmi", first["packages"])

    def test_effective_msa_hashes_track_same_path_content_changes(self):
        from msa_cache import DEFAULT_URL, cache_path

        source = self.root / "input.yaml"
        source.write_text("sequences:\n- protein:\n    id: A\n    sequence: AAA\n")
        resource = cache_path(self.root, "AAA", DEFAULT_URL, "greedy")
        first = runtime_state.msa_resource_hashes(source, "cache", self.root)
        self.assertIsNone(first[str(resource)])
        resource.write_text("key,sequence\n-1,AAA\n")
        second = runtime_state.msa_resource_hashes(source, "cache", self.root)
        resource.write_text("key,sequence\n-1,AAA\n1,AAG\n")
        third = runtime_state.msa_resource_hashes(source, "cache", self.root)
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        source.write_text(f"sequences:\n- protein:\n    id: A\n    sequence: AAA\n    msa: {resource.name}\n")
        self.assertEqual(third, runtime_state.msa_resource_hashes(source, "server", self.root))
        resource.unlink()
        with self.assertRaisesRegex(ValueError, "MSA file not found"):
            runtime_state.msa_resource_hashes(source, "cache", self.root)


if __name__ == "__main__":
    unittest.main()
