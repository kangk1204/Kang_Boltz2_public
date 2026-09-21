from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_report  # noqa: E402


def _model(run: Path, *, union: bool = True) -> dict:
    pred = run / "predictions"
    pred.mkdir(parents=True, exist_ok=True)
    cif = pred / "job_model_0.pdb"
    cif.write_text(
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 80.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    (pred / "confidence_job_model_0.json").write_text(
        json.dumps({"confidence_score": 0.5, "ptm": 0.4, "iptm": 0.7}),
        encoding="utf-8",
    )
    np.savez(pred / "pae_job_model_0.npz", pae=np.array([[1.0]], dtype=float))
    np.savez(pred / "plddt_job_model_0.npz", plddt=np.array([80.0], dtype=float))
    return {
        "index": 0,
        "stem": "job_model_0",
        "cif": str(cif),
        "confidence_json": str(pred / "confidence_job_model_0.json"),
        "pae_npz": str(pred / "pae_job_model_0.npz"),
        "boltz": {"ptm": 0.4, "confidence_score": 0.5},
        "boltz_pair_iptm": {
            "nanobody_in_antigen_frame": 0.72,
            "antigen_in_nanobody_frame": 0.61,
        },
        "iptm_from_pae": {"nanobody_in_antigen_frame": 0.88},
        "ipsae": {
            "source": "official",
            "max": {"ipsae": 0.51, "ipsae_d0chn": 0.52, "pdockq2": 0.42},
            "union": {"ipsae": 0.91} if union else {},
        },
        "interface_8A": {"pae_mean_ab": 4.2, "n_contacts": 8},
        "plddt": {"mean": 80.0, "per_chain": {"B": {"mean": 86.0}}},
        "cdr_mean_plddt": {"CDR3": 75.0},
        "dockq": {
            "headline": 0.55,
            "headline_kind": "interface mean",
            "best_mapping_str": "A:B",
            "interfaces": {
                "AB": {
                    "DockQ": 0.55,
                    "F1": 0.6,
                    "iRMSD": 1.2,
                    "LRMSD": 2.3,
                    "fnat": 0.7,
                    "fnonnat": 0.1,
                    "nat_correct": 7,
                    "nat_total": 10,
                    "clashes": 0,
                    "CAPRI": "Medium",
                }
            },
        },
    }


def _results(run: Path, *, figure_path: str | None = None) -> dict:
    return {
        "run_name": "job",
        "run_dir": str(run),
        "predictions_dir": str(run / "predictions"),
        "created": "2026-09-20T00:00:00+09:00",
        "chains": [
            {"id": "A", "role": "antigen", "n_res": 10, "sequence": "A" * 10},
            {"id": "C", "role": "antigen", "n_res": 12, "sequence": "C" * 12},
            {"id": "B", "role": "nanobody", "n_res": 11, "sequence": "B" * 11},
        ],
        "antigen_chain": "A",
        "antigen_chains": ["A", "C"],
        "nanobody_chain": "B",
        "role_source": "metadata",
        "settings": {
            "reference": str(run / "reference.cif"),
            "run_params": {"samples": 1, "seed": 42, "steps": 200, "recycles": 3, "msa": "cache"},
        },
        "rank_key": "primary_interface_scores",
        "cdr": {"available": False},
        "models": [_model(run)],
        "best_model_index": 0,
        "best_model_stem": "job_model_0",
        "warnings": [],
        "figures": {"model_comparison": figure_path} if figure_path else {},
    }


class ReportAuditTests(unittest.TestCase):
    def test_dockq_rows_include_per_interface_capri_cell(self):
        with tempfile.TemporaryDirectory() as d:
            data = _results(Path(d))
            rendered = make_report.build_dockq_section(data, data["models"][0])
            row = re.search(r"<tbody>(.*?)</tbody>", rendered, re.DOTALL).group(1)
            self.assertEqual(row.count("<td"), 10)
            self.assertIn("CAPRI", rendered)
            self.assertIn(">Medium<", rendered)

    def test_summary_tsv_uses_primary_union_source_and_adds_scope_source(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "summary.tsv"
            make_report.write_summary_tsv(_results(Path(d)), out)
            lines = out.read_text(encoding="utf-8").splitlines()
            header = lines[0].split("\t")
            row = dict(zip(header, lines[1].split("\t"), strict=True))
            self.assertEqual(row["ipSAE"], "0.9100")
            self.assertEqual(row["ipSAE_source"], "builtin-union")
            self.assertEqual(row["primary_scope"], "antigen-union")
            self.assertIn("pae_union_iptm", row["primary_source"])
            self.assertIn("builtin-union", row["primary_source"])

    def test_build_html_embeds_model_comparison_only_from_safe_path(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            run.mkdir()
            fig = run / "analysis" / "figures" / "model_comparison.png"
            fig.parent.mkdir(parents=True)
            fig.write_bytes(b"not-a-real-png")
            rendered = make_report.build_html(_results(run, figure_path=str(fig)), run)
            self.assertIn("모델 비교 요약", rendered)
            self.assertIn("data:image/png;base64,", rendered)

            unsafe = Path(d) / "outside.png"
            unsafe.write_bytes(b"outside")
            rendered_unsafe = make_report.build_html(_results(run, figure_path=str(unsafe)), run)
            self.assertNotIn("outside", rendered_unsafe)

    def test_rebase_result_paths_confines_figures_overlay_and_run_dir_to_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            external = root / "external"
            local = root / "copy"
            local_analysis = local / "analysis"
            external.mkdir()
            local_analysis.mkdir(parents=True)
            data = _results(external, figure_path=str(external / "model_comparison.png"))
            data["models"][0]["figures"] = {
                "pae": str(external / "pae_model_0.png"),
                "unsafe": str(root / "not_in_copy.png"),
            }
            data["models"][0]["overlay"] = {"path": str(external / "overlay_model_0.cif")}
            for rel in ("model_comparison.png", "pae_model_0.png", "overlay_model_0.cif"):
                path = local / rel
                path.write_bytes(b"local")
            (root / "not_in_copy.png").write_bytes(b"unsafe")

            local_run = make_report.rebase_result_paths(data, local_analysis / "results.json")

            self.assertEqual(local_run, local)
            self.assertEqual(data["run_dir"], str(local))
            self.assertEqual(data["figures"]["model_comparison"], str(local / "model_comparison.png"))
            self.assertEqual(data["models"][0]["figures"]["pae"], str(local / "pae_model_0.png"))
            self.assertNotIn("unsafe", data["models"][0]["figures"])
            self.assertEqual(data["models"][0]["overlay"]["path"], str(local / "overlay_model_0.cif"))

    def test_interface_residue_label_includes_insertion_code(self):
        with tempfile.TemporaryDirectory() as d:
            data = _results(Path(d))
            entry = data["models"][0]
            entry["interface_8A"]["residues_a"] = [
                {"chain": "A", "resnum": 42, "icode": "B", "resname": "TYR", "n_contacts": 1, "plddt": 81}
            ]
            rendered = make_report.build_interface_table(data, entry, "antigen")
            self.assertIn("A42BTYR", rendered)

    def test_html_warns_for_legacy_missing_analysis_provenance_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            data = _results(run)
            before = json.dumps(data, sort_keys=True)
            rendered = make_report.build_html(data, run)
            self.assertIn("Legacy analysis warning", rendered)
            self.assertIn("analysis_provenance", rendered)
            self.assertEqual(json.dumps(data, sort_keys=True), before)

    def test_html_does_not_warn_when_settings_analysis_provenance_exists(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            data = _results(run)
            data["settings"]["analysis_provenance"] = {"schema_version": 2, "sources": {}}
            rendered = make_report.build_html(data, run)
            self.assertNotIn("Legacy analysis warning", rendered)

    def test_model_tabs_are_keyboard_accessible_aria_tabs(self):
        with tempfile.TemporaryDirectory() as d:
            rendered = make_report.build_html(_results(Path(d)), Path(d))
            self.assertIn('role="tablist"', rendered)
            self.assertIn("role='tab'", rendered)
            self.assertIn("aria-selected='true'", rendered)
            self.assertIn("ArrowLeft", rendered)
            self.assertIn("ArrowRight", rendered)

    def test_generated_viewer_script_keeps_last_intent_after_ab_a_race(self):
        if not shutil.which("node"):
            self.skipTest("node is not installed")
        script = (
            r"""
const tabs = [
  { dataset: { key: 'A' }, classList: { toggle() {} }, setAttribute() {}, tabIndex: 0 },
  { dataset: { key: 'B' }, classList: { toggle() {} }, setAttribute() {}, tabIndex: -1 }
];
global.document = {
  querySelectorAll: () => tabs,
  querySelector: () => ({ addEventListener() {} }),
  getElementById: () => ({ textContent: '', style: {} }),
  addEventListener() {}
};
"""
            +
            make_report.JS_TEMPLATE
            .replace("__STRUCTS__", json.dumps({"A": "a-cif", "B": "b-cif",
                                               "overlay_A": "oa", "overlay_B": "ob"}))
            .replace("__META__", json.dumps({"__best__": "A", "A": {"label": "A"}, "B": {"label": "B"}}))
            + r"""
const loads = [];
global.RB = {
  load: (_key, data) => new Promise((resolve) => loads.push({ data, resolve })),
  setTheme: () => Promise.resolve(),
  diag: (msg, isErr) => { if (isErr) console.error(msg); }
};
(async () => {
  const first = showModel('A');
  if (loads.length !== 1) throw new Error('initial load not queued');
  loads[0].resolve();
  await first;
  const b = showModel('B');
  const a = showModel('A');
  if (loads.length !== 3) throw new Error('A/B/A did not queue final A load');
  loads[1].resolve();
  await Promise.resolve();
  loads[2].resolve();
  await Promise.all([b, a]);
  if (LOADED !== 'A' || CURRENT !== 'A') {
    throw new Error(`expected final A, got LOADED=${LOADED} CURRENT=${CURRENT}`);
  }
  const pendingB = showModel('B');
  const pendingOverlay = showOverlay();
  const finalA = showModel('A');
  loads[3].resolve();
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  if (loads.length !== 5) throw new Error('stale overlay superseded a newer model request');
  loads[4].resolve();
  await Promise.all([pendingB, pendingOverlay, finalA]);
  if (LOADED !== 'A') throw new Error('stale overlay changed final model');
  const latestB = showModel('B');
  const latestOverlay = showOverlay();
  loads[5].resolve();
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  if (loads[6]?.data !== 'ob') throw new Error('overlay did not use requested B');
  loads[6].resolve();
  await Promise.all([latestB, latestOverlay]);
  if (LOADED !== 'overlay_B' || CURRENT !== 'B') throw new Error('incorrect overlay state');
})().catch((e) => {
  console.error(e.stack || e.message);
  process.exit(1);
});
"""
        )
        proc = subprocess.run(["node", "-e", script], check=False, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_viewer_reloads_previous_model_after_model_or_overlay_load_failure(self):
        if not shutil.which("node"):
            self.skipTest("node is not installed")
        script = r"""
global.window = global;
global.document = {
  getElementById: () => ({ textContent: '', style: {} }),
  querySelectorAll: () => [],
  addEventListener() {}
};
let displayed = null;
let loadCalls = 0;
let failNext = null;
const viewer = { plugin: {
  clear: async () => { displayed = null; },
  builders: {
    data: { rawData: async ({ data }) => { loadCalls++; return data; } },
    structure: {
      parseTrajectory: async data => {
        if (data === failNext) {
          failNext = null;
          throw new Error('injected trajectory failure');
        }
        return data;
      },
      hierarchy: { applyPreset: async data => { displayed = data; } }
    }
  },
  managers: { camera: { reset() {} } }
} };
global.molstar = { Viewer: { create: async () => viewer } };
"""
        script += (ROOT / "assets" / "report.js").read_text(encoding="utf-8")
        script += (make_report.JS_TEMPLATE
                   .replace("__STRUCTS__", json.dumps({"A": "A", "B": "B", "overlay_A": "overlay_A"}))
                   .replace("__META__", "{}"))
        script += r"""
(async () => {
  await showModel('A');
  for (const failing of ['B', 'overlay_A']) {
    failNext = failing;
    if (failing === 'B') await showModel('B');
    else await showOverlay();
    if (displayed !== null) throw new Error('failure did not clear viewer');
    const before = loadCalls;
    await showModel('A');
    if (displayed !== 'A' || loadCalls !== before + 1) {
      throw new Error(`returning to A after ${failing} failure did not restore its structure`);
    }
    await showModel('A');
    if (loadCalls !== before + 1) throw new Error('already displayed model was unnecessarily reloaded');
  }
  // A failed request must also remain retryable on its own tab.
  failNext = 'B';
  await showModel('B');
  await showModel('B');
  if (displayed !== 'B') throw new Error('retrying the failed model did not restore its structure');
})().catch(e => { console.error(e.stack || e.message); process.exit(1); });
"""
        proc = subprocess.run(["node", "-e", script], check=False, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_interpretation_guide_uses_uncalibrated_thresholds_and_msa_empty(self):
        self.assertIn("탐색용 휴리스틱", make_report.INTERPRET_GUIDE)
        self.assertIn("--msa empty", make_report.INTERPRET_GUIDE)
        self.assertNotIn("--no-msa", make_report.INTERPRET_GUIDE)
        self.assertIn("친화도(Kd)", make_report.INTERPRET_GUIDE)

    def test_make_report_script_generates_report_from_rebased_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source"
            local = root / "local"
            local_analysis = local / "analysis"
            local_analysis.mkdir(parents=True)
            data = _results(source, figure_path=str(source / "model_comparison.png"))
            local_model = _model(local)
            data["predictions_dir"] = str(source / "predictions")
            data["models"][0].update({
                "cif": str(source / "predictions" / Path(local_model["cif"]).name),
                "confidence_json": str(source / "predictions" / "confidence_job_model_0.json"),
                "pae_npz": str(source / "predictions" / "pae_job_model_0.npz"),
            })
            (local / "model_comparison.png").write_bytes(b"png")
            results_path = local_analysis / "results.json"
            results_path.write_text(json.dumps(data), encoding="utf-8")
            outdir = local / "report"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "make_report.py"),
                    "--results",
                    str(results_path),
                    "--outdir",
                    str(outdir),
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            html = (outdir / "index.html").read_text(encoding="utf-8")
            self.assertIn("모델 비교 요약", html)
            self.assertIn("Legacy analysis warning", html)
            self.assertIn("primary_scope", (outdir / "summary.tsv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
