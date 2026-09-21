#!/usr/bin/env python
"""Build an interactive batch report (sortable table + Mol* detail view)."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report import (
    CSS,
    INTERPRET_GUIDE,
    dockq_badge,
    ipsae_badge,
    iptm_badge,
    plddt_badge,
    primary_scores,
    rebase_result_paths,
    validate_results,
)


def safe_job_result_path(root: Path, name: str) -> Path:
    """Return a contained results.json path for a user-visible job name."""
    if not name or name in {".", ".."}:
        raise ValueError("empty/dot job name")
    p = Path(name)
    if p.is_absolute() or len(p.parts) != 1 or any(part in {".", ".."} for part in p.parts):
        raise ValueError("job name must be one path segment")
    if "/" in name or "\\" in name:
        raise ValueError("job name must not contain path separators")
    res = root / name / "analysis" / "results.json"
    if not _within(res, root):
        raise ValueError("job path escapes jobs root")
    return res


def _within(path: Path, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False


def job_has_nanobody(job: dict) -> bool | None:
    """Infer binder presence for failed rows from manifest metadata.

    Unknown remains None so all-unknown failed batches keep the conservative
    binder-oriented table.
    """

    if isinstance(job.get("target_only"), bool):
        return not job["target_only"]
    if isinstance(job.get("has_nanobody"), bool):
        return job["has_nanobody"]
    if "nanobody_chain" in job:
        return bool(job["nanobody_chain"])
    if isinstance(job.get("nanobody_len"), int) and job["nanobody_len"] >= 0:
        return job["nanobody_len"] > 0
    if job.get("nanobody_file"):
        return True
    return None


def is_target_only_batch(rows: list[dict]) -> bool:
    known = [r.get("has_nanobody") for r in rows if r.get("has_nanobody") is not None]
    return bool(known) and all(value is False for value in known)


BATCH_CSS = """
.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0}
.toolbar input[type=text]{padding:7px 10px;border:1px solid var(--line);border-radius:8px;font-size:13px;min-width:240px}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{background:#e6ebf1}
th.sortable .sortbtn{all:unset;cursor:pointer;display:inline-block;width:100%;height:100%}
th.sortable .sortbtn:focus,tr.row:focus{outline:2px solid #1f5aa8;outline-offset:-2px}
th .arrow{color:#1f5aa8;font-size:10px}
tr.row{cursor:pointer}
tr.row:hover{background:#f0f6ff}
tr.row.sel{background:#e3f0ff}
#detail{display:none}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:10px}
.kpi{border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:#fff}
.kpi .n{font-size:11px;color:var(--muted)}
.kpi .v{font-size:17px;font-weight:600}
.backbtn{margin:6px 0 12px}
#molstar-viewer{position:relative}
.pill{display:inline-block;font-size:11px;border:1px solid var(--line);border-radius:20px;padding:1px 8px;color:#555;background:#fafbfc;margin-left:6px}
.rank{color:#999;font-size:11px}
"""

JS = r"""
const JOBS = __JOBS__;          // job name -> {cif, figs, report, m, ...}
const ROWS = __ROWS__;          // array of row dicts (sortable columns)
const COLS = __COLS__;          // [{key,label,type,fmt,higherBetter}]
let sortKey = __SORTKEY__, sortDir = -1, filter = '';
let selected = null;
let pendingSortFocus = null;

function fmtVal(v, type, nd) {
  if (v === null || v === undefined || v === '') return '-';
  if (type === 'num') {
    const x = Number(v);
    if (!isFinite(x)) return '-';
    return x.toFixed(nd === undefined ? (Math.abs(x) >= 100 ? 0 : 3) : nd);
  }
  return String(v);
}
function badgeClass(key, v) {
  if (v === null || v === undefined || v === '') return '';
  if (key === 'dockq') return 'b-' + (v >= 0.8 ? 'ok' : v >= 0.49 ? 'warn' : v >= 0.23 ? 'warn' : 'bad');
  if (key === 'ipsae' || key === 'ipsae_d0chn' || key === 'ipsae_d0dom') return 'b-' + (v >= 0.6 ? 'ok' : v >= 0.4 ? 'warn' : 'bad');
  if (key === 'pdockq2') return 'b-' + (v >= 0.8 ? 'ok' : v >= 0.5 ? 'warn' : 'bad');
  if (key === 'pdockq') return 'b-' + (v >= 0.5 ? 'ok' : v >= 0.23 ? 'warn' : 'bad');
  if (key === 'lis') return 'b-' + (v >= 0.35 ? 'ok' : v >= 0.2 ? 'warn' : 'bad');
  if (key === 'iptm_nb' || key === 'iptm_ag' || key === 'ptm' || key === 'iptm_d0chn') return 'b-' + (v >= 0.8 ? 'ok' : v >= 0.6 ? 'warn' : 'bad');
  if (key === 'nb_plddt' || key === 'cdr3_plddt') return 'b-' + (v >= 90 ? 'ok' : v >= 70 ? 'warn' : v >= 50 ? 'warn' : 'bad');
  if (key === 'iface_pae') return 'b-' + (v < 5 ? 'ok' : v < 10 ? 'warn' : 'bad');
  return '';
}
function esc(s) {
  return String(s === null || s === undefined ? '' : s).replace(/[&<>"']/g, ch => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));
}
function renderTable() {
  const q = filter.trim().toLowerCase();
  let rows = ROWS.filter(r => !q || (r.job + ' ' + (r.notes || '')).toLowerCase().includes(q));
  const col = COLS.find(c => c.key === sortKey) || {};
  const dir = (col.higherBetter === false) ? -sortDir : sortDir;   // iface PAE 는 낮을수록 좋음
  const mixedGroups = new Set(ROWS.map(r => JSON.stringify([r.group || null, r.primary_scope || null]))).size > 1;
  rows.sort((a, b) => {
    const A = a[sortKey], B = b[sortKey];
    const na = (A === null || A === undefined || A === ''), nb = (B === null || B === undefined || B === '');
    if (na && nb) return 0;
    if (na) return 1;              // missing values always last
    if (nb) return -1;
    if (typeof A === 'number' || typeof B === 'number') return (Number(A) - Number(B)) * dir;
    return String(A).localeCompare(String(B), undefined, { numeric: true, sensitivity: 'base' }) * dir;
  });
  const thead = document.getElementById('thead');
  thead.innerHTML = '<tr>' + COLS.map(c => {
    const actualDir = (c.higherBetter === false) ? -sortDir : sortDir;
    const arrow = (c.key === sortKey) ? (' <span class="arrow">' + (actualDir > 0 ? '▲' : '▼') + '</span>') : '';
    const aria = c.key === sortKey ? (actualDir > 0 ? 'ascending' : 'descending') : 'none';
    return `<th class="sortable" aria-sort="${aria}"><button type="button" class="sortbtn" data-sort-key="${esc(c.key)}">${c.label}${arrow}</button></th>`;
  }).join('') + '</tr>';
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = rows.map((r, i) => {
    const bestKey = (ROWS.some(r => r.iptm_nb !== null && r.iptm_nb !== undefined)) ? 'iptm_nb' : 'ptm';
    const isBest = (sortKey === bestKey && sortDir === -1 && i === 0 && !q && !mixedGroups);
    const cells = COLS.map(c => {
      if (c.key === 'job') {
        const star = isBest ? ' <span title="기본 정렬 상단 미리보기" aria-label="기본 정렬 상단 미리보기">★</span>' : '';
        const pill = r.status !== 'ok' ? ' <span class="pill">' + esc(r.status || '') + '</span>' : '';
        const note = r.notes ? '<div class="rank">' + esc(r.notes) + '</div>' : '';
        return `<td class="l">${esc(r.job)}${star}${pill}${note}</td>`;
      }
      const v = r[c.key];
      const cls = badgeClass(c.key, v);
      const badge = cls ? ` <span class="badge ${cls}"></span>` : '';
      const txt = esc(fmtVal(v, c.type, c.nd));
      return `<td>${txt}${badge}</td>`;
    });
    return `<tr class="row${r.job === selected ? ' sel' : ''}" tabindex="0" data-job="${esc(r.job)}">${cells.join('')}</tr>`;
  }).join('');
  document.getElementById('count').textContent = rows.length + ' / ' + ROWS.length + ' jobs';
  if (pendingSortFocus) {
    const btn = Array.from(document.querySelectorAll('[data-sort-key]')).find(x => x.dataset.sortKey === pendingSortFocus);
    pendingSortFocus = null;
    if (btn) btn.focus();
  }
}
function setSort(key, keepFocus) {
  if (sortKey === key) sortDir = -sortDir; else { sortKey = key; sortDir = -1; }
  if (keepFocus) pendingSortFocus = key;
  renderTable();
}
function setFilter(v) { filter = v; renderTable(); }
async function showOverlay() {
  const j = JOBS[selected];
  if (!j || !j.overlay) { RB.diag('이 job에는 참조 구조 겹침이 없습니다 (reference 미지정)'); return; }
  try { await RB.load('main', j.overlay, 'chain-id'); }
  catch (e) { RB.diag('showOverlay failed: ' + e.message, true); }
}
async function openJob(job) {
  selected = job;
  const j = JOBS[job];
  if (!j) return;
  document.getElementById('table-view').style.display = 'none';
  document.getElementById('detail').style.display = 'block';
  document.getElementById('detail-title').textContent = job + (j.subtitle ? ' - ' + j.subtitle : '');
  document.getElementById('detail-kpis').innerHTML = j.kpis;
  document.getElementById('detail-figs').innerHTML = j.figs;
  const reportLink = document.getElementById('detail-report-link');
  if (j.report) {
    reportLink.href = j.report;
    reportLink.style.display = '';
  } else {
    reportLink.removeAttribute('href');
    reportLink.style.display = 'none';
  }
  document.getElementById('detail-tables').innerHTML = j.tables || '';
  const card = document.getElementById('viewer-card');
  const missing = document.getElementById('viewer-missing');
  const host = document.getElementById('molstar-viewer');
  if (!j.cif) {
    if (missing) missing.style.display = 'block';
    if (host) host.style.display = 'none';
    if (missing) {
      missing.innerHTML = (j.status && j.status !== 'ok')
        ? ('이 job 은 실패했습니다 (status: ' + esc(j.status) + '). <span class="mono">./run.sh --batch ... --jobs ' + esc(job) + '</span> 로 재실행하세요.')
        : missing.innerHTML;
    }
    RB.diag('no structure for ' + job + ' (미완료/실패 job)');
  } else {
    if (missing) missing.style.display = 'none';
    if (host) host.style.display = 'block';
    try { await RB.load('main', j.cif, 'plddt-confidence'); }
    catch (e) { RB.diag('openJob failed: ' + e.message, true); }
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
  renderTable();
}
function backToTable() {
  document.getElementById('detail').style.display = 'none';
  document.getElementById('table-view').style.display = 'block';
  renderTable();
}
document.addEventListener('DOMContentLoaded', () => {
  renderTable();
  const tb = document.getElementById('tbody');
  if (tb) tb.addEventListener('click', e => {
    const tr = e.target.closest('tr[data-job]');
    if (tr && tr.dataset.job) openJob(tr.dataset.job);
  });
  if (tb) tb.addEventListener('keydown', e => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const tr = e.target.closest('tr[data-job]');
    if (tr && tr.dataset.job) { e.preventDefault(); openJob(tr.dataset.job); }
  });
  const th = document.getElementById('thead');
  if (th) {
    th.addEventListener('click', e => {
      const cell = e.target.closest('[data-sort-key]');
      if (cell) setSort(cell.dataset.sortKey, true);
    });
    th.addEventListener('keydown', e => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      const cell = e.target.closest('[data-sort-key]');
      if (cell) { e.preventDefault(); setSort(cell.dataset.sortKey, true); }
    });
  }
});
document.addEventListener('click', e => {
  const btn = e.target.closest('[data-theme]');
  if (btn && window.RB && RB.setTheme) {
    RB.setTheme('main', btn.dataset.theme).catch(err => RB.diag('setTheme 실패: ' + err.message, true));
  }
});
"""


def metrics_dockq_headline(d):
    """표시용 DockQ 대표값 (인터페이스 평균 headline 우선)."""
    if not d:
        return None
    return d.get("headline", d.get("best_dockq"))


def _num(v, nd=3):
    """HTML 셀용 안전 수치 포맷 (문자열/NaN/inf -> '-')."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:.{int(nd)}f}" if nd else str(int(x))


def _finite(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _stats(values: list[object], expected_n: int | None = None) -> dict:
    finite = [x for x in (_finite(v) for v in values) if x is not None]
    partial = expected_n is not None and len(finite) not in {0, expected_n}
    return {
        "n": len(finite),
        "mean": _mean(finite),
        "std": _sample_std(finite),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "expected_n": expected_n,
        "partial": partial,
    }


def _stat_summary(stats: dict, nd: int = 3) -> str:
    if not stats or not stats.get("n"):
        return "n=0"
    parts = [
        f"n={stats['n']}",
        f"mean={_num(stats.get('mean'), nd)}",
        f"min={_num(stats.get('min'), nd)}",
        f"max={_num(stats.get('max'), nd)}",
    ]
    if stats.get("n", 0) > 1:
        parts.insert(2, f"std={_num(stats.get('std'), nd)}")
    else:
        parts.append("single sample; descriptive only")
    return ", ".join(parts)


def ensemble_metric_stats(results: dict) -> dict[str, dict]:
    """Return descriptive stats over all finite model values for batch deltas."""

    models = results.get("models") or []
    expected_n = len(models)
    rows = {
        "iptm_nb": [],
        "ipsae": [],
        "pdockq2": [],
        "cdr3_plddt": [],
        "nb_plddt": [],
    }
    nb_chain = results.get("nanobody_chain")
    for model in models:
        primary = primary_scores(model, results)
        rows["iptm_nb"].append(primary.get("iptm"))
        rows["ipsae"].append(primary.get("ipsae"))
        rows["pdockq2"].append(((model.get("ipsae") or {}).get("max") or {}).get("pdockq2"))
        rows["cdr3_plddt"].append((model.get("cdr_mean_plddt") or {}).get("CDR3"))
        rows["nb_plddt"].append(((model.get("plddt") or {}).get("per_chain") or {}).get(nb_chain, {}).get("mean"))
    return {key: _stats(values, expected_n) for key, values in rows.items()}


def ensemble_score_sources(results: dict) -> dict[str, object]:
    """Return source sets for model-level primary metrics used in ensemble deltas."""

    iptm_sources = set()
    ipsae_sources = set()
    for model in results.get("models") or []:
        primary = primary_scores(model, results)
        if primary.get("iptm") is not None:
            iptm_sources.add(str(primary.get("iptm_source") or "missing"))
        if primary.get("ipsae") is not None:
            ipsae_sources.add(str(primary.get("ipsae_source") or "missing"))
    return {
        "iptm": sorted(iptm_sources),
        "ipsae": sorted(ipsae_sources),
        "consistent": len(iptm_sources) <= 1 and len(ipsae_sources) <= 1,
    }


def _metric_stats_kpis(stats: dict[str, dict], metric_labels: dict[str, str]) -> str:
    cards = []
    for key, label in metric_labels.items():
        st = stats.get(key) or {}
        cards.append(
            "<div class=\"kpi\"><div class=\"n\">"
            f"{html.escape(label)} sampling spread</div>"
            f"<div class=\"v\" style=\"font-size:12px\">{html.escape(_stat_summary(st))}</div></div>"
        )
    return "".join(cards)


def collect(results_path: Path, job: dict, report_rel: str, fig_rel_dir: str, outdir: Path,
            embed: bool = True):
    """Build the JS payload for one job."""
    if not results_path.exists():
        return {
            "row": {"job": job["name"], "status": "미완료",
                    "notes": job.get("notes") or "", "has_nanobody": job_has_nanobody(job),
                    "group": tuple(job.get("target_lens") or []) or (job.get("reference"),)},
            "joins": failed_detail_payload(
                job.get("name"), "미완료", "analysis/results.json not found", report_rel
            ),
        }
    results = json.loads(results_path.read_text(encoding="utf-8"))
    local_run_dir = rebase_result_paths(results, results_path)
    validate_results(results, local_run_dir)
    best = results["models"][0]
    mx = best["ipsae"].get("max") or {}
    iface = best["interface_8A"] or {}
    nb_chain = results["nanobody_chain"]
    cdr3 = (best.get("cdr_mean_plddt") or {}).get("CDR3")
    dockq = metrics_dockq_headline(best.get("dockq"))
    primary = primary_scores(best, results)
    iptm_nb = primary.get("iptm")
    iptm_ag = best["boltz_pair_iptm"]["antigen_in_nanobody_frame"]
    primary_ipsae = primary.get("ipsae")
    metric_stats = ensemble_metric_stats(results)
    score_sources = ensemble_score_sources(results)

    notes = job.get("notes") or ""
    m_nm = re.match(r"\s*m(\d+)\s*:", notes)
    n_mut = int(m_nm.group(1)) if m_nm else None

    raw_src = primary.get("ipsae_source") or ((best.get("ipsae") or {}).get("source")) or "-"
    src = {"official": "공식", "official:max": "공식",
           "builtin-fallback": "내장 폴백", "builtin-fallback:max": "내장 폴백",
           "builtin-union": "내장 union", "missing": "-", "-": "-"}.get(raw_src, raw_src)
    row = {
        "job": job["name"], "status": "ok", "notes": notes, "n_mut": n_mut,
        "ipsae_src": src, "has_nanobody": results.get("nanobody_chain") is not None,
        "group": comparison_group(results, job),
        "iptm_nb": iptm_nb, "iptm_ag": iptm_ag, "ptm": best["boltz"].get("ptm"),
        "ipsae": primary_ipsae, "ipsae_d0chn": mx.get("ipsae_d0chn"),
        "ipsae_d0dom": mx.get("ipsae_d0dom"), "iptm_d0chn": mx.get("iptm_d0chn"),
        "pdockq": mx.get("pdockq"), "pdockq2": mx.get("pdockq2"), "lis": mx.get("lis"),
        "nb_plddt": (best["plddt"]["per_chain"].get(nb_chain) or {}).get("mean"),
        "cdr3_plddt": cdr3, "iface_pae": iface.get("pae_mean_ab"),
        "n_contacts": iface.get("n_contacts"), "dockq": dockq,
        "n_models": len(results["models"]),
        "primary_scope": primary.get("scope"),
        "primary_source": primary.get("source"),
        "primary_ipsae_source": primary.get("ipsae_source"),
        "metric_stats": metric_stats,
        "score_sources": score_sources,
        "run_params": (results.get("settings") or {}).get("run_params") or {},
        "analysis_settings": comparison_settings(results),
        "rank_key": results.get("rank_key"),
        "best_model_index": results.get("best_model_index"),
    }

    # figures copied into the batch report folder
    figs_html = []
    fig_src = best.get("figures") or {}
    for key, cap in (("pae", "PAE 히트맵"), ("plddt", "잔기별 pLDDT"),
                     ("pae_iface", "인터페이스 PAE"), ("ipsae_byres", "잔기별 ipSAE")):
        p = fig_src.get(key)
        if not p or not Path(p).exists():
            continue
        dest = outdir / "figures" / job["name"] / f"{key}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        rel = f"figures/{job['name']}/{key}.png"
        figs_html.append(f"<figure style='margin:0'><img class='fig' src='{html.escape(rel, quote=True)}' "
                         f"alt='{html.escape(str(cap), quote=True)}'>"
                         f"<figcaption class='note'>{html.escape(str(cap))}</figcaption></figure>")

    def kpi(name, val, badge=None, nd=3):
        b = f' <span class="badge {badge[0]}">{badge[1]}</span>' if badge else ""
        return (f'<div class="kpi"><div class="n">{html.escape(str(name))}</div>'
                f'<div class="v">{_num(val, nd)}{b}</div></div>')

    css, lab = iptm_badge(iptm_nb)
    css_s, lab_s = ipsae_badge(primary_ipsae)
    css_q, lab_q = dockq_badge(dockq)
    css_p, lab_p = plddt_badge((best["plddt"]["per_chain"].get(nb_chain) or {}).get("mean"))
    css_c, lab_c = plddt_badge(cdr3)
    if results.get("nanobody_chain"):
        kpis = "".join([
            kpi((primary.get("iptm_label") or "대표 ipTM") + " (best preview)", iptm_nb, (css, lab)),
            kpi((primary.get("ipsae_label") or "대표 ipSAE") + " (best preview)", primary_ipsae, (css_s, lab_s)),
            kpi("pDockQ2", mx.get("pdockq2"), nd=3),
            kpi("나노바디 pLDDT", (best["plddt"]["per_chain"].get(nb_chain) or {}).get("mean"),
                (css_p, lab_p), 1),
            kpi("CDR3 pLDDT", cdr3, (css_c, lab_c), 1),
            kpi("인터페이스 PAE (Å)", iface.get("pae_mean_ab"), nd=2),
            kpi("인터페이스 컨택트", iface.get("n_contacts"), nd=0),
            kpi("DockQ", dockq, (css_q, lab_q) if dockq is not None else None),
            kpi("ipTM (항원|나노바디)", iptm_ag, nd=3),
            kpi("pTM", best["boltz"].get("ptm"), nd=3),
        ])
        kpis += _metric_stats_kpis(metric_stats, {
            "iptm_nb": "ipTM(nb|ag)",
            "ipsae": "ipSAE",
            "pdockq2": "pDockQ2",
            "nb_plddt": "nanobody pLDDT",
            "cdr3_plddt": "CDR3 pLDDT",
        })
    else:
        # R4-03: target-only 상세는 pTM/전체·체인 pLDDT 중심으로
        chain_cards = "".join(
            kpi(f"체인 {c['id']} 평균 pLDDT",
                (best["plddt"]["per_chain"].get(c["id"]) or {}).get("mean"), None, 1)
            for c in results["chains"])
        kpis = "".join([
            kpi("pTM (best preview)", best["boltz"].get("ptm"), nd=3),
            kpi("전체 평균 pLDDT", best["plddt"].get("mean"), None, 1),
            kpi("DockQ", dockq, (css_q, lab_q) if dockq is not None else None),
            chain_cards,
        ])

    cdrs = results.get("cdr") or {}
    cdr_html = ""
    if cdrs.get("available") and results.get("nanobody_chain"):
        vals = best.get("cdr_mean_plddt") or {}
        rows = "".join(
            f"<tr><td class='l'>{html.escape(str(k))}</td>"
            f"<td class='l mono'>{html.escape(str(v))}</td><td>{_num(vals.get(k), 1)}</td></tr>"
            for k, v in (cdrs.get("cdr_sequences") or {}).items())
        cdr_html = (f"<h4>CDR 서열 (IMGT)</h4><div class='tw'><table><thead><tr><th class='l'>CDR</th>"
                    f"<th class='l'>서열</th><th>평균 pLDDT</th></tr></thead><tbody>{rows}</tbody></table></div>")
    iface_html = ""
    if results.get("nanobody_chain"):
        antigen_chains = results.get("antigen_chains") or [results["antigen_chain"]]
        antigen_label = ",".join(antigen_chains)
        antigen_title = "항원 union" if len(antigen_chains) > 1 else "항원"
        for which, chain in ((antigen_title, antigen_label), ("나노바디", nb_chain)):
            key = "residues_b" if which == "나노바디" else "residues_a"
            res = (iface or {}).get(key) or []
            if not res:
                continue
            rows = "".join(
                f"<tr><td class='l'>{html.escape(str(r.get('chain')))}{_num(r.get('resnum'), 0)}"
                f"{html.escape(str(r.get('resname')))}</td>"
                f"<td>{_num(r['n_contacts'], 0)}</td><td>{_num(r['plddt'], 1)}</td></tr>"
                for r in res)
            iface_html += (f"<h4>{which} 인터페이스 잔기 ({html.escape(str(chain))})</h4>"
                           f"<div class='tw' style='max-height:260px;overflow:auto'>"
                           f"<table><thead><tr><th class='l'>잔기</th><th>접촉</th><th>pLDDT</th></tr></thead>"
                           f"<tbody>{rows}</tbody></table></div>")
    else:
        cdr_html = ("<p class='note'>target-only 입력이라 CDR/인터페이스(나노바디) 지표는 "
                    "계산되지 않습니다. pTM 과 체인별 pLDDT 를 보세요.</p>")

    cif_text = ""
    overlay_text = ""
    if embed:
        cif_text = Path(best["cif"]).read_text(encoding="utf-8") if Path(best["cif"]).exists() else ""
        ov = best.get("overlay") or {}
        if ov.get("path") and Path(ov["path"]).exists():
            overlay_text = Path(ov["path"]).read_text(encoding="utf-8")
    chain_line = ", ".join(f"{c['id']}={c['role']}({c['n_res']}aa)" for c in results["chains"])
    return {
        "row": row,
        "joins": {
            "cif": cif_text,
            "overlay": overlay_text,
            "figs": "".join(figs_html),
            "kpis": kpis,
            "tables": cdr_html + iface_html,
            "report": report_rel,
            "subtitle": f"{chain_line} · model_{best['index']} · {len(results['models'])} models",
        },
    }


COLUMNS = [
    ("job", "Job", "str", None, True),
    ("n_mut", "변이수", "num", 0, True),
    ("iptm_nb", "ipTM<br>(nb|ag)", "num", 3, True),
    ("d_iptm", "ΔipTM<br>vs 대조<br><span style=\"font-weight:400\">(조건 일치 시)</span>", "num", 3, True),
    ("ipsae", "ipSAE", "num", 3, True),
    ("ipsae_src", "ipSAE<br>출처", "str", None, False),
    ("primary_scope", "primary<br>scope", "str", None, True),
    ("primary_ipsae_source", "primary<br>ipSAE source", "str", None, True),
    ("d_ipsae", "ΔipSAE<br>vs 대조<br><span style=\"font-weight:400\">(조건 일치 시)</span>", "num", 3, True),
    ("ipsae_d0chn", "ipSAE<br>d0chn", "num", 3, True),
    ("ipsae_d0dom", "ipSAE<br>d0dom", "num", 3, True),
    ("pdockq2", "pDockQ2", "num", 3, True),
    ("d_pdockq2", "ΔpDockQ2", "num", 3, True),
    ("pdockq", "pDockQ", "num", 3, True),
    ("lis", "LIS", "num", 3, True),
    ("iptm_ag", "ipTM<br>(ag|nb)", "num", 3, True),
    ("ptm", "pTM", "num", 3, True),
    ("iptm_d0chn", "ipTM<br>d0chn", "num", 3, True),
    ("nb_plddt", "nb<br>pLDDT", "num", 1, True),
    ("cdr3_plddt", "CDR3<br>pLDDT", "num", 1, True),
    ("d_cdr3", "ΔCDR3<br>pLDDT", "num", 1, True),
    ("iface_pae", "iface PAE(Å)<br><span style=\"font-weight:400\">(나노바디 잔기,<br>항원 기준)</span>", "num", 2, False),
    ("n_contacts", "컨택트", "num", 0, True),
    ("dockq", "DockQ", "num", 3, True),
    ("n_models", "models", "num", 0, True),
]


def comparison_group(results: dict, job: dict | None = None) -> tuple:
    """Comparable designs must share the actual antigen chain content, not only lengths."""
    chains = {c.get("id"): c for c in (results.get("chains") or [])}
    antigen_chains = results.get("antigen_chains") or (
        [results.get("antigen_chain")] if results.get("antigen_chain") else [])
    group = []
    for ch in antigen_chains:
        c = chains.get(ch) or {}
        seq = c.get("sequence")
        if not seq:
            return ("missing-antigen-sequence", tuple(job.get("target_lens") or []) if job else ())
        group.append((ch, seq))
    return tuple(group)


def comparison_settings(results: dict) -> dict:
    settings = results.get("settings") or {}
    constraints = dict(settings.get("input_constraints") or {})
    if not constraints.get("antigen_chain_ids") and results.get("antigen_chains"):
        constraints["antigen_chain_ids"] = list(results["antigen_chains"])
    if "nanobody_chain_id" not in constraints and "nanobody_chain" in results:
        constraints["nanobody_chain_id"] = results.get("nanobody_chain")
    return {
        "pae_cutoff": settings.get("pae_cutoff"),
        "dist_cutoff": settings.get("dist_cutoff"),
        "contact_cutoff": settings.get("contact_cutoff"),
        "reference": bool(settings.get("reference")),
        "dockq_mapping": settings.get("dockq_mapping"),
        "dockq_allowed_mismatches": settings.get("dockq_allowed_mismatches"),
        "effective_msa_policy": settings.get("effective_msa_policy"),
        "input_constraints": constraints,
        "analysis_provenance": settings.get("analysis_provenance"),
        "ranking_policy": settings.get("ranking_policy"),
        "score_policy": results.get("rank_key"),
    }


def _chain_key(ids) -> tuple[str, ...]:
    if isinstance(ids, (list, tuple)):
        return tuple(str(i) for i in ids)
    if ids is None:
        return ()
    return (str(ids),)


def _antigen_ids(constraints: dict | None) -> set[str]:
    """Read explicit chain roles without assuming a positional binder order."""

    if not isinstance(constraints, dict):
        return set()
    explicit = constraints.get("antigen_chain_ids") or constraints.get("antigen_chains")
    if explicit:
        return {str(chain) for chain in _chain_key(explicit)}
    if constraints.get("target_only"):
        return {str(chain) for chain in constraints.get("chain_ids") or []}
    return set()


def _project_msa_policy(policy: dict | None, constraints: dict | None) -> tuple[bool, object]:
    """Comparable antigen MSA policy, excluding expected binder-only variation."""
    if not isinstance(policy, dict):
        return False, "missing effective MSA policy"
    if policy.get("available") is not True:
        return False, policy.get("reason") or "effective MSA policy unavailable"
    if policy.get("delta_eligible") is not True:
        reasons = policy.get("delta_ineligible_reasons") or policy.get("reason")
        return False, reasons or "effective MSA policy not delta-eligible"
    antigen_ids = _antigen_ids(constraints)
    if not antigen_ids:
        return False, "missing antigen chain ids for MSA comparison"
    chains = []
    for chain in policy.get("chains") or []:
        ids = _chain_key(chain.get("ids"))
        if not set(ids) & antigen_ids:
            continue
        msa = chain.get("msa") or {}
        if msa.get("delta_eligible") is not True:
            return False, f"antigen MSA not delta-eligible: {','.join(ids) or '?'}"
        chains.append({
            "ids": ids,
            "sequence_hash": chain.get("sequence_hash"),
            "length": chain.get("length"),
            "msa_class": msa.get("class"),
            "msa_pairing": msa.get("pairing"),
            "msa_sequence_hash": msa.get("sequence_hash"),
            "msa_content_sha256": msa.get("content_sha256"),
            "templates_hash": chain.get("templates_hash"),
            "modifications_hash": chain.get("modifications_hash"),
            "properties_hash": chain.get("properties_hash"),
            "cyclic": bool(chain.get("cyclic", False)),
        })
    if not chains:
        return False, "missing antigen MSA chain provenance"
    return True, {
        "schema_version": policy.get("schema_version"),
        "requested_msa_policy": policy.get("requested_msa_policy"),
        "pairing_policy": policy.get("pairing_policy"),
        "constraints_hash": policy.get("constraints_hash"),
        "constraints_count": policy.get("constraints_count"),
        "antigen_chains": sorted(chains, key=lambda c: c["ids"]),
    }


def _project_input_constraints(value: dict | None) -> tuple[bool, object]:
    if not isinstance(value, dict) or not value:
        return False, "missing input constraints"
    target_only = bool(value.get("target_only"))
    antigen_ids = _antigen_ids(value)
    if not antigen_ids:
        return False, "missing antigen chain ids for input constraints"
    tmpl = []
    for item in value.get("templates_modifications_properties") or []:
        ids = _chain_key(item.get("ids"))
        if not set(ids) & antigen_ids:
            continue
        tmpl.append({
            "ids": ids,
            "templates_hash": item.get("templates_hash"),
            "modifications_hash": item.get("modifications_hash"),
            "properties_hash": item.get("properties_hash"),
            "cyclic": bool(item.get("cyclic", False)),
        })
    return True, {
        "schema_version": value.get("schema_version"),
        "antigen_chain_ids": tuple(sorted(antigen_ids)),
        "antigen_chain_lengths": {k: (value.get("chain_lengths") or {}).get(k) for k in sorted(antigen_ids)},
        "target_only": target_only,
        "yaml_constraints_hash": value.get("yaml_constraints_hash"),
        "yaml_constraints_count": value.get("yaml_constraints_count"),
        "templates_modifications_properties": sorted(tmpl, key=lambda x: x["ids"]),
    }


def _project_analysis_provenance(value: dict | None) -> tuple[bool, object]:
    if not isinstance(value, dict):
        return False, "missing analysis provenance"
    if value.get("schema_version") != 2:
        return False, "analysis provenance must use schema_version 2"
    sources = value.get("sources")
    packages = value.get("packages")
    if not isinstance(sources, dict) or not sources:
        return False, "missing analysis provenance sources"
    if not isinstance(packages, dict) or not packages:
        return False, "missing analysis provenance packages"
    return True, {
        "schema_version": 2,
        "sources": sources,
        "packages": packages,
    }


def _project_runtime_provenance(value: dict | None) -> tuple[bool, object]:
    if not isinstance(value, dict):
        return False, "missing runtime provenance"
    required = (
        "schema_version",
        "python",
        "boltz_version",
        "torch_version",
        "boltz_executable_sha256",
        "kernel_mode",
    )
    missing = [key for key in required if value.get(key) in (None, "")]
    if missing:
        return False, "incomplete runtime provenance: " + ", ".join(missing)
    return True, {key: value.get(key) for key in required}


def _project_msa_subsample(params: dict) -> tuple[bool, object]:
    value = params.get("msa_subsample", 0)
    if type(value) is not int or value < 0:
        return False, "invalid run param: msa_subsample"
    return True, value


def delta_eligibility(row: dict, control: dict) -> tuple[bool, str]:
    """Return whether WT-mutant deltas can be interpreted as a matched comparison."""
    if not control:
        return False, "no matched control"
    if row.get("group") != control.get("group"):
        return False, "different antigen group"
    if (row.get("group") or (None,))[0] == "missing-antigen-sequence":
        return False, "missing antigen sequence provenance"
    rp = row.get("run_params") or {}
    cp = control.get("run_params") or {}
    keys = ("samples", "seed", "steps", "recycles", "msa", "parallel_samples", "devices")
    missing = [k for k in keys if rp.get(k) in (None, "") or cp.get(k) in (None, "")]
    if missing:
        return False, "missing run params: " + ", ".join(missing)
    mismatched = [k for k in keys if rp.get(k) != cp.get(k)]
    if mismatched:
        return False, "mismatched run params: " + ", ".join(mismatched)
    ok_sub_r, sub_r = _project_msa_subsample(rp)
    ok_sub_c, sub_c = _project_msa_subsample(cp)
    if not ok_sub_r or not ok_sub_c:
        return False, str(sub_r if not ok_sub_r else sub_c)
    if sub_r != sub_c:
        return False, "mismatched run params: msa_subsample"
    ok_rt_r, rt_r = _project_runtime_provenance(rp.get("runtime"))
    ok_rt_c, rt_c = _project_runtime_provenance(cp.get("runtime"))
    if not ok_rt_r or not ok_rt_c:
        return False, str(rt_r if not ok_rt_r else rt_c)
    if rt_r != rt_c:
        return False, "mismatched runtime provenance"
    rs = row.get("analysis_settings") or {}
    cs = control.get("analysis_settings") or {}
    required_settings = (
        "pae_cutoff",
        "dist_cutoff",
        "contact_cutoff",
        "effective_msa_policy",
        "input_constraints",
        "analysis_provenance",
        "ranking_policy",
        "score_policy",
    )
    missing_settings = [
        k for k in required_settings
        if rs.get(k) is None or cs.get(k) is None or (k == "input_constraints" and (not rs.get(k) or not cs.get(k)))
    ]
    if missing_settings:
        return False, "missing analysis provenance: " + ", ".join(missing_settings)
    ok_msa_r, msa_r = _project_msa_policy(rs.get("effective_msa_policy"), rs.get("input_constraints"))
    ok_msa_c, msa_c = _project_msa_policy(cs.get("effective_msa_policy"), cs.get("input_constraints"))
    if not ok_msa_r or not ok_msa_c:
        return False, "MSA delta not eligible: " + str(msa_r if not ok_msa_r else msa_c)
    ok_ic_r, ic_r = _project_input_constraints(rs.get("input_constraints"))
    ok_ic_c, ic_c = _project_input_constraints(cs.get("input_constraints"))
    if not ok_ic_r or not ok_ic_c:
        return False, "input constraints not comparable: " + str(ic_r if not ok_ic_r else ic_c)
    ok_ap_r, ap_r = _project_analysis_provenance(rs.get("analysis_provenance"))
    ok_ap_c, ap_c = _project_analysis_provenance(cs.get("analysis_provenance"))
    if not ok_ap_r or not ok_ap_c:
        return False, "analysis provenance not comparable: " + str(ap_r if not ok_ap_r else ap_c)
    comparable = dict(rs)
    comparable_control = dict(cs)
    comparable["effective_msa_policy"] = msa_r
    comparable_control["effective_msa_policy"] = msa_c
    comparable["input_constraints"] = ic_r
    comparable_control["input_constraints"] = ic_c
    comparable["analysis_provenance"] = ap_r
    comparable_control["analysis_provenance"] = ap_c
    mismatched_settings = [k for k in required_settings if comparable.get(k) != comparable_control.get(k)]
    if mismatched_settings:
        return False, "mismatched analysis settings: " + ", ".join(mismatched_settings)
    if row.get("n_models") != control.get("n_models"):
        return False, "mismatched n_models"
    if row.get("rank_key") != control.get("rank_key"):
        return False, "mismatched model-selection rule"
    if row.get("primary_scope") != control.get("primary_scope"):
        return False, "mismatched primary-score scope"
    if row.get("primary_source") != control.get("primary_source"):
        return False, "mismatched primary-score source"
    if not (row.get("score_sources") or {}).get("consistent", True):
        return False, "mixed per-model score sources"
    if not (control.get("score_sources") or {}).get("consistent", True):
        return False, "control has mixed per-model score sources"
    return True, "matched sampling/MSA/selection"


def append_delta_exclusion_cards(rows: list[dict], jobs_js: dict) -> None:
    """Attach each row's own exclusion reason to its detail panel."""

    for row in rows:
        reason = row.get("delta_reason")
        if (row.get("job") not in jobs_js or row.get("delta_ok") or not reason
                or not row.get("has_nanobody", True)):
            continue
        jobs_js[row["job"]]["kpis"] += (
            "<div class=\"kpi\"><div class=\"n\">Δ 계산 제외</div>"
            f"<div class=\"v\" style=\"font-size:13px\">{html.escape(str(reason))}</div></div>"
        )


def failed_detail_payload(job_name: object, status: object, detail: object, report_rel: str | None = None) -> dict:
    """Return an escaped detail payload for a failed/incomplete batch row."""

    safe_status = html.escape(str(status or "불완전"))
    safe_detail = html.escape(str(detail or "detail unavailable"))
    return {
        "cif": "",
        "overlay": "",
        "figs": "",
        "status": str(status or "불완전"),
        "kpis": (
            "<div class=\"kpi\"><div class=\"n\">status</div>"
            f"<div class=\"v\" style=\"font-size:13px\">{safe_status}</div></div>"
            "<div class=\"kpi\"><div class=\"n\">detail</div>"
            f"<div class=\"v\" style=\"font-size:12px;white-space:pre-wrap\">{safe_detail}</div></div>"
        ),
        "tables": "",
        "report": report_rel or "",
        "subtitle": f"failed/incomplete: {safe_status}",
    }


def assign_control_deltas(rows: list[dict]) -> dict:
    """Find matched controls by group and attach ensemble-mean deltas to rows."""

    def _is_control(r):
        return (r.get("status") == "ok"
                and (r.get("n_mut") == 0
                     or re.search(r"(_WT|_wt|_control|_ctrl|_NC)$", r.get("job", ""))))

    controls = {}
    dup_groups = set()
    for r in rows:
        if not _is_control(r):
            continue
        g = r.get("group")
        if g in controls:
            dup_groups.add(g)
        else:
            controls[g] = r
    for g in dup_groups:
        controls.pop(g, None)

    delta_keys = {"d_iptm": "iptm_nb", "d_ipsae": "ipsae", "d_pdockq2": "pdockq2",
                  "d_cdr3": "cdr3_plddt", "d_plddt": "nb_plddt"}
    for r in rows:
        c = controls.get(r.get("group"))
        eligible, reason = delta_eligibility(r, c) if c else (False, "no matched control")
        r["delta_ok"] = bool(eligible)
        delta_notes = []
        for dk, base in delta_keys.items():
            if not eligible:
                r[dk] = None
                continue
            row_stat = (r.get("metric_stats") or {}).get(base) or {}
            ctrl_stat = (c.get("metric_stats") or {}).get(base) or {}
            row_mean, ctrl_mean = _finite(row_stat.get("mean")), _finite(ctrl_stat.get("mean"))
            if row_stat.get("partial") or ctrl_stat.get("partial"):
                r[dk] = None
                delta_notes.append(f"{base} has partially missing ensemble values")
                continue
            if row_stat.get("n") != r.get("n_models") or ctrl_stat.get("n") != c.get("n_models"):
                r[dk] = None
                delta_notes.append(f"{base} lacks full finite ensemble")
                continue
            r[dk] = round(row_mean - ctrl_mean, 4) if (row_mean is not None and ctrl_mean is not None) else None
        r["delta_reason"] = reason if not eligible else (
            "matched sampling/MSA/selection; mean deltas use full finite ensembles"
            + (("; " + "; ".join(delta_notes)) if delta_notes else "")
        )
    return controls


def select_active_columns(rows: list[dict]) -> list[tuple]:
    active_cols = list(COLUMNS)
    target_only_batch = is_target_only_batch(rows)
    if target_only_batch:
        # RR09: 모든 job 이 나노바디 없음 -> 나노바디 전용 열 제거 (표/TSV 공통)
        drop = {
            "iptm_nb", "d_iptm", "ipsae", "ipsae_src", "primary_scope",
            "primary_ipsae_source", "d_ipsae", "ipsae_d0chn", "ipsae_d0dom",
            "pdockq2", "d_pdockq2", "pdockq", "lis", "iptm_ag", "iptm_d0chn",
            "nb_plddt", "d_plddt", "cdr3_plddt", "d_cdr3", "iface_pae", "n_contacts",
        }
        active_cols = [c for c in COLUMNS if c[0] not in drop]
    return active_cols


def write_summary_tsv(path: Path, rows: list[dict], active_cols: list[tuple]) -> None:
    """Write the visible columns plus the audit-only delta exclusion reason."""

    keys = [key for key, *_ in active_cols]
    stat_metrics = tuple(
        metric for metric in ("iptm_nb", "ipsae", "pdockq2", "cdr3_plddt", "nb_plddt")
        if metric in keys
    )
    stat_fields = ("n", "mean", "std", "min", "max")
    stat_keys = [f"{metric}_{field}" for metric in stat_metrics for field in stat_fields]
    summary_keys = [*keys, *stat_keys, "delta_reason"]
    num_like = {key for key, _label, kind, _digits, _default in active_cols if kind == "num"}
    num_like.update(stat_keys)
    lines = ["\t".join(summary_keys)]
    for row in rows:
        cells = []
        for key in summary_keys:
            value = row.get(key)
            if value is None:
                for metric in stat_metrics:
                    prefix = f"{metric}_"
                    if key.startswith(prefix):
                        value = ((row.get("metric_stats") or {}).get(metric) or {}).get(key[len(prefix):])
                        break
            if value is None:
                cells.append("")
            elif key in num_like:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    cells.append("")
                else:
                    cells.append(f"{number:g}" if math.isfinite(number) else "")
            else:
                cells.append(str(value))
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--jobs-dir", required=True, type=Path, help="e.g. outputs/batch_myscreen")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--no-embed", action="store_true",
                    help="3D 구조 파일(CIF)을 HTML 에 넣지 않음 - job 이 많을 때 파일 크기 절감")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    outdir = args.outdir or (args.jobs_dir / "report")
    outdir.mkdir(parents=True, exist_ok=True)
    for asset in ("molstar.js", "molstar.css", "report.js"):
        shutil.copy2(Path(__file__).resolve().parent.parent / "assets" / asset, outdir / asset)

    rows, jobs_js = [], {}
    broken = []
    jobs_root = args.jobs_dir.resolve()
    for job in manifest["jobs"]:
        name = str(job.get("name") or "")
        try:
            res = safe_job_result_path(jobs_root, name)
        except ValueError:
            broken.append(name)
            print(f"[batch-report] 경고: 안전하지 않은 job 이름을 건너뜁니다: {name!r}")
            rows.append({"job": name, "status": "이름 오류",
                         "notes": job.get("notes") or "", "has_nanobody": job_has_nanobody(job),
                         "group": tuple(job.get("target_lens") or [])})
            jobs_js[name] = failed_detail_payload(name, "이름 오류", "unsafe job name")
            continue
        rep = f"../{name}/report/index.html"
        try:
            payload = collect(res, job, rep, "figures", outdir, embed=not args.no_embed)
        except Exception as exc:  # noqa: BLE001 - 한 job 이 깨져도 나머지 표를 살린다
            broken.append(job["name"])
            print(f"[batch-report] 경고: {job['name']} 결과를 읽지 못했습니다 ({exc})")
            payload = {"row": {"job": job["name"], "status": "불완전",
                               "notes": job.get("notes") or "", "has_nanobody": job_has_nanobody(job),
                               "group": tuple(job.get("target_lens") or []),
                               "delta_reason": str(exc)},
                       "joins": failed_detail_payload(job.get("name"), "불완전", exc, rep)}
        rows.append(payload["row"])
        if payload["joins"]:
            jobs_js[job["name"]] = payload["joins"]
    if broken:
        print(f"[batch-report] 읽기 실패 {len(broken)}건: {', '.join(broken[:5])}")

    # 대조군(야생형) 탐색 -> delta 컬럼 계산
    # H-02: 항원이 다른 job 을 대조군으로 삼지 않도록 '그룹(=항원 서열 길이 조합)'별로 계산한다.
    controls = assign_control_deltas(rows)
    multi_groups = len({r.get("group") for r in rows}) > 1
    if multi_groups and controls:
        print(f"[batch-report] 항원 그룹 {len({r.get('group') for r in rows})}개 -> Δ 는 "
              f"그룹별 대조군 기준으로만 계산합니다")
    if controls:
        print("[batch-report] 대조군: " + ", ".join(f"{c['job']}(그룹 {g})"
              for g, c in list(controls.items())[:3]) + (" ..." if len(controls) > 3 else ""))
        for r in rows:
            d = r.get("d_ipsae")
            if d is None or r.get("n_mut") == 0 or r["job"] not in jobs_js:
                continue
            c = controls.get(r.get("group"))
            if c is None or c["job"] == r["job"]:
                continue
            card = (f'<div class="kpi"><div class="n">ΔipSAE (조건 일치 대조군 '
                    f'{html.escape(str(c["job"]))} 대비, ensemble mean)</div>'
                    f'<div class="v">{d:+.3f}</div></div>')
            jobs_js[r["job"]]["kpis"] += card
    append_delta_exclusion_cards(rows, jobs_js)

    # ranking context for the header line
    ok = [r for r in rows if r.get("status") == "ok"
          and (r.get("iptm_nb") is not None or r.get("ptm") is not None)]
    def _f(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return x if math.isfinite(x) else None

    def _cand_key(r):
        if _f(r.get("iptm_nb")) is not None:
            return (1, _f(r["iptm_nb"]), _f(r.get("ipsae")) or 0.0)
        return (0, _f(r.get("ptm")) or 0.0, _f(r.get("ipsae")) or 0.0)
    # B-19: 지표가 동일/부재일 때 파일 순서에 따라 최우선 후보가 바뀌지 않도록 이름으로 tie-break
    ok.sort(key=_cand_key, reverse=True)
    if ok:
        top = _cand_key(ok[0])
        same = [r for r in ok if _cand_key(r) == top]
        best = min(same, key=lambda r: r["job"]) if len(same) > 1 else ok[0]
    else:
        best = None

    active_cols = select_active_columns(rows)
    target_only_batch = is_target_only_batch(rows)
    cols_js = [{"key": k, "label": lbl, "type": t, "nd": nd, "higherBetter": hb}
               for k, lbl, t, nd, hb in active_cols]

    def _js_json(obj):
        return json.dumps(obj).replace("<", "\\u003c")

    sort_key_js = "iptm_nb" if any(c["key"] == "iptm_nb" for c in cols_js) else "ptm"
    # M-17: 데이터에 들어 있는 치환자(__ROWS__ 등)와 충돌하지 않도록 한 번에 치환한다
    js = (JS.replace("__JOBS__", "\x00JOBS\x00")
            .replace("__ROWS__", "\x00ROWS\x00")
            .replace("__COLS__", "\x00COLS\x00")
            .replace("__SORTKEY__", "\x00SORTKEY\x00"))
    js = (js.replace("\x00JOBS\x00", _js_json(jobs_js))
            .replace("\x00ROWS\x00", _js_json(rows))
            .replace("\x00COLS\x00", _js_json(cols_js))
            .replace("\x00SORTKEY\x00", json.dumps(sort_key_js)))

    failed = [r["job"] for r in rows if r.get("status") != "ok"]
    warn = ""
    if failed:
        warn = ("<div class='warnbox'><b>미완료/실패 job</b>: "
                + ", ".join(html.escape(str(f)) for f in failed) +
                "<br>터미널에서 <span class='mono'>./run.sh --batch ... --jobs " +
                html.escape(",".join(str(f) for f in failed)) +
                "</span> 로 재실행할 수 있습니다.</div>")

    page = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boltz-2 batch report - {html.escape(str(manifest['batch']))}</title>
<link rel="stylesheet" href="molstar.css">
<style>{CSS}{BATCH_CSS}</style>
</head><body><div class="wrap">
<h1>Boltz-2 배치 스크리닝 리포트</h1>
<p class="sub">배치: <b>{html.escape(str(manifest['batch']))}</b> · job {len(rows)}개 · 생성 {html.escape(str(manifest.get('created', '')))}
{('· 기본 정렬 상단 미리보기: <b>' + html.escape(str(best['job'])) + '</b> ('
   + ('ipTM ' + _num(best['iptm_nb']) + ', ipSAE ' + _num(best.get('ipsae'))
      if best.get('iptm_nb') is not None else 'pTM ' + _num(best.get('ptm')))
   + ')') if best else ''}
<br>입력 표: <span class="mono">{html.escape(str(manifest.get('batch_file', '')))}</span> ·
표 정렬: 컬럼 헤더 클릭 · 상세 보기: 행 클릭 · <a href="summary.tsv">summary.tsv</a></p>
{warn}

<div id="table-view">
<div class="toolbar">
<label for="batch-search" class="note">검색</label>
<input id="batch-search" type="text" aria-label="job 이름 또는 메모 검색" placeholder="job 이름 / 메모 검색..." oninput="setFilter(this.value)">
<span class="note" id="count"></span>
<span class="note">기본 정렬: {('pTM 내림차순 (target-only 배치)' if target_only_batch else 'ipTM(나노바디|항원) 내림차순')} · 값 없는 항목은 항상 아래로</span>
</div>
<div class="card" style="padding:0">
<div class="tw" style="max-height:70vh;overflow:auto">
<table><thead id="thead"></thead><tbody id="tbody"></tbody></table>
</div></div>
<p class="note">{("target-only(나노바디 없음) 배치입니다. pTM 과 체인별 pLDDT 를 중심으로 보세요. "
  "ipTM/ipSAE/CDR/인터페이스 지표는 나노바디가 있는 입력에서만 계산됩니다."
  if target_only_batch else
  "기본 정렬 상단 미리보기는 항원/primary scope 그룹을 가로지르는 전역 최우수 판정이 아닙니다. "
  "ipTM(nb|ag) = 항원 좌표계에 정렬했을 때 나노바디의 배치 신뢰도 (핵심 지표). "
  "ipSAE ≥ 0.6 / ipTM ≥ 0.8 / 인터페이스 PAE &lt; 5 Å 이면 신뢰도가 높은 편입니다. "
  "DockQ는 참조 구조가 있는 job에서만 계산됩니다. 배지 색: 초록=양호, 주황=중간, 빨강=낮음.")}</p>
</div>

<div id="detail">
<button class="btn backbtn" onclick="backToTable()">← 표로 돌아가기</button>
<h2 style="margin-top:0" id="detail-title"></h2>
<div class="card" id="viewer-card">
<div id="viewer-missing" class="note" style="display:none;padding:14px 4px">
이 job은 3D 구조가 포함되지 않았습니다 (예측 미완료이거나 <span class="mono">--no-embed</span> 로 생성).
개별 리포트(<span class="mono">&lt;job&gt;/report/index.html</span>)를 열면 3D 를 볼 수 있습니다.</div>
<div id="molstar-viewer" style="width:100%;height:520px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff"></div>
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
<button class="btn" data-theme="chain-id">체인별 색상</button>
<button class="btn" data-theme="plddt-confidence">pLDDT 색상</button>
<button class="btn" data-theme="uncertainty">Uncertainty 색상</button>
<button class="btn" onclick="showOverlay()">참조 구조 겹쳐보기</button>
<button class="btn" onclick="RB.resetCamera('main')">카메라 리셋</button>
<a class="btn" id="detail-report-link" target="_blank" style="text-decoration:none;color:inherit">전체 리포트 새 탭으로 열기</a>
</div>
<div class="legend">
<span><span class="dot" style="background:#0053d6"></span>pLDDT &gt;90</span>
<span><span class="dot" style="background:#65cbf3"></span>70-90</span>
<span><span class="dot" style="background:#ffdb13"></span>50-70</span>
<span><span class="dot" style="background:#ff7d45"></span>&lt;50</span>
</div>
<div id="diag" class="diag">Mol* viewer 로딩...</div>
</div>
<div class="card"><h3 style="margin-top:0">핵심 지표 (선택된 모델 기준)</h3>
<div class="kpis" id="detail-kpis"></div></div>
<div class="card"><h3 style="margin-top:0">분석 그림</h3>
<div class="grid" id="detail-figs"></div></div>
<div class="card" id="detail-tables"></div>
<button class="btn backbtn" onclick="backToTable()">← 표로 돌아가기</button>
</div>

<details><summary>지표 해석 가이드 펼치기</summary>{INTERPRET_GUIDE}</details>
<p class="sub">개별 모델의 모든 지표/그림/인터페이스 잔기 목록은 각 job의 전체 리포트
(<span class="mono">&lt;job&gt;/report/index.html</span>)에서 볼 수 있습니다.</p>
</div>
<script src="molstar.js"></script>
<script src="report.js"></script>
<script>{js}</script>
</body></html>
"""
    (outdir / "index.html").write_text(page, encoding="utf-8")
    size_mb = (outdir / "index.html").stat().st_size / 1e6
    if size_mb > 20 and not args.no_embed:
        print(f"[batch-report] HTML {size_mb:.0f}MB - job 수가 많으면 --no-embed 로 줄일 수 있습니다")
    write_summary_tsv(outdir / "summary.tsv", rows, active_cols)
    print(f"[batch-report] wrote {outdir / 'index.html'} ({len(rows)} jobs)")
    print(f"[batch-report] table: {outdir / 'summary.tsv'}")
    if broken:
        print("[batch-report] incomplete batch report inputs; fix/re-run failed jobs before handoff")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
