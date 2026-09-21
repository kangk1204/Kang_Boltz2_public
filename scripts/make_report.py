#!/usr/bin/env python
"""Build a self-contained HTML report (Korean UI) with Mol* 3D viewer from results.json."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import shutil
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
from runtime_state import validate_prediction_outputs  # noqa: E402

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1d21;--muted:#666;--line:#e3e6ea;
--ok:#1b7f3b;--warn:#b06b00;--bad:#b3261e;--accent:#1f5aa8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR","Malgun Gothic",sans-serif;
 line-height:1.55;}
.wrap{max-width:1180px;margin:0 auto;padding:24px 20px 80px;}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:19px;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:16px;margin:20px 0 8px}
.sub{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0;
 box-shadow:0 1px 2px rgba(0,0,0,.03)}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600;color:#fff}
.b-ok{background:var(--ok)}.b-warn{background:var(--warn)}.b-bad{background:var(--bad)}.b-grey{background:#7a7f85}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:right;white-space:nowrap}
th{background:#f0f2f5;font-weight:600;text-align:right;position:sticky;top:0}
td:first-child,th:first-child,td.l,th.l{text-align:left}
tr.best{background:#fffbe6}
tr.best td{font-weight:600}
.tw{overflow-x:auto}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
img.fig{width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
#viewer-host{position:relative;width:100%;height:560px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff}
.model-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
.tab{cursor:pointer;border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 12px;font-size:13px}
.tab:focus-visible{outline:3px solid #9ec5ff;outline-offset:2px}
.tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.tab .s{display:block;font-size:11px;opacity:.85}
.btn{cursor:pointer;border:1px solid var(--line);background:#fff;border-radius:8px;padding:5px 10px;font-size:12px}
.btn:hover{background:#eef2f7}
.kv{font-size:13px}
.kv b{display:inline-block;min-width:190px;color:#333}
.note{font-size:12.5px;color:var(--muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
details{border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:10px 0;background:#fff}
summary{cursor:pointer;font-weight:600}
.diag{font-size:11px;color:#888;margin-top:6px}
.warnbox{background:#fff8e1;border:1px solid #ffe082;border-radius:8px;padding:10px 12px;font-size:13px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#444;margin-top:8px}
.legend span{display:inline-flex;align-items:center;gap:5px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
"""

JS_TEMPLATE = r"""
const STRUCTS = __STRUCTS__;
const META = __META__;
let CURRENT = null;
let REQUESTED = null;
let LOADED = null;
let REQUEST_ID = 0;
let ACTIVE_LOADS = 0;
let LAST_LOAD = Promise.resolve();
function setStatus(msg) {
  const el = document.getElementById('viewer-info');
  if (el) el.textContent = msg;
}
function setActiveTab(key) {
  document.querySelectorAll('.tab').forEach(t => {
    const active = t.dataset.key === key;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', active ? 'true' : 'false');
    t.tabIndex = active ? 0 : -1;
  });
}
async function showModel(key) {
  if (!STRUCTS[key]) {
    RB.diag('showModel missing structure for ' + key, true);
    setStatus('구조 파일을 찾을 수 없습니다: ' + key);
    return;
  }
  const requestId = ++REQUEST_ID;
  REQUESTED = key;
  setActiveTab(key);
  const meta = META[key] || { label: key, sub: '' };
  setStatus(meta.label + (meta.sub ? ' - ' + meta.sub : ''));
  if (LOADED === key && ACTIVE_LOADS === 0) {
    CURRENT = key;
    return;
  }
  try {
    // Loading clears the viewer; the previous structure is no longer cached.
    LOADED = null;
    ACTIVE_LOADS += 1;
    const load = RB.load('main', STRUCTS[key], 'plddt-confidence');
    LAST_LOAD = load.catch(() => {});
    await load;
    if (requestId !== REQUEST_ID) return;
    LOADED = key;
    CURRENT = key;
  } catch (e) {
    RB.diag('showModel failed: ' + e.message, true);
  } finally {
    ACTIVE_LOADS = Math.max(0, ACTIVE_LOADS - 1);
  }
}
async function showOverlay() {
  const modelKey = REQUESTED || CURRENT;
  if (!modelKey) return;
  const key = 'overlay_' + modelKey;
  if (!STRUCTS[key]) {
    setStatus('이 모델에는 참조 구조 겹침이 없습니다 (--reference 미지정)');
    return;
  }
  // Preserve click order: a later model request must supersede this pending overlay.
  const requestId = ++REQUEST_ID;
  try {
    await LAST_LOAD;
  } catch (e) {
    RB.diag('showOverlay wait failed: ' + e.message, true);
  }
  if (requestId !== REQUEST_ID) return;
  if (LOADED === key && ACTIVE_LOADS === 0) return;
  try {
    LOADED = null;
    ACTIVE_LOADS += 1;
    const load = RB.load('main', STRUCTS[key], 'chain-id');
    LAST_LOAD = load.catch(() => {});
    await load;
    if (requestId !== REQUEST_ID) return;
    LOADED = key;
    CURRENT = modelKey;
    setActiveTab(CURRENT);
    const meta = META[key] || {};
    setStatus('예측(A/B) vs 참조(RA/RB) 겹침 - 항원 체인 기준 정렬'
              + (meta.sub ? ' (' + meta.sub + ')' : ''));
  } catch (e) {
    RB.diag('showOverlay failed: ' + e.message, true);
  } finally {
    ACTIVE_LOADS = Math.max(0, ACTIVE_LOADS - 1);
  }
}
async function setTheme(themeName) {
  const key = 'main';
  try {
    await LAST_LOAD;
    await RB.setTheme(key, themeName);
  } catch (e) {
    RB.diag('setTheme failed: ' + e.message, true);
  }
}
document.addEventListener('DOMContentLoaded', () => {
  const best = META && META.__best__ ? META.__best__ : Object.keys(STRUCTS)[0];
  const host = document.querySelector('.model-tabs');
  if (host) {
    host.addEventListener('click', e => {
      const t = e.target.closest('.tab');
      if (t && t.dataset.key) showModel(t.dataset.key);
    });
    host.addEventListener('keydown', e => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
      const tabs = Array.from(host.querySelectorAll('.tab'));
      if (!tabs.length) return;
      const current = Math.max(0, tabs.indexOf(document.activeElement));
      let next = current;
      if (e.key === 'ArrowLeft') next = (current - 1 + tabs.length) % tabs.length;
      if (e.key === 'ArrowRight') next = (current + 1) % tabs.length;
      if (e.key === 'Home') next = 0;
      if (e.key === 'End') next = tabs.length - 1;
      e.preventDefault();
      tabs[next].focus();
      showModel(tabs[next].dataset.key);
    });
  }
  showModel(best);
});
"""


def metrics_dockq_headline(d):
    """결과 dict 에서 표시용 DockQ 대표값(인터페이스 평균)을 꺼낸다."""
    if not d:
        return None
    return d.get("headline", d.get("best_dockq"))


def primary_scores(model: dict, results: dict | None = None) -> dict:
    """Return the shared headline interface score contract used by reports."""
    enriched = dict(model)
    if results:
        enriched.setdefault("antigen_chains", results.get("antigen_chains") or [])
        enriched.setdefault("has_nanobody", results.get("nanobody_chain") is not None)
        enriched.setdefault("target_only", results.get("nanobody_chain") is None)
    out = metrics.primary_interface_scores(enriched)
    if "source" not in out:
        out["source"] = ",".join(str(out.get(k, "")) for k in ("iptm_source", "ipsae_source")).strip(",")
    return out


def primary_source_label(primary: dict) -> str:
    """Human/TSV label for the source of the displayed primary interface scores."""
    return ",".join(
        str(primary.get(k, "")).strip()
        for k in ("iptm_source", "ipsae_source")
        if str(primary.get(k, "")).strip()
    ) or str(primary.get("source") or "-")


def has_analysis_provenance(results: dict) -> bool:
    """Fresh results store analysis provenance under settings; accept legacy top-level too."""
    settings = results.get("settings") or {}
    return bool(settings.get("analysis_provenance") or results.get("analysis_provenance"))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _infer_local_run_dir(results_path: Path, explicit: Path | None = None) -> Path:
    if explicit:
        return explicit.resolve(strict=False)
    parent = results_path.resolve(strict=False).parent
    if parent.name in {"analysis", "report"}:
        return parent.parent
    return parent


def _rebase_one(path_value: str | None, local_run_dir: Path) -> str | None:
    if not path_value:
        return path_value
    p = Path(path_value)
    if p.exists() and _is_within(p, local_run_dir):
        return str(p)
    name = p.name
    matches = [m for m in local_run_dir.rglob(name) if m.is_file()]
    if len(matches) == 1:
        return str(matches[0])
    return path_value


def _rebase_confined_file(path_value: str | None, local_run_dir: Path) -> str | None:
    if not path_value:
        return path_value
    p = Path(path_value)
    if p.is_file() and _is_within(p, local_run_dir):
        return str(p)
    matches = [m for m in local_run_dir.rglob(p.name) if m.is_file()]
    if len(matches) == 1:
        return str(matches[0])
    return None


def _rebase_dir(path_value: str | None, local_run_dir: Path) -> str | None:
    if not path_value:
        return path_value
    p = Path(path_value)
    if p.is_dir() and _is_within(p, local_run_dir):
        return str(p)
    matches = [m for m in local_run_dir.rglob(p.name) if m.is_dir()]
    if len(matches) == 1:
        return str(matches[0])
    return path_value


def rebase_result_paths(results: dict, results_path: Path, run_dir: Path | None = None) -> Path:
    """Prefer artifacts located under the copied/local run directory."""
    local_run_dir = _infer_local_run_dir(results_path, run_dir)
    results["run_dir"] = str(local_run_dir)
    pred_dir = results.get("predictions_dir")
    if pred_dir:
        rebased = _rebase_dir(pred_dir, local_run_dir)
        if rebased and Path(rebased).is_dir():
            results["predictions_dir"] = rebased
    if isinstance(results.get("figures"), dict):
        results["figures"] = {
            key: rebased
            for key, value in results["figures"].items()
            if (rebased := _rebase_confined_file(value, local_run_dir))
        }
    for m in results.get("models") or []:
        for key in ("cif", "confidence_json", "pae_npz", "plddt_npz"):
            if m.get(key):
                m[key] = _rebase_one(m.get(key), local_run_dir)
        if isinstance(m.get("figures"), dict):
            m["figures"] = {
                key: rebased
                for key, value in m["figures"].items()
                if (rebased := _rebase_confined_file(value, local_run_dir))
            }
        overlay = m.get("overlay") or {}
        if overlay.get("path"):
            rebased = _rebase_confined_file(overlay.get("path"), local_run_dir)
            if rebased:
                overlay["path"] = rebased
            else:
                overlay.pop("path", None)
    return local_run_dir


def validate_results(results: dict, local_run_dir: Path | None = None) -> None:
    """Refuse report generation from incomplete analysis artifacts."""
    models = results.get("models") or []
    if not models:
        raise ValueError("results.json has no models")
    run_params = (results.get("settings") or {}).get("run_params") or {}
    expected_samples = run_params.get("samples")
    if expected_samples is None:
        expected_samples = len(models)
    pred_dir = Path(results.get("predictions_dir") or "")
    if local_run_dir and (not pred_dir.exists() or not _is_within(pred_dir, local_run_dir)):
        raise ValueError(f"prediction directory is not local to this run: {pred_dir}")
    checked = validate_prediction_outputs(pred_dir, int(expected_samples))
    checked_indexes = [r["index"] for r in checked]
    result_indexes = sorted(int(m.get("index")) for m in models)
    if result_indexes != checked_indexes:
        raise ValueError(f"results model indexes {result_indexes} do not match predictions {checked_indexes}")
    for m in models:
        idx = m.get("index", "?")
        required = (
            ("cif", "CIF"),
            ("confidence_json", "confidence JSON"),
            ("pae_npz", "PAE npz"),
        )
        for key, label in required:
            value = m.get(key)
            if not value or not Path(value).is_file():
                raise ValueError(f"model_{idx}: missing {label}: {value or '-'}")
            if local_run_dir and not _is_within(Path(value), local_run_dir):
                raise ValueError(f"model_{idx}: {label} is outside this run copy: {value}")


def esc(v):
    return html.escape(str(v), quote=True)


def b64(path):
    p = Path(path)
    if not path or not p.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def safe_report_path(path_value, allowed_roots):
    """Return a readable file only when it stays under one of the report/run roots."""
    if not path_value:
        return None
    try:
        p = Path(path_value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not p.is_file():
        return None
    roots = [Path(r).resolve(strict=False) for r in allowed_roots if r]
    return p if any(_is_within(p, root) for root in roots) else None


def figure_html(path_value, caption, allowed_roots):
    p = safe_report_path(path_value, allowed_roots)
    if not p:
        return ""
    return (f"<figure style='margin:0'><img class='fig' src='{b64(p)}' alt='{esc(caption)}'>"
            f"<figcaption class='note'>{esc(caption)}</figcaption></figure>")


def fmt(v, nd=3):
    """표시용 공통 포맷터 (R6-01).

    - None / 비유한 수치 -> "-"
    - 수치로 해석되는 값(정수/실수/숫자 문자열) -> 지정 자릿수로 포맷
    - 그 외 텍스트 -> HTML escape 해서 텍스트로만 표시 (raw 태그 삽입 방지)
    """
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "1" if v else "0"
    try:
        x = float(v)          # 숫자(문자열 포함)로 해석되면 수치 계약 적용
    except (TypeError, ValueError):
        return esc(str(v))    # 그 외 텍스트만 escape 해서 표시
    return "-" if not math.isfinite(x) else f"{x:.{int(nd)}f}"


def safe_float(v):
    """유한한 실수면 float, 아니면 None (문자열/NaN/inf 방어)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fmt_num(v, nd=3):
    """HTML 셀용 수치 포맷: 유한 수치만 표시, 그 외 '-'. (문자열 raw 삽입 방지)"""
    x = safe_float(v)
    if x is None:
        return "-"
    return f"{x:.{int(nd)}f}" if nd else str(int(x))


def fmt_sci(v):
    x = safe_float(v)
    if x is None:
        return "-" if v is None else esc(str(v))
    return f"{x:.2e}"


def badge_for(value, thresholds, labels, invert=False):
    """thresholds: list of (cut, css). Returns (css, label)."""
    value = safe_float(value)
    if value is None:
        return "b-grey", "N/A"
    for cut, css, lab in thresholds:
        if (value >= cut) if not invert else (value <= cut):
            return css, lab
    return "b-grey", "N/A"


def iptm_badge(v):
    # H-03: 0.6 미만도 '낮음'으로 표시한다 (기존에는 회색 N/A 로 떨어져 실패가 '데이터 없음'으로 보였다)
    return badge_for(v, [(0.8, "b-ok", "높음"), (0.6, "b-warn", "중간"),
                         (float("-inf"), "b-bad", "낮음")], None)


def dockq_badge(v):
    x = safe_float(v)
    if x is None:
        return "b-grey", "N/A"
    return badge_for(x, [(0.8, "b-ok", "High"), (0.49, "b-warn", "Medium"),
                         (0.23, "b-warn", "Acceptable"), (0.0, "b-bad", "Incorrect")], None)


def plddt_badge(v):
    if v is None:
        return "b-grey", "N/A"
    return badge_for(v, [(90, "b-ok", "매우 높음"), (70, "b-warn", "양호"), (50, "b-warn", "낮음"), (0, "b-bad", "매우 낮음")], None)


def pdockq2_badge(v):
    """pDockQ2 는 0.005~1.315 범위로 보정된 점수다 (문헌 기준 0.8/0.5)."""
    return badge_for(v, [(0.8, "b-ok", "높음"), (0.5, "b-warn", "중간"),
                         (float("-inf"), "b-bad", "낮음")], None)


def pdockq_badge(v):
    """pDockQ 문헌 기준 (0.5 높음, 0.23 중간)."""
    return badge_for(v, [(0.5, "b-ok", "높음"), (0.23, "b-warn", "중간"),
                         (float("-inf"), "b-bad", "낮음")], None)


def lis_badge(v):
    """LIS 경험 기준 (~0.203 이상이면 결합 가능성)."""
    return badge_for(v, [(0.35, "b-ok", "높음"), (0.2, "b-warn", "중간"),
                         (float("-inf"), "b-bad", "낮음")], None)


def ipsae_badge(v):
    x = safe_float(v)
    if x is None:
        return "b-grey", "N/A"
    if x >= 0.6:
        return "b-ok", "양호"
    if x >= 0.4:
        return "b-warn", "중간"
    return "b-bad", "낮음"


def verdict_html(results):
    """Auto-generated Korean verdict for the best model."""
    m = results["models"][0]
    if not results.get("nanobody_chain"):
        # target-only: 나노바디 지표 대신 전체/체인 신뢰도 중심 판정
        plddt_chain = m["plddt"]["per_chain"]
        parts = [(f'<span class="badge b-grey">타겟 단독/복합체</span> '
                  f'pTM = <b>{fmt(m["boltz"].get("ptm"))}</b>'),
                 f'전체 평균 pLDDT = <b>{fmt(m["plddt"].get("mean"), 1)}</b>']
        for ch, st in (plddt_chain or {}).items():
            css, lab = plddt_badge(st.get("mean"))
            parts.append(f'체인 {esc(ch)}: {fmt(st.get("mean"), 1)} '
                         f'<span class="badge {css}">{lab}</span>')
        parts.append("나노바디 관련 지표(ipTM/ipSAE/CDR/인터페이스)는 이 입력에 해당하지 않습니다.")
        return "<br>".join(parts)
    primary = primary_scores(m, results)
    nb_ag = primary.get("iptm")
    ipsae = primary.get("ipsae")
    iface = m["interface_8A"] or {}
    pae_if = iface.get("pae_mean_ab")
    dq = metrics_dockq_headline(m.get("dockq"))
    cdr = m.get("cdr_mean_plddt") or {}

    lines = []
    css, lab = iptm_badge(nb_ag)
    lines.append(f'<span class="badge {css}">ipTM {lab}</span> '
                 f'{esc(primary.get("iptm_label", "ipTM"))} = <b>{fmt(nb_ag)}</b>')
    css, lab = ipsae_badge(ipsae)
    lines.append(f'<span class="badge {css}">ipSAE {lab}</span> '
                 f'{esc(primary.get("ipsae_label", "ipSAE"))} = <b>{fmt(ipsae)}</b>')
    pae_if_f = safe_float(pae_if)
    if pae_if_f is not None:
        css = "b-ok" if pae_if_f < 5 else ("b-warn" if pae_if_f < 10 else "b-bad")
        lab = "신뢰" if pae_if_f < 5 else ("보통" if pae_if_f < 10 else "불확실")
        lines.append(f'<span class="badge {css}">인터페이스 PAE {lab}</span> 평균 {fmt(pae_if, 2)} Å')
    if dq is not None:
        css, lab = dockq_badge(dq)
        lines.append(f'<span class="badge {css}">DockQ {lab}</span> {fmt_num(dq)} '
                     f'({esc((m.get("dockq") or {}).get("best_mapping_str", ""))})')
    if cdr.get("CDR3") is not None:
        css, lab = plddt_badge(cdr["CDR3"])
        lines.append(f'<span class="badge {css}">CDR3 pLDDT {lab}</span> {fmt(cdr["CDR3"], 1)}')
    return "<br>".join(lines)


def chain_role_label(results, ch):
    for c in results["chains"]:
        if c["id"] == ch:
            return {"antigen": "항원", "nanobody": "나노바디"}.get(c["role"], c["role"])
    return ch


def build_summary_table(results):
    target_only = not results.get("nanobody_chain")
    n_chains = len(results["chains"])
    multi_note = ("" if n_chains <= 2 else
                  f'<br><b>체인 {n_chains}개 입력</b>: 대표 ipTM/ipSAE는 항원 체인 union 기준입니다. '
                  '체인별 쌍 지표는 모델별 상세의 "체인 쌍 지표" 표를 보세요.')
    rows = []
    for e in results["models"]:
        primary = primary_scores(e, results)
        nb_ag = primary.get("iptm")
        ag_nb = e["boltz_pair_iptm"]["antigen_in_nanobody_frame"]
        mx = e["ipsae"].get("max") or {}
        primary_ipsae = primary.get("ipsae")
        plddt_chain = e["plddt"]["per_chain"].get(results["nanobody_chain"], {})
        iface = e["interface_8A"] or {}
        dq = metrics_dockq_headline(e.get("dockq"))
        is_best = e["index"] == results["best_model_index"]
        css_i, lab_i = iptm_badge(nb_ag)
        css_s, lab_s = ipsae_badge(primary_ipsae)
        css_q, lab_q = dockq_badge(dq)
        css_p, lab_p = plddt_badge(plddt_chain.get("mean"))
        if target_only:
            # 나노바디 전용 열은 숨기고 복합체/체인 지표만 표시 (RR09)
            plddt_chain = (e["plddt"]["per_chain"] or {})
            chain_cells = "".join(
                f"<td>{fmt((plddt_chain.get(c['id']) or {}).get('mean'), 1)}</td>"
                for c in results["chains"])
            rows.append(f"""<tr class="{'best' if is_best else ''}">
<td class="l">model_{fmt_num(e.get('index'), 0)}{' ★' if is_best else ''}</td>
<td>{fmt(e['boltz'].get('ptm'))}</td>
<td>{fmt(e['plddt'].get('mean'), 1)}</td>
{chain_cells}
</tr>""")
            continue
        rows.append(f"""<tr class="{'best' if is_best else ''}">
<td class="l">model_{fmt_num(e.get('index'), 0)}{' ★' if is_best else ''}</td>
<td>{fmt(nb_ag)} <span class="badge {css_i}">{lab_i}</span></td>
<td>{fmt(ag_nb)}</td>
<td>{fmt(e['boltz']['ptm'])}</td>
<td>{fmt(primary_ipsae)} <span class="badge {css_s}">{lab_s}</span></td>
<td>{fmt(mx.get('ipsae_d0chn'))}</td>
<td>{fmt(mx.get('ipsae_d0dom'))}</td>
<td>{fmt(mx.get('iptm_d0chn'))}</td>
<td>{fmt(mx.get('pdockq'))}</td>
<td>{fmt(mx.get('pdockq2'))}</td>
<td>{fmt(mx.get('lis'))}</td>
<td>{fmt(plddt_chain.get('mean'), 1)} <span class="badge {css_p}">{lab_p}</span></td>
<td>{fmt(iface.get('pae_mean_ab'), 2)}</td>
<td>{fmt_num(dq)}{(' <span class="badge ' + css_q + '">' + lab_q + '</span>') if dq is not None else ''}</td>
</tr>""")
    if target_only:
        chain_heads = "".join(f"<th>체인 {esc(c['id'])}<br>pLDDT</th>" for c in results["chains"])
        return f"""<div class="tw"><table><thead><tr>
<th class="l">모델</th><th>pTM</th><th>전체 평균<br>pLDDT</th>{chain_heads}
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">target-only(나노바디 없음) 입력입니다. ipTM/ipSAE/CDR/인터페이스 지표는
나노바디가 있는 입력에서만 계산됩니다.</p>"""
    return f"""<div class="tw"><table>
<thead><tr>
<th class="l">모델</th><th>대표 ipTM<br>(nb|ag)</th><th>pair ipTM<br>(ag|nb)</th><th>pTM</th>
<th>대표 ipSAE</th><th>pair ipSAE<br>d0chn</th><th>pair ipSAE<br>d0dom</th><th>pair ipTM<br>d0chn</th>
<th>pDockQ</th><th>pDockQ2</th><th>LIS</th><th>나노바디<br>pLDDT</th>
<th>인터페이스<br>PAE(Å)</th><th>DockQ</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">★ = 기본 선택 모델 (분석의 primary interface score 기준 정렬).
ipTM/ipSAE의 "(X|Y)" 표기는 <b>Y를 기준 좌표계(frame)로 정렬했을 때 X가 얼마나 잘 배치되었는지</b>를
뜻합니다. 예를 들어 <b>ipTM(나노바디|항원)</b>은 항원 좌표계에서 본 나노바디 배치의 신뢰도이고,
<b>ipTM(항원|나노바디)</b>는 그 반대 방향입니다.
DockQ는 참조 구조가 있을 때만 계산됩니다.{multi_note}</p>"""


def build_chain_pair_table(results, entry):
    """모든 체인 쌍의 ipTM / 인터페이스 PAE / 접촉 수 (다중 체인 복합체용)."""
    rows = entry.get("chain_pairs") or []
    if not rows:
        return ""
    body = ""
    for r in rows:
        nb_label = lambda c: chain_role_label(results, c)  # noqa: E731
        body += (f"<tr><td class='l'>{esc(r['chain_a'])} ({esc(nb_label(r['chain_a']))}) - "
                 f"{esc(r['chain_b'])} ({esc(nb_label(r['chain_b']))})</td>"
                 f"<td>{fmt(r.get('iptm_boltz_b_scored_in_a_frame'))}</td>"
                 f"<td>{fmt(r.get('iptm_boltz_a_scored_in_b_frame'))}</td>"
                 f"<td>{fmt(r.get('iface_pae_a_frame_b_scored'), 2)}</td>"
                 f"<td>{fmt(r.get('iface_pae_b_frame_a_scored'), 2)}</td>"
                 f"<td>{fmt_num(r.get('n_contacts'), 0)}</td></tr>")
    return f"""<h4>체인 쌍 지표 (모든 조합)</h4>
<div class="tw"><table><thead><tr><th class="l">체인 쌍</th>
<th>ipTM<br>(뒤 체인 | 앞 체인 기준)</th><th>ipTM<br>(앞 체인 | 뒤 체인 기준)</th>
<th>PAE(Å)<br>앞 기준/뒤 잔기</th><th>PAE(Å)<br>뒤 기준/앞 잔기</th><th>접촉 수</th>
</tr></thead><tbody>{body}</tbody></table></div>
<p class="note">ipTM 방향 표기: "X | Y 기준" = Y를 기준 좌표계로 정렬했을 때 X의 배치 신뢰도.
인터페이스 PAE는 접촉(8 Å) 잔기 쌍 평균입니다.</p>"""


def _pct(v):
    x = safe_float(v)
    return "-" if x is None else f"{100 * x:.0f}%"


def build_chain_table(results, entry):
    rows = []
    for ch, st in entry["plddt"]["per_chain"].items():
        rows.append(f"""<tr><td class="l">{esc(ch)} ({esc(chain_role_label(results, ch))})</td>
<td>{fmt_num(st.get('n'), 0)}</td><td>{fmt(st.get('mean'), 1)}</td>
<td>{fmt(st.get('median'), 1)}</td>
<td>{fmt(st.get('min'), 1)}</td><td>{fmt(st.get('max'), 1)}</td>
<td>{_pct(st.get('frac_gt90'))}</td><td>{_pct(st.get('frac_lt50'))}</td></tr>""")
    return f"""<div class="tw"><table><thead><tr>
<th class="l">체인</th><th>잔기 수</th><th>평균 pLDDT</th><th>중앙값</th><th>최소</th><th>최대</th>
<th>&gt;90 비율</th><th>&lt;50 비율</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"""


def build_interface_table(results, entry, which):
    iface = entry["interface_8A"] or {}
    rows = iface.get("residues_a" if which == "antigen" else "residues_b", [])
    if which == "antigen":
        antigen_chains = results.get("antigen_chains") or [results["antigen_chain"]]
        chain = ",".join(antigen_chains)
        title = "항원 union" if len(antigen_chains) > 1 else "항원"
    else:
        chain = results["nanobody_chain"]
        title = "나노바디"
    if not rows:
        return f"<p class='note'>{title} 인터페이스 잔기가 없습니다 (접촉 8 Å 기준).</p>"
    body = "".join(
        f"<tr><td class='l'>{esc(r.get('chain'))}{fmt_num(r.get('resnum'), 0)}{esc(r.get('icode') or '')}"
        f"{esc(r.get('resname'))}</td>"
        f"<td>{fmt_num(r.get('n_contacts'), 0)}</td><td>{fmt(r.get('plddt'), 1)}</td></tr>"
        for r in rows
    )
    return (f"<h4>{title} 인터페이스 잔기 ({esc(chain)}) - 잔기 {len(rows)}개, "
            f"총 컨택트 {fmt_num(iface.get('n_contacts'), 0)}쌍 (CB-CB ≤ 8 Å)</h4>"
            f"<div class='tw'><table><thead><tr><th class='l'>잔기</th>"
            f"<th>접촉 수</th><th>pLDDT</th></tr></thead><tbody>{body}</tbody></table></div>")


def build_cdr_table(results, entry):
    cdr = results.get("cdr") or {}
    if not cdr.get("available"):
        return ("<p class='note'>CDR 자동 주석을 사용할 수 없습니다 (ANARCI/hmmscan 미설치). "
                "setup.sh 또는 README의 'ANARCI(선택)' 항목을 참고하세요.</p>")
    vals = entry.get("cdr_mean_plddt") or {}
    rows = []
    for name in ("CDR1", "CDR2", "CDR3"):
        seq = (cdr.get("cdr_sequences") or {}).get(name, "")
        v = vals.get(name)
        css, lab = plddt_badge(v)
        rng = (cdr.get("token_ranges") or {}).get(name)
        rng_txt = (f"{fmt_num(rng[0], 0)}-{fmt_num(rng[1], 0)}" if rng else "-")
        rows.append(f"<tr><td class='l'>{name}</td><td class='l mono'>{html.escape(seq)}</td>"
                    f"<td>{len(seq)}</td><td>{fmt(v, 1)} <span class='badge {css}'>{lab}</span></td>"
                    f"<td class='l mono'>{rng_txt}</td></tr>")
    hit = cdr.get("hit") or {}
    return (f"<h4>CDR 영역 (IMGT 번호 기준, ANARCI)</h4><div class='tw'><table><thead><tr>"
            f"<th class='l'>CDR</th><th class='l'>서열</th><th>길이</th><th>평균 pLDDT</th>"
            f"<th class='l'>토큰 범위</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<p class='note'>ANARCI 최적 히트: {html.escape(str(hit.get('id') or '-'))}, "
            f"E-value {fmt_sci(hit.get('evalue'))}. CDR3 pLDDT가 낮으면 paratope loop의 형태가 "
            f"불확실하다는 뜻이므로, 해당 loop의 결합 기여 해석에 주의하세요.</p>")


def build_dockq_section(results, entry):
    dq = entry.get("dockq") or {}
    if not results["settings"].get("reference"):
        return ("<p class='note'>참조 구조(정답 complex)가 제공되지 않았거나 파일을 찾지 못해 DockQ를 계산하지 않았습니다. "
                "알려진 항원-나노바디 복합체 결정구조가 있다면 "
                "<span class='mono'>./run.sh --reference ref.cif --name &lt;run&gt;</span> 또는 "
                "<span class='mono'>--report-only</span>로 다시 실행하면 DockQ/CAPRI 등급이 추가됩니다.</p>")
    if "error" in dq:
        return f"<p class='note'>DockQ 계산 실패: {html.escape(str(dq['error']))}</p>"
    ifaces = dq.get("interfaces") or {}
    rows = ""
    for key, v in ifaces.items():
        css_i, lab_i = dockq_badge(v.get("DockQ"))
        capri = v.get("CAPRI") or v.get("capri") or lab_i
        rows += (f"<tr><td class='l'>{esc(key)}</td><td>{fmt_num(v.get('DockQ'))}</td>"
                 f"<td>{fmt_num(v.get('F1'))}</td><td>{fmt_num(v.get('iRMSD'), 2)}</td>"
                 f"<td>{fmt_num(v.get('LRMSD'), 2)}</td><td>{fmt_num(v.get('fnat'))}</td>"
                 f"<td>{fmt_num(v.get('fnonnat'))}</td>"
                 f"<td>{fmt_num(v.get('nat_correct'), 0)}/{fmt_num(v.get('nat_total'), 0)}</td>"
                 f"<td>{fmt_num(v.get('clashes'), 0)}</td>"
                 f"<td><span class='badge {css_i}'>{esc(capri)}</span></td></tr>")
    css, lab = dockq_badge(metrics_dockq_headline(dq))
    return f"""<p>참조 구조: <span class='mono'>{html.escape(str(results['settings']['reference']))}</span>,
매핑 <span class='mono'>{esc(dq.get('best_mapping_str', ''))}</span>,
{f"<b>서열 불일치 {fmt_num(dq['auto_allowed_mismatches'], 0)}개 허용(자동)</b>, " if dq.get('auto_allowed_mismatches') else ""}
DockQ(대표값, {esc(dq.get('headline_kind', '인터페이스 평균'))}) <b>{fmt_num(metrics_dockq_headline(dq))}</b>
<span class="badge {css}">CAPRI {lab}</span>
{f" · 인터페이스 {dq.get('n_interfaces')}개 합계 {fmt_num(dq.get('total_dockq'))}" if dq.get('n_interfaces', 0) > 1 else ""}</p>
<p class="note">DockQ v2의 합계(total)는 인터페이스별 점수의 합이므로 대표값으로 쓰지 않습니다.
아래 표의 인터페이스별 DockQ와 CAPRI 등급을 함께 보세요.</p>
<div class='tw'><table><thead><tr><th class='l'>인터페이스</th><th>DockQ</th><th>F1</th>
<th>iRMSD(Å)</th><th>LRMSD(Å)</th><th>f_nat</th><th>f_non-nat</th><th>정답/native 접촉</th>
<th>clashes</th><th>CAPRI</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class='note'>DockQ/CAPRI 등급: ≥0.80 High, 0.49-0.80 Medium, 0.23-0.49 Acceptable, &lt;0.23 Incorrect.
f_nat = native 접촉 중 재현된 비율, iRMSD/LRMSD는 인터페이스/전체 리간드 RMSD입니다.</p>"""


def build_model_section(results, entry, allowed_roots=None):
    idx = entry["index"]
    ipsae_fallback = bool((entry.get("ipsae") or {}).get("fallback"))
    best = idx == results["best_model_index"]
    mx = entry["ipsae"].get("max") or {}
    primary = primary_scores(entry, results)
    asym = entry["ipsae"].get("asym_nb_frame_ag") or {}
    iface = entry["interface_8A"] or {}
    figs = entry.get("figures") or {}
    trusted_roots = allowed_roots or [Path(results.get("run_dir") or ".")]
    img = (lambda p, cap: figure_html(p, cap, trusted_roots)) if figs else None

    target_only = not results.get("nanobody_chain")
    kpis = [] if target_only else [
        (primary.get("iptm_label") or "대표 ipTM", primary.get("iptm"), iptm_badge),
        ("ipTM (항원|나노바디)", entry["boltz_pair_iptm"]["antigen_in_nanobody_frame"], iptm_badge),
        ("pTM (전체)", entry["boltz"]["ptm"], iptm_badge),
        ((primary.get("ipsae_label") or "대표 ipSAE")
         + (" [내장 폴백]" if ipsae_fallback else ""), primary.get("ipsae"), ipsae_badge),
        ("ipSAE_d0chn (max)", mx.get("ipsae_d0chn"), ipsae_badge),
        ("ipSAE_d0dom (max)", mx.get("ipsae_d0dom"), ipsae_badge),
        ("ipTM_d0chn", mx.get("iptm_d0chn"), iptm_badge),
        ("pDockQ", mx.get("pdockq"), pdockq_badge),
        ("pDockQ2", mx.get("pdockq2"), pdockq2_badge),
        ("LIS", mx.get("lis"), lis_badge),
        ("인터페이스 평균 PAE (Å, 나노바디 잔기/항원 기준)", iface.get("pae_mean_ab"), None, 2),
        ("인터페이스 평균 pLDDT (나노바디)", iface.get("plddt_mean_b"), plddt_badge, 1),
        ("모델 신뢰도 (confidence_score)", entry["boltz"].get("confidence_score"), None),
    ]
    if target_only:
        kpis = [
            ("pTM (전체)", entry["boltz"].get("ptm"), iptm_badge),
            ("전체 평균 pLDDT", entry["plddt"].get("mean"), plddt_badge),
            ("체인 수", len(results["chains"]), None, 0),
            ("모델 신뢰도 (confidence_score)", entry["boltz"].get("confidence_score"), None),
        ]
    kpi_html = ""
    for item in kpis:
        name, val, bfn = item[0], item[1], item[2]
        nd = item[3] if len(item) > 3 else 3
        css, lab = (bfn(val) if bfn else ("b-grey", ""))
        badge = f' <span class="badge {css}">{lab}</span>' if bfn and val is not None else ""
        kpi_html += f"<tr><td class='l'>{name}</td><td>{fmt(val, nd)}{badge}</td></tr>"
    if asym:
        kpi_html += (f"<tr><td class='l'>ipSAE 방향성 (나노바디 기준→항원)</td>"
                     f"<td>{fmt(asym.get('ipsae'))}</td></tr>")
        kpi_html += (f"<tr><td class='l'>ipSAE 방향성 (항원 기준→나노바디)</td>"
                     f"<td>{fmt((entry['ipsae'].get('asym_ag_frame_nb') or {}).get('ipsae'))}</td></tr>")

    fig_block = ""
    if img:
        top = "".join(img(figs.get(k), cap) for k, cap in
                      (("pae", "PAE 히트맵 (전체)"), ("plddt", "잔기별 pLDDT")))
        bottom = "".join(img(figs.get(k), cap) for k, cap in
                         (("pae_iface", "인터페이스 PAE (나노바디/항원 블록)"),
                          ("ipsae_byres", "잔기별 ipSAE (나노바디가 aligned chain)")))
        fig_block = f'<div class="grid">{top}</div>'
        if bottom:
            fig_block += f'<div class="grid" style="margin-top:14px">{bottom}</div>'

    dockq_rows = ""
    _dq_head = metrics_dockq_headline(entry.get("dockq"))
    if _dq_head is not None:
        css, lab = dockq_badge(_dq_head)
        kind = (entry["dockq"] or {}).get("headline_kind", "")
        dockq_rows = (f"<tr><td class='l'>DockQ (vs 참조 구조, {esc(kind)})</td>"
                      f"<td>{fmt_num(_dq_head)} "
                      f"<span class='badge {css}'>{lab}</span></td></tr>")

    _dq_head = metrics_dockq_headline(entry.get('dockq'))
    dq_txt = ('/ DockQ ' + fmt_num(_dq_head) if _dq_head is not None else '')
    idx_txt = fmt_num(idx, 0)
    if target_only:
        summary_line = (f"model_{idx_txt} {'★ (선택된 모델)' if best else ''}"
                        f" - pTM {fmt(entry['boltz'].get('ptm'))} /"
                        f" 평균 pLDDT {fmt(entry['plddt'].get('mean'), 1)} {dq_txt}")
    else:
        summary_line = (
            f"model_{idx_txt} {'★ (선택된 모델)' if best else ''}"
            f" - 대표 ipTM {fmt(primary.get('iptm'))} /"
            f" 대표 ipSAE {fmt(primary.get('ipsae'))} / pDockQ2 {fmt(mx.get('pdockq2'))} {dq_txt}")
    return f"""
<details {'open' if best else ''}>
<summary>{summary_line}</summary>
<div class="grid" style="margin-top:12px">
 <div><h4>핵심 지표</h4><div class="tw"><table><tbody>{kpi_html}{dockq_rows}</tbody></table></div>
   <h4 style="margin-top:14px">체인별 pLDDT 통계</h4>{build_chain_table(results, entry)}
   {build_chain_pair_table(results, entry)}
 </div>
 <div><h4>CDR / 인터페이스</h4>{build_cdr_table(results, entry)}
   <p class="note">파일: <span class="mono">{html.escape(Path(entry['cif']).name) if entry.get('cif') else "-"}</span>,
   <span class="mono">{html.escape(Path(entry['pae_npz']).name) if entry.get('pae_npz') else "-"}</span>,
   <span class="mono">{html.escape(Path(entry['confidence_json']).name) if entry.get('confidence_json') else "-"}</span>
   {', <span class="mono">' + html.escape(Path(entry['ipsae']['pml']).name) + '</span> (PyMOL)' if entry['ipsae'].get('pml') else ''}
   </p>
 </div>
</div>
{fig_block}
{"" if target_only else
  '<h4 style="margin-top:16px">인터페이스 잔기 목록</h4>'
  '<div class="grid">' + build_interface_table(results, entry, 'antigen')
  + build_interface_table(results, entry, 'nanobody') + '</div>'}
<h4 style="margin-top:16px">DockQ 상세</h4>
{build_dockq_section(results, entry)}
</details>"""


INTERPRET_GUIDE = """
<h2>지표 해석 가이드</h2>
<div class="card">
<p class="note"><b>중요:</b> 아래 임계값은 탐색용 휴리스틱이며, 이 데이터셋/표적에 대해 보정된 판정 기준이
아닙니다. ipTM, ipSAE, pDockQ 계열 지표는 구조 모델의 상대 배치와 인터페이스 신뢰도를 요약할 뿐,
친화도(Kd), 결합 에너지, 실제 결합 여부의 합격/불합격 판정으로 해석하지 마세요.</p>
<h3>1) pLDDT (잔기별 국소 신뢰도, 0-100)</h3>
<ul>
<li><b>&gt;90</b>: 매우 높음 - 국소 골격 배치의 신뢰도가 높은 편</li>
<li><b>70-90</b>: 양호 - 전체 fold 해석에 유용. 세부 loop/측쇄 회전은 주의</li>
<li><b>50-70</b>: 낮음 - 국소 구조 불확실</li>
<li><b>&lt;50</b>: 매우 낮음 - 무질서(disordered) 또는 잘못된 모델 가능성</li>
</ul>
<p class="note">나노바디에서는 <b>CDR3(및 CDR1/2)의 pLDDT</b>가 특히 중요합니다. CDR3 pLDDT가 낮으면 paratope
형태가 불확실하므로 인터페이스 pose 해석의 신뢰도가 낮아집니다. 체인 평균 pLDDT(프레임워크 위주)가 높더라도 CDR3가
낮을 수 있으니 반드시 별도로 확인하세요.</p>

<h3>2) PAE (Predicted Aligned Error, Å)</h3>
<p>두 잔기(또는 도메인)의 <b>상대적 배치</b>에 대한 예측 오차입니다. 도메인 내부(pLDDT)가 아니라
<b>인터페이스 기하</b>를 평가할 때 사용합니다.</p>
<ul>
<li><b>&lt;5 Å</b>: 상대 배치 불확실성이 낮은 편</li>
<li><b>5-10 Å</b>: 보통 - 대략적 pose 중심으로 해석</li>
<li><b>&gt;10 Å</b>: 불확실 - 도킹 pose 해석 주의</li>
</ul>

<h3>3) pTM / ipTM (0-1)</h3>
<ul>
<li><b>pTM</b>: 전체 복합체의 접힘 신뢰도 (글로벌 토폴로지)</li>
<li><b>ipTM</b>: 체인 간 상대 배치 신뢰도. 보고서의 "ipTM(나노바디|항원)"은
<b>항원 좌표계에 정렬했을 때 나노바디가 얼마나 정확히 놓이는지</b>를 나타내며, 나노바디 결합 pose 평가의
핵심 지표입니다.</li>
<li>탐색용 참고 기준(미보정): <b>&gt;0.8</b> 높음, <b>0.6-0.8</b> 중간, <b>&lt;0.6</b> 낮음.
새 construct에서 보편 합격 기준으로 쓰기보다 다른 지표와 함께 보세요.</li>
</ul>

<h3>4) ipSAE (인터페이스 예측 점수, 0-1)</h3>
<p>Dunbrack lab이 제안한 인터페이스 점수로, <b>PAE가 좋은 잔기 쌍</b>을 중심으로 계산하므로 무질서
영역/부속 도메인의 영향을 줄이는 데 도움이 됩니다. 이 저장소에서는 구조 가설 우선순위화 지표로 사용합니다.</p>
<ul>
<li><b>ipSAE</b>: 잔기별 d0 사용 (주 지표) / <b>ipSAE_d0chn</b>: 체인 전체 길이 기반 d0 /
<b>ipSAE_d0dom</b>: 도메인 크기 기반 d0</li>
<li>탐색용 참고 기준(미보정): <b>≥0.6</b> 높음, 0.4-0.6 중간, &lt;0.4 낮음
(절대값보다는 여러 모델/디자인 간 <b>상대 비교 지표</b>로 쓰는 것이 안전)</li>
</ul>

<h3>5) pDockQ / pDockQ2 / LIS</h3>
<ul>
<li><b>pDockQ</b>: 인터페이스 pLDDT 기반 (품질 추정)</li>
<li><b>pDockQ2</b>: 인터페이스 pLDDT + PAE 조합, 0.005-1.315 범위로 보정.
탐색용 참고 기준(미보정)은 &gt;0.8 높음, 0.5-0.8 중간, &lt;0.5 낮음입니다.</li>
<li><b>LIS</b>: 인터페이스 접촉의 국소 상호작용 점수 (PAE 변환)</li>
</ul>

<h3>6) DockQ (참조 구조가 있을 때, 0-1)</h3>
<p>실험 구조(정답)가 있을 때만 계산 가능. CAPRI 등급: <b>≥0.80</b> High, <b>0.49-0.80</b> Medium,
<b>0.23-0.49</b> Acceptable, <b>&lt;0.23</b> Incorrect. 새 나노바디처럼 정답 구조가 없는 경우에는
사용할 수 없습니다. ipTM/ipSAE/PAE는 구조 신뢰도 지표이며 Kd, 결합 에너지, 결합력 변화가 아닙니다.</p>

<h3>7) 해석 시 주의사항 (실험 연구자용)</h3>
<ul>
<li>모델은 <b>서열 정보만</b> 사용합니다. 글리코실화, 금속 이온, 보조인자, 비천연 변형,
pH/온도 조건 등은 반영되지 않습니다.</li>
<li>항원에 <b>당쇄(glycan)</b>가 있거나 막 단백질인 경우 예측 정확도가 떨어질 수 있습니다.</li>
<li>MSA 품질이 낮은 항원(고유 서열, 소형 단백질)은 ipTM이 낮게 나올 수 있습니다.
(MSA를 의도적으로 비우려면 <span class="mono">--msa empty</span>를 사용하고, 가능하면 MSA 서버/캐시와 비교하세요.)</li>
<li>유연한 loop, 긴 CDR3, 여러 도메인 구조는 낮은 pLDDT로 나타날 수 있습니다. 이는 실제로
유연하다는 의미일 수 있으므로 "틀렸다"기보다 "불확실하다"로 해석하세요.</li>
<li>확산 샘플 수(diffusion samples)를 늘려 여러 모델을 비교하면 수렴 여부를 확인할 수 있습니다.
여러 모델에서 <b>일관되게</b> 같은 인터페이스가 나오면 신뢰도가 높습니다.</li>
<li>최종 판단은 SPR/BLI 결합 실험, 결정구조/Cryo-EM 등 <b>실험 검증</b>과 함께 하십시오.</li>
</ul>
</div>
"""


def build_html(results, run_dir):
    models = results["models"]
    best = models[0]
    structs = {m["stem"]: Path(m["cif"]).read_text(encoding="utf-8")
               for m in models if m.get("cif") and Path(m["cif"]).exists()}
    labels = {}
    for m in models:
        ov = m.get("overlay") or {}
        if ov.get("path") and Path(ov["path"]).exists():
            structs[f"overlay_{m['stem']}"] = Path(ov["path"]).read_text(encoding="utf-8")
    for m in models:
        mx = m["ipsae"].get("max") or {}
        if results.get("nanobody_chain"):
            primary = primary_scores(m, results)
            sub = (f"primary ipTM={fmt(primary.get('iptm'))} · "
                   f"primary ipSAE={fmt(primary.get('ipsae'))} · pDockQ2={fmt(mx.get('pdockq2'))}")
        else:
            sub = f"pTM={fmt(m['boltz'].get('ptm'))} · 평균 pLDDT={fmt(m['plddt'].get('mean'), 1)}"
        labels[m["stem"]] = {"label": f"model_{m['index']}", "sub": sub}
    meta = dict(labels)
    meta["__best__"] = best["stem"]
    for m in models:
        ov = m.get("overlay") or {}
        if ov.get("path"):
            meta[f"overlay_{m['stem']}"] = {
                "label": f"overlay model_{m['index']}",
                "sub": f"항원 CA 정렬 RMSD {fmt(ov.get('rmsd'), 2)} Å" if ov.get("rmsd") else "",
            }

    tabs = "".join(
        f"<button type='button' role='tab' class='tab{' active' if m['stem'] == best['stem'] else ''}' "
        f"aria-selected='{'true' if m['stem'] == best['stem'] else 'false'}' "
        f"tabindex='{'0' if m['stem'] == best['stem'] else '-1'}' "
        f"data-key='{esc(m['stem'])}'><span class='t'>{esc(labels[m['stem']]['label'])}</span>"
        f"<span class='s'>{esc(labels[m['stem']]['sub'])}</span></button>"
        for m in models
    )

    warn = ""
    legacy_warn = ""
    if results.get("warnings"):
        warn = "<div class='warnbox'><b>경고</b><ul>" + "".join(
            f"<li>{html.escape(w)}</li>" for w in results["warnings"]) + "</ul></div>"
    if not has_analysis_provenance(results):
        legacy_warn = (
            "<div class='warnbox'><b>Legacy analysis warning</b><br>"
            "이 results.json에는 <span class='mono'>analysis_provenance</span>가 없습니다. "
            "보고서는 원본 historical source를 수정하지 않고 그대로 표시하지만, 실행 환경/분석 버전 "
            "추적 정보가 부족하므로 서로 다른 실행 간 비교 해석에 주의하세요.</div>"
        )

    chains_line = ", ".join(
        f"{esc(c['id'])} = {esc(chain_role_label(results, c['id']))}"
        f"({fmt_num(c.get('n_res'), 0)} aa)" for c in results["chains"])

    cdr = results.get("cdr") or {}
    cdr_summary = ""
    if cdr.get("available") and results.get("nanobody_chain"):
        cdr_summary = "<br>CDR (IMGT): " + ", ".join(
            f"<span class='mono'>{esc(k)}={esc(v)}</span>"
            for k, v in (cdr.get("cdr_sequences") or {}).items())

    def _js_json(obj):
        return json.dumps(obj).replace("<", "\\u003c")

    js = (JS_TEMPLATE
          .replace("__STRUCTS__", _js_json(structs))
          .replace("__META__", _js_json(meta)))

    report_figures = results.get("figures") or {}
    comparison_fig = figure_html(
        report_figures.get("model_comparison"),
        "모델 비교 요약 (분석 단계에서 생성된 경우)",
        [Path(run_dir)],
    )
    comparison_fig_block = f'<div class="card">{comparison_fig}</div>' if comparison_fig else ""

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boltz-2 nanobody report - {html.escape(results['run_name'])}</title>
<link rel="stylesheet" href="molstar.css">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>{"Boltz-2 나노바디-항원 구조 리포트" if results.get("nanobody_chain")
     else "Boltz-2 구조 예측 리포트 (타겟 단독/복합체)"}</h1>
<p class="sub">실행 이름: <b>{html.escape(results['run_name'])}</b> · 생성: {html.escape(results['created'])} ·
모델 수: {len(models)} · 체인: {chains_line} · {
  f"나노바디 체인: <b>{esc(results['nanobody_chain'])}</b> (역할 판별: {esc(results['role_source'])})"
  if results.get("nanobody_chain") else
  "나노바디 없음 (target-only 모드: CDR/파라토프 등 나노바디 전용 지표는 표시되지 않습니다)"}
{cdr_summary}<br>결과 폴더: <span class="mono">{html.escape(str(run_dir))}</span></p>

<div class="card"><h3 style="margin-top:0">종합 요약 (선택된 모델: model_{fmt_num(best.get('index'), 0)})</h3>
{verdict_html(results)}
    <p class="note" style="margin-bottom:0">위 배지는 보정되지 않은 탐색용 휴리스틱입니다. 선택된 모델 표시는
    정렬 기준에 따른 대표 구조 표시이며 ensemble consensus나 결합/비결합 판정이 아닙니다.</p>
</div>
{warn}
{legacy_warn}

<h2>1. 모델 비교</h2>
{build_summary_table(results)}
{comparison_fig_block}

<h2>2. 3D 구조 (Mol* 뷰어)</h2>
<div class="card">
<div class="model-tabs" role="tablist" aria-label="모델 선택">{tabs}</div>
<div id="viewer-host"><div id="molstar-viewer" style="width:100%;height:100%"></div></div>
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px" id="theme-target" data-key="">
<button class="btn" onclick="setTheme('chain-id')">체인별 색상</button>
<button class="btn" onclick="setTheme('plddt-confidence')">pLDDT 색상</button>
<button class="btn" onclick="setTheme('uncertainty')" title="B-factor/pLDDT 기반: 파랑=낮음, 빨강=높음 (pLDDT 는 높을수록 신뢰)">pLDDT(B-factor) 색상</button>
<button class="btn" onclick="setTheme('element-symbol')">원소별 색상</button>
<button class="btn" onclick="showOverlay()">참조 구조 겹쳐보기</button>
<button class="btn" onclick="RB.resetCamera('main')">카메라 리셋</button>
</div>
<p class="note" id="viewer-info"></p>
<div class="legend">
<span><span class="dot" style="background:#0053d6"></span>pLDDT &gt;90 (매우 높음)</span>
<span><span class="dot" style="background:#65cbf3"></span>70-90</span>
<span><span class="dot" style="background:#ffdb13"></span>50-70</span>
<span><span class="dot" style="background:#ff7d45"></span>&lt;50</span>
<span>· Mol* 우측 패널(Controls)에서 Coloring/Representation 변경 가능</span>
</div>
<p class="note">참조 구조를 함께 지정했다면 <b>참조 구조 겹쳐보기</b>로 예측 구조(모델 체인)와
실험 구조(참조 체인, 예: RA/RB)를 겹쳐 볼 수 있습니다. 항원 체인 기준으로 정렬합니다.</p>
<p class="note">버튼의 색상 변경은 현재 표시된 모델의 기본 표현에 적용됩니다. 적용이 안 되면 Mol* 좌측
패널에서 직접 Coloring을 선택하세요. 뷰어는 로컬 <span class="mono">molstar.js</span>를 사용하므로
인터넷 없이 동작합니다.</p>
<div id="diag" class="diag">Mol* viewer 로딩...</div>
</div>

<h2>3. 모델별 상세</h2>
{''.join(build_model_section(results, m, [Path(run_dir)]) for m in models)}

{INTERPRET_GUIDE}

<h2>부록: 파일과 재현 방법</h2>
<div class="card kv">
<p><b>예측 구조 (mmCIF)</b><br><span class="mono">{html.escape(str(results.get('predictions_dir') or '-'))}/*.cif</span></p>
<p><b>신뢰도(JSON)</b> Boltz가 계산한 pTM/ipTM/pLDDT 원본<br><span class="mono">{html.escape(str(results.get('predictions_dir') or '-'))}/confidence_*.json</span></p>
<p><b>PAE 행렬</b> numpy npz (pae/plddt) <br><span class="mono">{html.escape(str(results.get('predictions_dir') or '-'))}/pae_*.npz</span></p>
<p><b>ipSAE 원본 출력</b> Dunbrack lab 스크립트가 생성한 표 + 잔기별 파일 + PyMOL 스크립트<br>
<span class="mono">{html.escape(str(best['ipsae'].get('out_txt') or ''))}</span><br>
<span class="mono">{html.escape(str(best['ipsae'].get('pml') or ''))}</span></p>
<p><b>리포트 재생성</b><br><span class="mono">./run.sh --report-only {html.escape(str(run_dir))} [--reference ref.cif]</span></p>
<p><b>PyMOL로 열기</b><br><span class="mono">pymol {html.escape(Path(best['cif']).name if best.get('cif') else '-')} {html.escape(Path(best['ipsae'].get('pml')).name if best['ipsae'].get('pml') else '-')}</span>
 (예측 폴더에서 실행)</p>
<p><b>인용</b><br>
Boltz-2: Passaro et al., 2025 (bioRxiv) · ipSAE: Dunbrack, 2025 (bioRxiv 2025.02.10.637595) ·
DockQ: Mirabello &amp; Wallner, 2024 · Mol*: Sehnal et al., 2021</p>
</div>
<p class="sub">이 리포트는 자동 생성되었습니다. 지표 계산 방식과 임계값은 README.md의
"지표 해석" 절을 참고하세요.</p>
</div>
<script src="molstar.js"></script>
<script src="report.js"></script>
<script>{js}</script>
</body>
</html>
"""


def _tsv_num(v, nd=4):
    """TSV 수치 셀: 유한 수치만 문자열화, 그 외 빈칸."""
    x = safe_float(v)
    if x is None:
        return ""
    return f"{x:.{int(nd)}f}"


def write_summary_tsv(results, path):
    cols = ["model", "ipTM_nb_in_ag", "ipTM_ag_in_nb", "pTM", "ipSAE", "ipSAE_source",
            "primary_scope", "primary_source",
            "ipSAE_d0chn", "ipSAE_d0dom", "ipTM_d0chn", "pDockQ", "pDockQ2", "LIS",
            "nanobody_pLDDT_mean", "iface_mean_PAE", "n_contacts", "DockQ", "DockQ_kind",
            "CAPRI"]
    lines = ["\t".join(cols)]
    for e in results["models"]:
        mx = e["ipsae"].get("max") or {}
        iface = e["interface_8A"] or {}
        primary = primary_scores(e, results)
        lines.append("\t".join(str(x) for x in [
            f"model_{e['index']}",
            _tsv_num(primary.get("iptm")),
            _tsv_num(e["boltz_pair_iptm"]["antigen_in_nanobody_frame"]),
            _tsv_num(e["boltz"].get("ptm")),
            _tsv_num(primary.get("ipsae")),
            primary.get("ipsae_source") or "-",
            primary.get("scope") or "-",
            primary_source_label(primary),
            _tsv_num(mx.get("ipsae_d0chn")), _tsv_num(mx.get("ipsae_d0dom")),
            _tsv_num(mx.get("iptm_d0chn")), _tsv_num(mx.get("pdockq")), _tsv_num(mx.get("pdockq2")),
            _tsv_num(mx.get("lis")),
            _tsv_num((e["plddt"]["per_chain"].get(results["nanobody_chain"]) or {}).get("mean"), 2),
            _tsv_num(iface.get("pae_mean_ab"), 2),
            iface.get("n_contacts") if iface.get("n_contacts") is not None else "-",
            _tsv_num((e.get("dockq") or {}).get("headline")),
            ((e.get("dockq") or {}).get("headline_kind") or "-"),
            ((e.get("dockq") or {}).get("capri") or "-"),
        ]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    run_dir = args.run_dir or Path(results["run_dir"])
    local_run_dir = rebase_result_paths(results, args.results, run_dir=args.run_dir)
    validate_results(results, local_run_dir)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for asset in ("molstar.js", "molstar.css", "report.js"):
        src = ASSETS / asset
        if not src.exists():
            raise FileNotFoundError(f"missing asset {src}; run setup.sh")
        shutil.copy2(src, outdir / asset)

    run_dir = local_run_dir
    (outdir / "index.html").write_text(build_html(results, run_dir), encoding="utf-8")
    write_summary_tsv(results, outdir / "summary.tsv")
    src_res = Path(args.results).resolve()
    dst_res = (outdir / "results.json").resolve()
    if src_res != dst_res:
        shutil.copy2(src_res, dst_res)
    print(f"[report] wrote {outdir / 'index.html'}")
    print(f"[report] summary table: {outdir / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
