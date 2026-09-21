#!/usr/bin/env python
"""Analyze Boltz-2 prediction outputs for nanobody/antigen complexes.

Collects confidence metrics, per-chain statistics, interface analysis, official
ipSAE scores (Dunbrack lab), DockQ (when a reference structure is given) and
generates figures. Writes a single results.json consumed by make_report.py.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics
import msa_cache
import runtime_state
from metrics import (
    annotate_cdrs_imgt,
    capri_class,
    chain_lengths,
    find_hmmer_dir,
    json_safe,
    per_chain_plddt_stats,
    read_structure_tokens,
    run_dockq,
    run_official_ipsae,
    sequence_of,
    token_chain_array,
)
from runtime_state import validate_prediction_outputs

# matplotlib(그림)은 분석 자체에는 필수가 아니다. 없으면 그림만 건너뛰고
# results.json 은 정상 생성한다 (예측까지 끝난 뒤 라이브러리 하나로 전체가 실패하지 않도록).
FIGURES_IMPORT_ERROR = None
try:
    import figures
except Exception as _fig_exc:  # noqa: BLE001
    figures = None
    FIGURES_IMPORT_ERROR = str(_fig_exc)

VHH_MOTIFS = ("WGQGT", "WGQG", "WGKGT", "RGQGT")
RESULTS_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# discovery helpers
# ---------------------------------------------------------------------------

def find_predictions_dir(run_dir: Path, boltz_dir: Path | None = None) -> Path:
    """예측 산출물 디렉터리를 찾는다.

    C-03: 캐시/비캐시 등 여러 boltz_results_* 가 있으면 '가장 최근'을 쓴다.
    run.sh 가 넘긴 --boltz-dir 이 있으면 그것을 우선한다.
    """
    if boltz_dir:
        cands = [p for p in [boltz_dir / "predictions" / d.name
                             for d in (boltz_dir / "predictions").glob("*")] if p.is_dir()]
        cands += sorted((boltz_dir / "predictions").glob("*"))
        for c in cands:
            if any(c.glob("confidence_*_model_*.json")):
                return c

    marker = run_dir / ".active_boltz_dir"
    if marker.exists():
        active = Path(marker.read_text().strip())
        if active.exists():
            try:
                active.resolve().relative_to(run_dir.resolve())
            except ValueError:
                active = None
        if active:
            for c in sorted((active / "predictions").glob("*")):
                if any(c.glob("confidence_*_model_*.json")):
                    return c

    cands = [c for c in run_dir.glob("**/predictions/*")
             if any(c.glob("confidence_*_model_*.json"))]
    if cands:
        return max(cands, key=lambda p: max(f.stat().st_mtime for f in p.glob("confidence_*_model_*.json")))
    raise SystemExit(
        f"[analyze] {run_dir} 안에 Boltz 예측 산출물이 없습니다.\n"
        "  - MSA 생성 실패 또는 예측이 중단되었을 수 있습니다.\n"
        "  - 해결: 같은 명령을 다시 실행하세요 (예: ./run.sh --batch <표> --name <배치> --jobs <job>).\n"
        "  - MSA 서버가 불안정하면 --msa empty 로 빠르게 확인할 수도 있습니다.")


def model_files(pred_dir: Path, expected_samples: int | None = None):
    try:
        validate_prediction_outputs(pred_dir, expected_samples=expected_samples)
    except ValueError as exc:
        raise SystemExit(f"[analyze] incomplete or invalid prediction outputs: {exc}") from exc
    out = []
    def _model_key(p):
        m = re.search(r"_model_(\d+)\.json$", p.name)
        return int(m.group(1)) if m else 0

    for conf in sorted(pred_dir.glob("confidence_*_model_*.json"), key=_model_key):
        m = re.search(r"_model_(\d+)\.json$", conf.name)
        if not m:
            continue
        idx = int(m.group(1))
        stem = conf.name[len("confidence_"):-len(".json")]
        cif = pred_dir / f"{stem}.cif"
        pae = pred_dir / f"pae_{stem}.npz"
        plddt = pred_dir / f"plddt_{stem}.npz"
        if not cif.exists():
            cif_pdb = pred_dir / f"{stem}.pdb"
            cif = cif_pdb if cif_pdb.exists() else cif
        out.append({"index": idx, "stem": stem, "confidence": conf, "cif": cif,
                    "pae": pae, "plddt": plddt})
    if not out:
        raise FileNotFoundError(f"no models found in {pred_dir}")
    return out


def load_yaml_chains(yaml_path: Path):
    """Minimal YAML reader for Boltz input files (protein chains only)."""
    import yaml  # boltz dependency

    data = yaml.safe_load(yaml_path.read_text())
    chains = []
    for entry in data.get("sequences", []):
        for kind, spec in entry.items():
            if kind != "protein":
                continue
            chains.append({"id": spec["id"], "sequence": spec["sequence"], "kind": kind})
    return chains


def detect_nanobody_chain(tokens):
    """Heuristic VHH detection: length + conserved WGxG motif + 2 cysteines."""
    best = None
    ties = 0
    for ch in chain_lengths(tokens):
        seq = sequence_of(tokens, ch)
        n = len(seq)
        if not (90 <= n <= 160):
            continue
        score = 0
        if any(motif in seq for motif in VHH_MOTIFS):
            score += 2
        if seq.count("C") >= 2:
            score += 1
        if "WGQG" in seq or "WGKG" in seq:
            score += 1
        if best is None or score > best[1]:
            best = (ch, score)
            ties = 1
        elif score == best[1]:
            ties += 1
    # B-18: 최소 점수 미달이면 자동 판별을 포기한다 (항원을 나노바디로 오인 방지)
    if best and best[1] >= 3 and ties == 1:
        return best[0]
    return None


def resolve_roles(tokens, metadata, args):
    """Return (antigen_chain, nanobody_chain) plus a 'role_source' label."""
    chain_ids = list(chain_lengths(tokens))
    # metadata 에 target_only 로 기록되어 있으면 플래그가 없어도 그대로 존중한다
    if args.no_nanobody or (metadata and metadata.get("target_only")
                            and metadata.get("nanobody_chain") is None
                            and not args.nanobody_chain):
        return (chain_ids[0] if chain_ids else None), None, "target-only(metadata)", \
            list(chain_ids)
    nanobody = args.nanobody_chain
    source = "cli" if nanobody else None
    if nanobody is None and metadata:
        nanobody = metadata.get("nanobody_chain")
        source = "metadata" if nanobody else None
    if nanobody is None:
        nanobody = detect_nanobody_chain(tokens)
        source = "auto-detect" if nanobody else None
    if nanobody is None:
        raise SystemExit(
            "나노바디 체인을 자동 판별하지 못했습니다. "
            "--nanobody-chain <ID> 또는 --no-nanobody 를 명시하세요 "
            f"(체인: {chain_ids})")
    if nanobody not in chain_ids:
        raise SystemExit(f"--nanobody-chain '{nanobody}' 이 입력 체인에 없습니다 (체인: {chain_ids})")
    antigen = args.antigen_chain
    if antigen is None and metadata:
        antigen = metadata.get("antigen_chain")
    # H-05: 항원은 체인 하나가 아니라 '집합'이다. 우선순위: CLI > 저장값 > metadata > 폴백.
    antigen_chains = []
    if metadata:
        antigen_chains = [c["id"] for c in metadata.get("chains", [])
                          if c.get("role") in ("antigen", "target") and c.get("id") != nanobody]
    restored_antigens = getattr(args, "_restored_antigen_chains", None)
    explicit_ag_chains = getattr(args, "antigen_chains", None)
    if explicit_ag_chains:
        # Fv(VH+VL) 처럼 binder 외에 다른 사슬(VL)이 있는 입력에서, 항원 집합을
        # 명시해 VL 이 항원 인터페이스에 섞이지 않게 한다.
        wanted = [c.strip() for c in str(explicit_ag_chains).split(",") if c.strip()]
        bad = [c for c in wanted if c not in chain_ids]
        if bad:
            raise SystemExit(f"--antigen-chains 에 없는 체인: {bad} (체인: {chain_ids})")
        if nanobody in wanted:
            raise SystemExit(f"--antigen-chains 에 나노바디 체인({nanobody})이 포함됐습니다")
        antigen_chains = wanted
    elif restored_antigens:
        antigen_chains = [c for c in restored_antigens if c in chain_ids and c != nanobody]
    if not antigen_chains:
        antigen_chains = [c for c in chain_ids if c != nanobody]
    if antigen is None:
        antigen = antigen_chains[0] if antigen_chains else None
    if antigen and antigen not in chain_ids:
        raise SystemExit(f"--antigen-chain '{antigen}' 이 입력 체인에 없습니다 (체인: {chain_ids})")
    if antigen and nanobody and antigen == nanobody and len(chain_ids) > 1:
        raise SystemExit("항원 체인과 나노바디 체인이 같습니다. --nanobody-chain/--antigen-chain 을 확인하세요")
    if antigen and antigen not in antigen_chains:
        antigen_chains = [antigen] + [c for c in antigen_chains if c != antigen]
    return antigen, nanobody, source, antigen_chains


def _explicit_flags(argv):
    flags = set()
    for item in argv:
        if not item.startswith("--"):
            continue
        flags.add(item.split("=", 1)[0])
    return flags


def _load_saved_settings(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "settings" in data:
        return data, data.get("settings") or {}
    return {}, data


def _restore_path(saved, run_dir, saved_run_dir, label, required=False):
    if not saved:
        return None
    p = Path(saved)
    candidates = []
    if not p.is_absolute():
        candidates.append(run_dir / p)
    else:
        try:
            rel = p.relative_to(saved_run_dir)
            candidates.append(run_dir / rel)
        except ValueError:
            pass
        try:
            p.relative_to(run_dir)
            candidates.append(p)
        except ValueError:
            pass
    for cand in candidates:
        if cand.exists():
            return cand
    if required:
        raise SystemExit(
            f"[analyze] 저장된 {label} 경로를 현재 실행 폴더에서 찾지 못했습니다: {saved}\n"
            "  복사본에서 재분석하려면 해당 파일도 실행 폴더 안의 같은 상대 위치에 두세요.")
    return None


def apply_restored_settings(args, run_dir, explicit):
    """Replay analysis settings from a previous results.json/settings JSON.

    Explicit CLI flags override saved values. Absolute paths from a copied source
    run are rebased under the current run directory; they are not followed back to
    the original source tree.
    """
    if not getattr(args, "restore_settings", None):
        return args
    saved_data, settings = _load_saved_settings(args.restore_settings)
    warnings = []
    if settings.get("analysis_contract_version") != 1:
        warnings.append(
            "restore-settings file predates analysis_contract_version=1; "
            "only fields present in that file were replayed")
    expected = {
        "boltz_dir", "yaml_snapshot", "metadata_snapshot", "no_nanobody", "nanobody_chain",
        "antigen_chain", "reference", "dockq_mapping", "dockq_allowed_mismatches",
        "pae_cutoff", "dist_cutoff", "no_figures",
    }
    missing = sorted(k for k in expected if k not in settings)
    if missing:
        warnings.append("restore-settings missing fields: " + ", ".join(missing))
    args._restore_warnings = warnings
    saved_run_dir = Path(settings.get("run_dir") or saved_data.get("run_dir") or run_dir)
    if "--boltz-dir" not in explicit:
        restored = _restore_path(settings.get("boltz_dir"), run_dir, saved_run_dir, "boltz_dir")
        if restored:
            args.boltz_dir = restored
    for flag, attr, label, required in (
        ("--yaml", "yaml", "yaml", False),
        ("--metadata", "metadata", "metadata", False),
        ("--reference", "reference", "reference", bool(settings.get("reference"))),
    ):
        if flag in explicit:
            continue
        saved_path = settings.get(f"{attr}_snapshot") or settings.get(attr)
        restored = _restore_path(saved_path, run_dir, saved_run_dir, label, required=required)
        if restored:
            setattr(args, attr, restored if attr in ("yaml", "metadata") else str(restored))
    if "--nanobody-chain" in explicit:
        args.no_nanobody = False
    for flag, attr in (
        ("--no-nanobody", "no_nanobody"),
        ("--nanobody-chain", "nanobody_chain"),
        ("--antigen-chain", "antigen_chain"),
        ("--dockq-mapping", "dockq_mapping"),
        ("--dockq-allowed-mismatches", "dockq_allowed_mismatches"),
        ("--pae-cutoff", "pae_cutoff"),
        ("--dist-cutoff", "dist_cutoff"),
        ("--ipsae-script", "ipsae_script"),
        ("--no-figures", "no_figures"),
    ):
        if attr == "no_nanobody" and "--nanobody-chain" in explicit:
            continue
        if flag not in explicit and attr in settings:
            setattr(args, attr, settings[attr])
    if "--no-nanobody" not in explicit and "--nanobody-chain" not in explicit:
        if "nanobody_chain" not in settings and "nanobody_chain" in saved_data:
            args.nanobody_chain = saved_data.get("nanobody_chain")
        if "no_nanobody" not in settings and saved_data.get("nanobody_chain") is None:
            args.no_nanobody = True
    if ("--antigen-chain" not in explicit and "antigen_chain" not in settings
            and saved_data.get("antigen_chain")):
        args.antigen_chain = saved_data.get("antigen_chain")
    # 항원 집합(복수 체인)도 복원한다. 저장 settings 우선, 없으면 top-level 값.
    if "--antigen-chains" not in explicit:
        saved_ags = settings.get("antigen_chains") or saved_data.get("antigen_chains")
        if saved_ags:
            args._restored_antigen_chains = list(saved_ags)
    return args


def _portable_snapshot(path, out_dir, label):
    """Copy a replay-critical input into analysis/inputs and return its path."""
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return None
    snap_dir = Path(out_dir) / "inputs"
    snap_dir.mkdir(parents=True, exist_ok=True)
    dest = snap_dir / f"{label}{src.suffix or '.txt'}"
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return str(dest)


def role_map(antigen_chains, nanobody_chain):
    roles = {ch: "antigen" for ch in antigen_chains if ch}
    if nanobody_chain:
        roles[nanobody_chain] = "nanobody"
    return roles


def _finite_rank_value(value) -> tuple[int, float]:
    """Sort observed finite scores ahead of missing/non-finite scores."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0, 0.0
    return (1, number) if math.isfinite(number) else (0, 0.0)


def model_rank_key(entry: dict, target_only: bool) -> tuple:
    """Return a deterministic model key without treating missing scores as zero."""

    index = -int(entry.get("index") or 0)
    if target_only:
        boltz = entry.get("boltz") or {}
        plddt = entry.get("plddt") or {}
        return (*_finite_rank_value(boltz.get("ptm")),
                *_finite_rank_value(boltz.get("confidence_score")),
                *_finite_rank_value(plddt.get("mean")), index)
    primary = entry.get("primary_interface") or metrics.primary_interface_scores(entry)
    return (*_finite_rank_value(primary.get("iptm")),
            *_finite_rank_value(primary.get("ipsae")), index)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_obj(value):
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()[:16]


def _normalize_for_hash(value):
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize_for_hash(v) for v in value]
    return value


def _classify_msa(msa_value, sequence, requested):
    if msa_value == "empty":
        return {
            "class": "empty",
            "delta_eligible": requested in (None, "empty"),
            "reason": None if requested in (None, "empty") else
            f"requested MSA policy was {requested!r}, YAML uses empty",
        }
    if msa_value in (None, "", 0, "0"):
        return {
            "class": "server" if requested != "cache" else "cache-requested-but-yaml-server",
            "delta_eligible": requested != "cache",
            "reason": None if requested != "cache" else
            "requested cache policy, but YAML has no per-chain cache MSA paths",
        }
    if msa_cache.is_user_msa(msa_value):
        path = Path(str(msa_value)).expanduser()
        n_rows, valid = msa_cache.read_msa_summary(path, query=sequence)
        path_name = path.name
        unpaired = False
        if valid:
            with path.open(newline="") as handle:
                rows = csv.DictReader(handle)
                unpaired = all(row.get("key", "").strip() == "-1" for row in rows)
        if requested == "cache" and valid and unpaired:
            return {
                "class": "cache-unpaired",
                "delta_eligible": True,
                "path_name": path_name,
                "rows": n_rows,
                "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pairing": "unpaired-per-sequence-cache",
                "sequence_hash": hashlib.sha256(sequence.strip().upper().encode()).hexdigest()[:16],
                "reason": None,
            }
        return {
            "class": "custom-known",
            "delta_eligible": False,
            "path_name": path_name,
            "rows": n_rows,
            "valid_for_sequence": bool(valid),
            "reason": "custom MSA provenance is not comparable unless explicitly matched",
        }
    if msa_cache.looks_like_msa_path(msa_value):
        return {
            "class": "custom-unknown-fail-closed",
            "delta_eligible": False,
            "path_name": Path(str(msa_value)).name,
            "reason": "MSA path was declared but file is unavailable in this copy",
        }
    return {
        "class": "custom-unknown-fail-closed",
        "delta_eligible": False,
        "value": str(msa_value),
        "reason": "unknown MSA declaration",
    }


def yaml_input_provenance(yaml_path, requested_msa=None):
    """Summarize actual Boltz YAML inputs for delta/comparison eligibility."""
    if not yaml_path:
        return {
            "schema_version": 1,
            "available": False,
            "delta_eligible": False,
            "reason": "no YAML snapshot available",
        }
    import yaml

    p = Path(yaml_path)
    if not p.exists():
        return {
            "schema_version": 1,
            "available": False,
            "delta_eligible": False,
            "reason": f"YAML snapshot not found: {p}",
        }
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    chains = []
    pae_reasons = []
    for entry in data.get("sequences") or []:
        protein = entry.get("protein") if isinstance(entry, dict) else None
        if not protein:
            if isinstance(entry, dict):
                pae_reasons.append("non-protein sequence entry: " + ",".join(entry.keys()))
            else:
                pae_reasons.append("non-dict sequence entry")
            continue
        ids = protein.get("id")
        if not isinstance(ids, list):
            ids = [ids]
        seq = str(protein.get("sequence") or "")
        msa_info = _classify_msa(protein.get("msa"), seq, requested_msa)
        chains.append({
            "ids": [str(i) for i in ids if i is not None],
            "sequence_hash": hashlib.sha256(seq.strip().upper().encode()).hexdigest()[:16],
            "length": len(seq),
            "msa": msa_info,
            "templates_hash": _hash_obj(_normalize_for_hash(protein.get("templates"))),
            "modifications_hash": _hash_obj(_normalize_for_hash(protein.get("modifications"))),
            "properties_hash": _hash_obj(_normalize_for_hash(protein.get("properties"))),
            "cyclic": bool(protein.get("cyclic", False)),
        })
        if protein.get("modifications"):
            pae_reasons.append(
                "protein modifications/PTM are not eligible for protein-only PAE-derived analysis"
            )
    constraints = _normalize_for_hash(data.get("constraints") or [])
    nonempty_constraint_hash = _hash_obj(constraints)
    reasons = [
        f"chain {','.join(c['ids'])}: {c['msa'].get('reason')}"
        for c in chains if not c["msa"].get("delta_eligible", True)
    ]
    # These top-level Boltz features require matching additional inputs and have
    # not been validated for WT-mutant confidence comparisons in this pipeline.
    advanced = {key: data.get(key) for key in ("templates", "properties")}
    nonprotein = any(not isinstance(e, dict) or "protein" not in e for e in data.get("sequences") or [])
    if not chains or nonprotein or any(advanced.values()):
        reasons.append("advanced/nonprotein input comparison is not validated")
    if nonprotein:
        pae_reasons.append("ligand/nucleic/non-protein sequence entries are unsupported")
    return {
        "schema_version": 1,
        "available": True,
        "requested_msa_policy": requested_msa,
        "chains": chains,
        "msa_classes": {cid: c["msa"].get("class") for c in chains for cid in c["ids"]},
        "pairing_policy": "cache-unpaired" if any(c["msa"].get("class") == "cache-unpaired"
                                                 for c in chains) else
        "server-or-empty-or-custom",
        "constraints_hash": nonempty_constraint_hash,
        "constraints_count": len(constraints) if isinstance(constraints, list) else 1,
        "constraints": constraints,
        "advanced_input_hash": _hash_obj(advanced),
        "delta_eligible": not reasons,
        "delta_ineligible_reasons": reasons,
        "comparison_excludes": ["role_source"],
        "pae_metrics_eligible": not pae_reasons,
        "pae_metrics_ineligible_reasons": pae_reasons,
    }


# ---------------------------------------------------------------------------
# per-model analysis
# ---------------------------------------------------------------------------

def _token_identity(t):
    return (
        str(t.get("chain", "")),
        int(t.get("resnum")),
        str(t.get("icode", "")),
        str(t.get("aa", "")),
    )


def token_identity_mismatch(reference_tokens, model_tokens):
    """Return a concise mismatch report for ordered chain/resnum/icode/aa tokens."""

    ref = [_token_identity(t) for t in reference_tokens]
    cur = [_token_identity(t) for t in model_tokens]
    if ref == cur:
        return None
    limit = min(len(ref), len(cur))
    first = next((i for i in range(limit) if ref[i] != cur[i]), None)
    if first is None:
        first = limit
    return {
        "reference_count": len(ref),
        "model_count": len(cur),
        "first_mismatch_index": int(first),
        "reference": list(ref[first]) if first < len(ref) else None,
        "model": list(cur[first]) if first < len(cur) else None,
    }


def pae_metric_eligibility(input_provenance):
    if not input_provenance or not input_provenance.get("available"):
        return True, []
    return (
        bool(input_provenance.get("pae_metrics_eligible", True)),
        list(input_provenance.get("pae_metrics_ineligible_reasons") or []),
    )


def _skipped_ipsae(reason):
    return {
        "union": None,
        "asym_nb_frame_ag": {},
        "asym_ag_frame_nb": {},
        "max": {},
        "out_txt": None,
        "byres": None,
        "pml": None,
        "error": None,
        "fallback": False,
        "fallback_error": None,
        "byres_rows": [],
        "source": "not-run",
        "skipped": reason,
    }


def _empty_interface(cutoff):
    return {"n_contacts": 0, "cutoff": cutoff, "residues_a": [], "residues_b": []}


def analyze_model(model, tokens, chains_roles, args, cdr_info, out_dir, input_provenance=None):
    """모델 한 개를 분석한다.

    주의(R01): 좌표/접촉 계산에는 반드시 '해당 모델'의 CIF 토큰을 써야 한다.
    인자로 받는 tokens 는 체인 구성/서열 확인용(모델 0)이며, 여기서 모델별로 다시 읽는다.
    """
    antigen_chain, nanobody_chain = chains_roles[0], chains_roles[1]
    antigen_chains = list(chains_roles[2]) if len(chains_roles) > 2 else (
        [antigen_chain] if antigen_chain else [])
    conf = json.loads(Path(model["confidence"]).read_text())
    plddt_raw = np.load(model["plddt"])["plddt"].astype(float)
    plddt = plddt_raw * 100.0 if (plddt_raw.size and float(plddt_raw.max()) <= 1.0) else plddt_raw
    pae = np.load(model["pae"])["pae"].astype(float)

    model_tokens = read_structure_tokens(model["cif"])       # R01: 모델별 좌표
    n_tokens = len(model_tokens)
    dims_ok = (plddt.shape[0] == n_tokens) and (pae.shape[0] == n_tokens)
    mismatch = token_identity_mismatch(tokens, model_tokens)
    identity_ok = mismatch is None
    schema_ok, schema_reasons = pae_metric_eligibility(input_provenance)
    pae_applicable = bool(dims_ok and identity_ok and schema_ok)

    entry = {
        "index": model["index"],
        "stem": model["stem"],
        "cif": str(model["cif"]),
        "confidence_json": str(model["confidence"]),
        "pae_npz": str(model["pae"]),
        "n_tokens": n_tokens,
        "token_dims_ok": bool(dims_ok),
        "token_identity_ok": bool(identity_ok),
        "token_identity_mismatch": mismatch,
        "pae_metrics_applicable": bool(pae_applicable),
        "pae_metrics_ineligible_reasons": list(schema_reasons),
        "boltz": conf,
        "antigen_chains": antigen_chains,
        "has_nanobody": bool(nanobody_chain),
        "target_only": not bool(nanobody_chain),
        "plddt": {
            "mean": float(plddt.mean()) if plddt.size else None,
            "per_chain": per_chain_plddt_stats(plddt, model_tokens) if pae_applicable else {},
        },
        "boltz_pair_iptm": {
            "nanobody_in_antigen_frame": _pair_iptm(conf, token_chain_array(model_tokens),
                                                    nanobody_chain, antigen_chain),
            "antigen_in_nanobody_frame": _pair_iptm(conf, token_chain_array(model_tokens),
                                                    antigen_chain, nanobody_chain),
        },
        "warnings": [],
    }

    if not pae_applicable:
        reasons = []
        if not dims_ok:
            reasons.append(
                f"CIF 단백질 잔기 {n_tokens}개 vs PAE/pLDDT 토큰 {plddt.shape[0]}개 불일치"
            )
        if not identity_ok:
            reasons.append("model 0 대비 ordered chain/resnum/icode/aa 토큰 정체성 불일치")
        reasons.extend(schema_reasons)
        msg = (
            "PAE/pLDDT 인덱스 기반 지표를 계산하지 않습니다: "
            + "; ".join(reasons)
            + " (Boltz confidence JSON 값과 DockQ 는 계속 제공)"
        )
        entry["warnings"].append(msg)
        entry["boltz_pair_iptm"] = {
            # RR04: pair_chains_iptm 키는 ligand 포함 전체 asym_id 기준이라
            # 단백질만으로 만든 ordinal 과 어긋날 수 있다 -> 매핑 불가 시 생략
            "nanobody_in_antigen_frame": None,
            "antigen_in_nanobody_frame": None,
            "skipped": "PAE/pLDDT 인덱스 기반 분석이 부적격이면 체인 ordinal 매핑을 보장할 수 없음",
        }
        entry.update({
            "chain_pairs": [],
            "iptm_from_pae": {},
            "interface_8A": _empty_interface(8.0),
            "interface_10A": _empty_interface(10.0),
            "interface_pae_direction": None,
            "ipsae": _skipped_ipsae("; ".join(reasons)),
            "dockq": _run_dockq_block(model, args, antigen_chain, out_dir, antigen_chains,
                                      nanobody_chain),
            "overlay": None,
            "cdr": cdr_info,
        })
        entry["primary_interface"] = metrics.primary_interface_scores(entry)
        if entry["dockq"] and "best_dockq" in (entry["dockq"] or {}):
            entry["overlay"] = write_overlay(model, Path(args.reference), entry["dockq"],
                                             antigen_chain, out_dir)
        return entry

    chains_arr = token_chain_array(model_tokens)
    d0_total = metrics.calc_d0(max(n_tokens, 19))
    # H-05: 항원 체인 전체를 union 으로 본다 (첫 체인만 보던 문제)
    ag_set = set(antigen_chains)
    idx_ag = np.where(np.isin(chains_arr, list(ag_set)))[0] if ag_set else np.array([], dtype=int)
    # M-04(정밀): 나노바디가 없으면(target-only) 인터페이스 ipTM 은 '해당 없음'이지
    # 0 이 아니다. 0 으로 두면 낮은 신뢰도와 구분되지 않고 순위 기준도 무의미해진다.
    nb_present = bool(nanobody_chain) and nanobody_chain in set(chains_arr.tolist())
    idx_nb = np.where(chains_arr == nanobody_chain)[0] if nb_present else np.array([], dtype=int)
    if nb_present:
        iptm_nb_frame_ag, row_nb = metrics.interchain_iptm(pae, idx_ag, idx_nb, d0_total)
        iptm_ag_frame_nb, row_ag = metrics.interchain_iptm(pae, idx_nb, idx_ag, d0_total)
    else:
        iptm_nb_frame_ag = iptm_ag_frame_nb = None
        row_nb = row_ag = None

    if nb_present and antigen_chains:
        iface = interface_report_union(model_tokens, plddt, pae, antigen_chains, nanobody_chain,
                                       cutoff=8.0)
        iface_ca = interface_report_union(model_tokens, plddt, pae, antigen_chains, nanobody_chain,
                                          cutoff=10.0)
    else:
        iface = {"n_contacts": 0, "cutoff": 8.0, "residues_a": [], "residues_b": []}
        iface_ca = iface

    entry.update({
        "chain_pairs": chain_pair_table(conf, model_tokens, pae, chains_arr, d0_total)
        if len(set(chains_arr.tolist())) > 1 else [],
        "iptm_from_pae": {
            "nanobody_in_antigen_frame": iptm_nb_frame_ag,
            "antigen_in_nanobody_frame": iptm_ag_frame_nb,
            "d0": d0_total,
            # row_nb 는 '나노바디 행(=항원 기준 정렬)' 에서 뽑힌 잔기이다
            "align_residue_nanobody_scored_in_antigen_frame": int(row_nb) if row_nb is not None else None,
            "align_residue_antigen_scored_in_nanobody_frame": int(row_ag) if row_ag is not None else None,
            "applicable": bool(nb_present),
        },
        "interface_8A": iface,
        "interface_10A": iface_ca,
        "interface_pae_direction": ("pae_mean_ab = 항원(A) 기준 정렬에서 나노바디(B) 잔기의 PAE"
                                    if nanobody_chain else None),
        "ipsae": _ipsae_from_official(model, args, pae, chains_arr,
                                      (antigen_chain, nanobody_chain), antigen_chains),
        "dockq": _run_dockq_block(model, args, antigen_chain, out_dir, antigen_chains,
                                  nanobody_chain),
        "overlay": None,
        "cdr": cdr_info,
    })
    entry["primary_interface"] = metrics.primary_interface_scores(entry)
    if entry["dockq"] and "best_dockq" in (entry["dockq"] or {}):
        entry["overlay"] = write_overlay(model, Path(args.reference), entry["dockq"],
                                         antigen_chain, out_dir)
    return entry


def _run_dockq_block(model, args, antigen_chain, out_dir, antigen_chains=None,
                     nanobody_chain=None):
    """DockQ 실행. 참조가 없으면 None.

    C-01: DockQ v2 의 best_dockq 는 인터페이스별 '합'이므로 대표값으로 쓰지 않는다.
    나노바디-항원 인터페이스들의 평균(headline)을 계산하고 CAPRI 등급도 평균 기준으로 매긴다.
    서열 불일치 허용은 --dockq-allowed-mismatches 로 명시한 값만 사용한다.
    """
    if not args.reference:
        return None
    ref = Path(args.reference)
    if not ref.exists():
        raise SystemExit(f"[analyze] reference file not found: {ref}")
    dockq = run_dockq(model["cif"], ref, chain_map=args.dockq_mapping,
                      dockq_exe=args.dockq_exe,
                      allowed_mismatches=args.dockq_allowed_mismatches)
    if dockq and "best_dockq" in dockq:
        ifaces = dockq.get("interfaces") or {}
        ag_set = set(antigen_chains or ([antigen_chain] if antigen_chain else []))
        chain_ids = list(ag_set) + ([nanobody_chain] if nanobody_chain else [])
        nb_vals = []
        for key, v in ifaces.items():
            a, b = metrics.split_interface_key(key, chain_ids)
            pair = {a, b}
            if nanobody_chain and nanobody_chain in pair and (pair - {nanobody_chain}) <= ag_set:
                nb_vals.append(float(v.get("DockQ", 0.0)))
                v["capri"] = capri_class(v.get("DockQ", 0.0))
            else:
                v["capri"] = capri_class(v.get("DockQ", 0.0))
        if nb_vals:
            headline = float(np.mean(nb_vals))
            kind = f"나노바디-항원 인터페이스 {len(nb_vals)}개 평균"
        else:
            headline = float(dockq.get("global_dockq") or 0.0)
            kind = f"전체 인터페이스 {len(ifaces)}개 평균"
        dockq["headline"] = headline
        dockq["headline_kind"] = kind
        dockq["capri"] = capri_class(headline)
    return dockq


def interface_report_union(tokens, plddt, pae, antigen_chains, nanobody_chain, cutoff=8.0):
    """항원 체인 여러 개를 하나의 union 으로 취급한 인터페이스 분석 (H-05).

    - 리간드/항원 쪽 잔기 목록: 체인별 결과를 그대로 이어붙인다 (잔기 단위 보존)
    - 나노바디 쪽 잔기 목록: 같은 잔기가 여러 체인과 접촉하면 접촉 수를 합산
    """
    ag = [c for c in antigen_chains if c]
    if not ag:
        return {"n_contacts": 0, "cutoff": cutoff, "residues_a": [], "residues_b": []}

    reports = [metrics.interface_report(tokens, plddt, pae, ch, nanobody_chain, cutoff=cutoff)
               for ch in ag]

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    res_a, nb_counts = [], {}

    def _stable_residue_key(item):
        if item.get("token") is not None:
            return ("token", int(item["token"]))
        return (item.get("chain"), int(item.get("resnum")), item.get("icode", ""), item.get("aa"))

    for r in reports:
        res_a.extend(r.get("residues_a") or [])
        for item in (r.get("residues_b") or []):
            key = _stable_residue_key(item)
            if key in nb_counts:
                nb_counts[key]["n_contacts"] += item["n_contacts"]
            else:
                nb_counts[key] = dict(item)

    # M-01(정밀): PAE 평균은 '체인 평균의 평균'이 아니라 접촉 쌍 수로 가중한 평균이다.
    # 체인별 접촉 수가 크게 다를 때 단순 평균은 union 인터페이스
    # 전체 평균과 달라진다. pLDDT 는 union 잔기(중복 제거) 단위 평균으로 계산한다.
    total_contacts = int(sum((r.get("n_contacts") or 0) for r in reports))

    def _contact_weighted(field):
        num = sum((r.get(field) or 0.0) * (r.get("n_contacts") or 0) for r in reports)
        return float(num / total_contacts) if total_contacts else None

    return {
        "n_contacts": total_contacts,
        "cutoff": cutoff,
        "residues_a": sorted(res_a, key=lambda r: -r["n_contacts"]),
        "residues_b": sorted(nb_counts.values(), key=lambda r: -r["n_contacts"]),
        "pae_mean_ab": _contact_weighted("pae_mean_ab"),
        "pae_mean_ba": _contact_weighted("pae_mean_ba"),
        "plddt_mean_a": _mean([r["plddt"] for r in res_a]),
        "plddt_mean_b": _mean([r["plddt"] for r in nb_counts.values()]),
        "weighting": "contact-count(PAE) / residue(union, pLDDT)",
        "antigen_chains": ag,
        "per_chain_contacts": {ch: (r.get("n_contacts") or 0)
                               for ch, r in zip(ag, reports, strict=False)},
    }


def _ipsae_from_official(model, args, pae, chains_arr, pair, antigen_chains=None):
    """공식 ipSAE 실행 + 실패 시 내장 구현 폴백. R02: 폴백 상태를 결과에 보존하고
    byres 경로가 없으면 파싱하지 않는다."""
    ipsae = run_official_ipsae(args.ipsae_script, model["pae"], model["cif"],
                               pae_cutoff=args.pae_cutoff, dist_cutoff=args.dist_cutoff)
    maybe_fallback = bool(ipsae.get("error")) or not ipsae.get("pairs")
    if maybe_fallback and pae is not None and chains_arr is not None and pair and pair[0] and pair[1]:
        antigen_chain, nanobody_chain = pair
        try:
            val_ab, _ = metrics.ipsae_pure(pae, chains_arr, antigen_chain, nanobody_chain,
                                           cutoff=args.pae_cutoff)
            val_ba, _ = metrics.ipsae_pure(pae, chains_arr, nanobody_chain, antigen_chain,
                                           cutoff=args.pae_cutoff)
            ipsae["fallback"] = True
            ipsae["pairs"] = {
                (antigen_chain, nanobody_chain): {"chain1": antigen_chain, "chain2": nanobody_chain,
                                                  "type": "asym", "ipsae": val_ab},
                (nanobody_chain, antigen_chain): {"chain1": nanobody_chain, "chain2": antigen_chain,
                                                  "type": "asym", "ipsae": val_ba},
            }
            ipsae["max"] = {tuple(sorted((antigen_chain, nanobody_chain))): {
                "chain1": antigen_chain, "chain2": nanobody_chain, "type": "max",
                "ipsae": max(val_ab, val_ba)}}
        except Exception as exc:  # noqa: BLE001
            ipsae["fallback_error"] = str(exc)

    # H-05: 항원이 여러 체인이면 union 기준 ipSAE 도 함께 기록 (내장 구현)
    union_ipsae = None
    if (pae is not None and chains_arr is not None and pair and pair[0] and pair[1]
            and antigen_chains and len(antigen_chains) > 1 and nanobody_chain_of(pair)):
        nb = pair[1]
        ids = np.where(np.isin(chains_arr, list(antigen_chains)), "AG", chains_arr)
        try:
            v_ab, _ = metrics.ipsae_pure(pae, ids, "AG", nb, cutoff=args.pae_cutoff)
            v_ba, _ = metrics.ipsae_pure(pae, ids, nb, "AG", cutoff=args.pae_cutoff)
            union_ipsae = {"ipsae": max(v_ab, v_ba), "direction_max": max(v_ab, v_ba),
                           "nb_frame_ag": v_ab, "ag_frame_nb": v_ba,
                           "note": "항원 체인 union 기준 내장 구현"}
        except Exception:  # noqa: BLE001 - union 계산 실패는 치명적이지 않음
            union_ipsae = None

    out = {
        "union": union_ipsae,
        "asym_nb_frame_ag": {},
        "asym_ag_frame_nb": {},
        "max": {},
        "out_txt": ipsae.get("out_txt"),
        "byres": ipsae.get("byres") if isinstance(ipsae.get("byres"), str) else None,
        "pml": ipsae.get("pml"),
        "error": ipsae.get("error"),
        "fallback": bool(ipsae.get("fallback")),
        "fallback_error": ipsae.get("fallback_error"),
        "byres_rows": [],
        "source": "official",
    }
    if pair and pair[0] and pair[1]:
        antigen_chain, nanobody_chain = pair
        pairs = ipsae.get("pairs") or {}
        # 방향 규약: asym_X_frame_Y = "Y 기준 정렬에서 X 를 채점"
        out["asym_nb_frame_ag"] = pairs.get((antigen_chain, nanobody_chain)) or {}      # rows=항원, cols=나노바디
        out["asym_ag_frame_nb"] = pairs.get((nanobody_chain, antigen_chain)) or {}      # rows=나노바디, cols=항원
        maxes = ipsae.get("max") or {}
        sorted_key = tuple(sorted(pair))            # R04: 항상 정렬된 키로 조회
        out["max"] = maxes.get(sorted_key) or maxes.get(pair) or {}
        if out["fallback"]:
            out["source"] = "builtin-fallback"
    if out["byres"]:
        out["byres_rows"] = metrics.parse_ipsae_byres(out["byres"])
    return out


def nanobody_chain_of(pair):
    return pair[1] if pair and len(pair) > 1 else None


def write_overlay(model, ref, dockq, antigen_chain, out_dir):
    """참조 구조가 있을 때 예측(B)/참조(RA) 를 겹친 CIF 생성 (항원 체인 기준 정렬)."""
    overlay_dir = Path(out_dir) / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_cif = overlay_dir / f"{model['stem']}_overlay.cif"
    # M-03(정밀): DockQ(v2)의 best_mapping 은 {native_chain: model_chain} 이다.
    # (DockQ.py format_mapping: mapping = {nm: mm ...}). 모델 체인 이름으로 native 를
    # 찾으려면 방향을 뒤집어야 한다. 이름이 같은 복합체에서는 차이가 가려진다.
    native_to_model = (dockq or {}).get("best_mapping") or {}
    model_to_native = {m: n for n, m in native_to_model.items()}
    ref_chain = model_to_native.get(antigen_chain) or antigen_chain
    try:
        info = metrics.write_superposition(model["cif"], ref, out_cif,
                                          model_chain=antigen_chain,
                                          reference_chain=ref_chain)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return info


def chain_pair_table(conf, tokens, pae, chains_arr, d0, contact_cutoff=8.0):
    """모든 체인 쌍에 대해 ipTM(볼츠 값), 인터페이스 평균 PAE, 접촉 수를 계산.

    다중 체인 복합체 리포트용. 방향 규약은 볼츠와 동일하게
    pair_chains_iptm[scored][frame] 를 그대로 쓰고, 인터페이스 PAE 는
    frame 체인 기준 정렬에서 scored 체인 잔기의 PAE (pae[frame, scored]) 평균이다.
    """
    order = list(dict.fromkeys(chains_arr.tolist()))
    rows = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            pairs = metrics.contact_pairs(tokens, a, b, cutoff=contact_cutoff)
            pae_ab = float(np.mean([pae[x, y] for x, y, _ in pairs])) if pairs else None
            pae_ba = float(np.mean([pae[y, x] for x, y, _ in pairs])) if pairs else None
            rows.append({
                "chain_a": a, "chain_b": b,
                "iptm_boltz_a_scored_in_b_frame": _pair_iptm(conf, chains_arr, a, b),
                "iptm_boltz_b_scored_in_a_frame": _pair_iptm(conf, chains_arr, b, a),
                "iface_pae_a_frame_b_scored": pae_ab,
                "iface_pae_b_frame_a_scored": pae_ba,
                "n_contacts": len(pairs),
            })
    return rows


def _pair_iptm(conf, chains_arr, scored_chain, frame_chain):
    """Boltz pair_chains_iptm[scored][frame] (verified convention)."""
    pc = conf.get("pair_chains_iptm") or {}
    order = list(dict.fromkeys(chains_arr.tolist()))
    try:
        i = order.index(scored_chain)
        j = order.index(frame_chain)
        return float(pc[str(i)][str(j)])
    except (KeyError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze Boltz-2 nanobody predictions")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--restore-settings", type=Path, default=None,
                    help="replay analysis settings from a previous results.json/settings JSON")
    ap.add_argument("--boltz-dir", type=Path, default=None,
                    help="이번 실행에서 쓴 boltz_results_* 디렉터리 (다중 존재 시 선택 고정)")
    ap.add_argument("--yaml", type=Path, default=None, help="input YAML used for the run")
    ap.add_argument("--metadata", type=Path, default=None, help="sidecar metadata from prepare_input")
    ap.add_argument("--no-nanobody", action="store_true",
                    help="나노바디 없는 입력(타겟 단독/복합체)용: CDR/파라토프 분석 생략")
    ap.add_argument("--nanobody-chain", default=None)
    ap.add_argument("--antigen-chain", default=None)
    ap.add_argument("--antigen-chains", default=None,
                    help="항원 체인을 콤마로 명시 (예: A 또는 A,D). "
                         "Fv(VH+VL) 입력처럼 binder 외 다른 사슬(VL)이 있을 때 "
                         "VL 이 항원 인터페이스에 섞이지 않도록 지정")
    ap.add_argument("--reference", default=None, help="reference complex (cif/pdb) for DockQ")
    ap.add_argument("--dockq-mapping", default=None, help="e.g. AB:AB")
    ap.add_argument("--dockq-exe", default=None)
    ap.add_argument("--dockq-allowed-mismatches", type=int, default=0,
                    help="CDR 변이체처럼 참조와 서열이 다를 때 허용할 불일치 수 (예: 5)")
    ap.add_argument("--pae-cutoff", type=float, default=10.0)
    ap.add_argument("--dist-cutoff", type=float, default=15.0)
    ap.add_argument("--ipsae-script", default=str(Path(__file__).parent / "vendor" / "ipsae_official.py"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--figdir", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    explicit = _explicit_flags(sys.argv[1:])
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    apply_restored_settings(args, run_dir, explicit)
    pred_dir = find_predictions_dir(run_dir, args.boltz_dir)
    out_path = args.out or (run_dir / "analysis" / "results.json")
    figdir = args.figdir or (run_dir / "analysis" / "figures")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    metadata = None
    if args.metadata and args.metadata.exists():
        metadata = json.loads(args.metadata.read_text())

    run_params = (json.loads((run_dir / ".run_params.json").read_text())
                  if (run_dir / ".run_params.json").exists() else None)
    expected_samples = None
    if isinstance(run_params, dict) and run_params.get("samples") is not None:
        expected_samples = int(run_params["samples"])
    models = model_files(pred_dir, expected_samples=expected_samples)
    tokens = read_structure_tokens(models[0]["cif"])
    antigen_chain, nanobody_chain, role_source, antigen_chains = resolve_roles(tokens, metadata, args)
    roles = role_map(antigen_chains, nanobody_chain)
    chain_labels = {ch: ("antigen" if ch in set(antigen_chains) else
                         "nanobody" if ch == nanobody_chain else "")
                    for ch in chain_lengths(tokens)}

    # CDR annotation (nanobody sequence, computed once)
    nb_seq = sequence_of(tokens, nanobody_chain) if nanobody_chain else ""
    hmmer_dir = find_hmmer_dir(extra=[os.environ.get("HMMER_DIR", "")])
    if hmmer_dir and hmmer_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hmmer_dir + os.pathsep + os.environ.get("PATH", "")
    cdr_info = {"sequence": nb_seq, "available": False, "source": "ANARCI/IMGT"}
    ann = annotate_cdrs_imgt(nb_seq) if nb_seq else None
    if ann:
        nb_start = int(np.where(token_chain_array(tokens) == nanobody_chain)[0][0])
        ranges = {}
        for name, items in ann["cdrs"].items():
            if items:
                s = nb_start + items[0]["seq_index"]
                e = nb_start + items[-1]["seq_index"]
                ranges[name] = (s, e)
        cdr_info.update({
            "available": True, "cdrs": ann["cdrs"], "cdr_sequences": ann["cdr_sequences"],
            "hit": ann["hit"], "token_ranges": ranges,
        })
    else:
        cdr_info["note"] = ("--no-nanobody (타겟 단독/복합체) 입력이거나 "
                            "ANARCI/hmmscan 을 찾지 못해 CDR 주석을 건너뛰었습니다.")

    yaml_snapshot = _portable_snapshot(args.yaml, out_path.parent, "input_yaml")
    metadata_snapshot = _portable_snapshot(args.metadata, out_path.parent, "metadata")
    reference_snapshot = _portable_snapshot(args.reference, out_path.parent, "reference")
    if reference_snapshot:
        args.reference = reference_snapshot
    requested_msa = (run_params or {}).get("msa") if isinstance(run_params, dict) else None
    input_provenance = yaml_input_provenance(yaml_snapshot or args.yaml, requested_msa)
    analysis_provenance = runtime_state.analysis_provenance(ipsae_script=args.ipsae_script)

    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "validation_limitations": {
            "schema_version": 1,
            "not_validated": [
                "experimental binding affinity",
                "experimental positive/negative controls",
                "score calibration across unrelated targets or input schemas",
            ],
            "policy": "unsupported validation claims are declared as limitations, not emitted as statistics",
        },
        "antigen_chains": list(antigen_chains),
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "predictions_dir": str(pred_dir),
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "chains": [
            {"id": ch, "role": roles.get(ch, "other"), "n_res": n, "sequence": sequence_of(tokens, ch)}
            for ch, n in chain_lengths(tokens).items()
        ],
        "antigen_chain": antigen_chain,
        "nanobody_chain": nanobody_chain,
        "role_source": role_source,
        "chain_labels": chain_labels,
        "settings": {
            "analysis_provenance": analysis_provenance,
            "analysis_contract_version": 1,
            "run_dir": str(run_dir),
            "run_params": run_params,
            "boltz_dir": str(pred_dir.parent.parent),
            "predictions_dir": str(pred_dir),
            "yaml": str(args.yaml) if args.yaml else None,
            "metadata": str(args.metadata) if args.metadata else None,
            "yaml_snapshot": yaml_snapshot,
            "metadata_snapshot": metadata_snapshot,
            "no_nanobody": bool(args.no_nanobody),
            "nanobody_chain": nanobody_chain,
            "antigen_chain": antigen_chain,
            "antigen_chains": list(antigen_chains),
            "target_only": not bool(nanobody_chain),
            "no_figures": bool(args.no_figures),
            "expected_samples": expected_samples,
            "observed_samples": len(models),
            "effective_msa_policy": input_provenance,
            "input_constraints": {
                "schema_version": 1,
                "chain_ids": list(chain_lengths(tokens)),
                "chain_lengths": chain_lengths(tokens),
                "antigen_chain_ids": list(antigen_chains),
                "nanobody_chain_id": nanobody_chain,
                "target_only": not bool(nanobody_chain),
                "yaml_constraints_hash": input_provenance.get("constraints_hash"),
                "yaml_constraints_count": input_provenance.get("constraints_count"),
                "templates_modifications_properties": [
                    {
                        "ids": c.get("ids"),
                        "templates_hash": c.get("templates_hash"),
                        "modifications_hash": c.get("modifications_hash"),
                        "properties_hash": c.get("properties_hash"),
                        "cyclic": c.get("cyclic"),
                    }
                    for c in input_provenance.get("chains", [])
                ],
            },
            "ranking_policy": ("available pTM, confidence_score, mean_pLDDT, -index"
                               if not nanobody_chain else
                               "available primary_interface.ipTM, primary_interface.ipSAE, -index"),
            "pae_cutoff": args.pae_cutoff,
            "dist_cutoff": args.dist_cutoff,
            "contact_cutoff": 8.0,
            "reference": args.reference,
            "reference_snapshot": reference_snapshot,
            "dockq_mapping": args.dockq_mapping,
            "dockq_allowed_mismatches": args.dockq_allowed_mismatches,
            "role_source": role_source,
            "path_restore_policy": "absolute saved paths are rebased under the current run directory",
            "restore_settings_source": str(args.restore_settings) if args.restore_settings else None,
            "restore_warnings": list(getattr(args, "_restore_warnings", [])),
        },
        "cdr": cdr_info,
        "models": [],
        "warnings": [],
    }
    results["warnings"].extend(getattr(args, "_restore_warnings", []))
    for m in models:
        entry = analyze_model(m, tokens, (antigen_chain, nanobody_chain, antigen_chains),
                              args, cdr_info, out_path.parent, input_provenance=input_provenance)
        if not entry["token_dims_ok"]:
            results["warnings"].append(
                f"model {entry['index']}: token dims of PAE/pLDDT do not match the CIF token "
                "count; PAE-based numbers were skipped."
            )
        if not entry.get("token_identity_ok", True):
            results["warnings"].append(
                f"model {entry['index']}: ordered token identity differs from model 0; "
                "index-based CDR/pLDDT/figure outputs were skipped."
            )
        if not entry.get("pae_metrics_applicable", True):
            results["warnings"].append(
                f"model {entry['index']}: PAE-derived metrics skipped: "
                + "; ".join(entry.get("pae_metrics_ineligible_reasons") or
                            ["token dimension/identity mismatch"])
            )
        if (cdr_info.get("available") and cdr_info.get("token_ranges")
                and entry.get("pae_metrics_applicable", True)):
            # RR03: PAE/pLDDT 토큰 수가 CIF 와 다르면(ligand 등) CDR 구간 슬라이스가 어긋난다
            vals = {}
            for name, (s, e) in cdr_info["token_ranges"].items():
                plddt_model = np.load(m["plddt"])["plddt"].astype(float)
                plddt_model = plddt_model * 100.0 if plddt_model.max() <= 1.0 else plddt_model
                vals[name] = float(plddt_model[s:e + 1].mean())
            entry["cdr_mean_plddt"] = vals
        elif cdr_info.get("available"):
            entry["warnings"].append("토큰 정체성/PAE 분석 부적격으로 CDR 별 pLDDT 를 계산하지 않았습니다")
        results["models"].append(entry)

    # H-06(정밀): 대표 모델 선택 기준을 '화면에 표시하는 점수'와 일치시킨다.
    #   유효한 대표 ipTM, 대표 ipSAE, model index 순으로 정렬하고 결측치는 유효한 0보다 낮춘다.
    # M-04(정밀): target-only(나노바디 없음)에서는 인터페이스 점수가 부적용이므로
    #   구조 신뢰도(pTM, confidence_score, 평균 pLDDT)로 순위를 매긴다.
    target_only = not nanobody_chain

    def rank_key(entry):
        return model_rank_key(entry, target_only)
    results["models"].sort(key=rank_key, reverse=True)
    results["rank_key"] = (
        "available pTM, confidence_score, mean_pLDDT, -index" if target_only else
        "available primary_interface.ipTM, primary_interface.ipSAE, -index"
    )
    for e in results["models"]:
        e["rank_score"] = list(rank_key(e))
    results["best_model_index"] = results["models"][0]["index"]
    results["best_model_stem"] = results["models"][0]["stem"]
    if not target_only:
        best_primary = (results["models"][0].get("primary_interface")
                        or metrics.primary_interface_scores(results["models"][0]))
        missing = [name for name in ("iptm", "ipsae") if not _finite_rank_value(best_primary.get(name))[0]]
        if missing:
            results["warnings"].append(
                "selected model lacks finite primary interface score(s): " + ", ".join(missing)
            )

    # figures
    if not args.no_figures and figures is None:
        results["warnings"].append(
            "그림 모듈(matplotlib)을 불러오지 못해 그림을 생략했습니다 "
            f"({FIGURES_IMPORT_ERROR}). 지표·리포트는 계속 생성됩니다. "
            "설치하려면: bash setup.sh --reuse-env")
        print(f"[analyze] 경고: matplotlib 미설치로 그림 생략 ({FIGURES_IMPORT_ERROR})")
    if not args.no_figures and figures is not None:
        results.setdefault("figures", {})
        for entry in results["models"]:
            if not entry.get("pae_metrics_applicable", True):
                # 토큰 차원/정체성/입력 스키마가 부적격이면 index 기반 그림을 만들지 않는다
                entry["figures"] = {}
                continue
            tag = f"model_{entry['index']}"
            candidates = {
                "pae": figdir / f"pae_{tag}.png",
                "pae_iface": figdir / f"pae_iface_{tag}.png",
                "plddt": figdir / f"plddt_{tag}.png",
                "ipsae_byres": figdir / f"ipsae_byres_{tag}.png",
            }
            # R3-01/RR02: 이전 실행 그림을 '생성 전에' 지워, 이번 실행분만 등록되게 한다.
            # (생성 후에 지우면 방금 만든 그림이 사라진다)
            for cand in candidates.values():
                if cand.exists():
                    cand.unlink()
            pae = np.load(entry["pae_npz"])["pae"].astype(float)
            plddt_path = models[[x["index"] for x in models].index(entry["index"])]["plddt"]
            plddt_raw = np.load(plddt_path)["plddt"].astype(float)
            plddt = plddt_raw * 100.0 if plddt_raw.max() <= 1.0 else plddt_raw
            figures.pae_heatmap(pae, token_chain_array(tokens), candidates["pae"],
                                chain_labels=chain_labels)
            if nanobody_chain and len(antigen_chains) > 1:
                figures.pae_interface_union_heatmap(pae, tokens, antigen_chains,
                                                    nanobody_chain, candidates["pae_iface"],
                                                    chain_labels=chain_labels)
            elif nanobody_chain:
                figures.pae_interface_heatmap(pae, tokens, antigen_chain, nanobody_chain,
                                              candidates["pae_iface"], chain_labels=chain_labels)
            cdr_ranges = cdr_info.get("token_ranges") if cdr_info.get("available") else None
            figures.plddt_track(tokens, plddt, candidates["plddt"],
                                nanobody_chain=nanobody_chain, cdr_ranges=cdr_ranges,
                                chain_labels=chain_labels)
            if entry["ipsae"].get("byres_rows"):
                figures.ipsae_byres_track(entry["ipsae"]["byres_rows"], nanobody_chain,
                                          candidates["ipsae_byres"],
                                          cdr_ranges=cdr_ranges, labels=chain_labels)
            # 이번 실행에서 실제로 만들어진 그림만 기록
            entry["figures"] = {k: str(v) for k, v in candidates.items() if v.exists()}
        summary_rows = []
        for e in results["models"]:
            primary = e.get("primary_interface") or metrics.primary_interface_scores(e)
            mx = e["ipsae"].get("max") or {}
            summary_rows.append({
                "model": f"model_{e['index']}",
                "scope": primary.get("scope"),
                "iptm_nb": primary.get("iptm"),
                "ipsae": primary.get("ipsae"),
                "ipsae_d0chn": mx.get("ipsae_d0chn"),
                "pdockq2": mx.get("pdockq2"),
                "iface_plddt_nb": (e["interface_8A"] or {}).get("plddt_mean_b"),
            })
        comparison_path = figdir / "model_comparison.png"
        if comparison_path.exists():
            comparison_path.unlink()
        figures.model_comparison_chart(summary_rows, comparison_path)
        if comparison_path.exists():
            results["figures"]["model_comparison"] = str(comparison_path)

    out_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    print(f"[analyze] wrote {out_path}")
    print(f"[analyze] nanobody chain = {nanobody_chain} ({role_source}), antigen chain = {antigen_chain}")
    for e in results["models"]:
        primary = e.get("primary_interface") or metrics.primary_interface_scores(e)
        mx = e["ipsae"].get("max") or {}
        dqd = e.get("dockq") or {}
        # H-05: 리포트·표와 동일하게 대표값(headline, 인터페이스 평균)을 찍는다.
        # best_dockq 는 인터페이스별 '합'이라 표시값과 혼동을 준다.
        print(f"  model_{e['index']}: ipTM={primary.get('iptm')} "
              f"({primary.get('iptm_source')}), ipSAE={primary.get('ipsae')} "
              f"({primary.get('ipsae_source')}), pDockQ2={mx.get('pdockq2')}, "
              f"DockQ={dqd.get('headline', dqd.get('best_dockq'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
