"""Metric computation helpers for Boltz-2 nanobody/antigen predictions.

All heavy-lifting interface scores (ipSAE, pDockQ, pDockQ2, LIS, ipTM_d0chn) are
taken from the official Dunbrack lab implementation
(scripts/vendor/ipsae_official.py, MIT license, https://github.com/DunbrackLab/IPSAE).
This module adds:
  - parsing of the official output files
  - contact / interface analysis from the predicted mmCIF
  - per-chain and per-region pLDDT statistics
  - an independent (fallback) ipSAE implementation
  - a DockQ wrapper
  - antibody/nanobody CDR annotation through ANARCI (optional)
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}

# CDR ranges (inclusive) per numbering scheme.
# IMGT: Lefranc; Kabat/Chothia/Martin: as used by ANARCI (see abysis/ANARCI docs).
CDR_RANGES_BY_SCHEME = {
    "imgt": {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)},
    "kabat": {"CDR1": (31, 35), "CDR2": (50, 65), "CDR3": (95, 102)},
    "chothia": {"CDR1": (26, 32), "CDR2": (52, 56), "CDR3": (95, 102)},
    # AbM(=martin) 정의: CDR1 26-35, CDR2 50-58, CDR3 95-102 (ANARCI abysis 정의와 동일)
    "martin": {"CDR1": (26, 35), "CDR2": (50, 58), "CDR3": (95, 102)},
}

# 참고: AHo/wolfguy 등은 경계 정의를 검증하지 않아 넣지 않았다 (ANARCI scheme 은 지원하지만
# CDR 범위 상수를 임의로 쓰면 framework 잔기가 섞여 들어온다).
IMGT_CDR_RANGES = CDR_RANGES_BY_SCHEME["imgt"]


# ---------------------------------------------------------------------------
# CIF parsing
# ---------------------------------------------------------------------------

def read_structure_tokens(cif_path):
    """Return ordered (token) residue records from a Boltz mmCIF file.

    Each record: {"chain", "resnum", "icode", "resname", "aa", "ca", "cb", "plddt"}
    The order matches Boltz's token order for protein-only inputs.
    """
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    model = st[0]
    tokens = []
    for chain in model:
        for res in chain:
            info = gemmi.find_tabulated_residue(res.name)
            if not info.is_amino_acid():
                continue
            name3 = res.name.upper()
            ca = res.find_atom("CA", "*")
            cb = res.find_atom("CB", "*")
            if cb is None:
                cb = ca
            bfac = np.array([a.b_iso for a in res], dtype=float)
            n_atoms = len(list(res))
            tokens.append(
                {
                    "chain": chain.name,
                    "resnum": res.seqid.num,
                    "icode": res.seqid.icode.strip(),
                    "resname": name3,
                    "aa": AA3_TO_1.get(name3, "X"),
                    "ca": np.array([ca.pos.x, ca.pos.y, ca.pos.z]) if ca is not None else None,
                    "cb": np.array([cb.pos.x, cb.pos.y, cb.pos.z]) if cb is not None else None,
                    "plddt": float(bfac.mean()) if bfac.size else float("nan"),
                    "n_atoms": int(n_atoms),
                }
            )
    return tokens


def chain_lengths(tokens):
    """Ordered dict chain -> number of tokens."""
    out = {}
    for t in tokens:
        out[t["chain"]] = out.get(t["chain"], 0) + 1
    return out


def token_chain_array(tokens):
    return np.array([t["chain"] for t in tokens])


def sequence_of(tokens, chain):
    return "".join(t["aa"] for t in tokens if t["chain"] == chain)


# ---------------------------------------------------------------------------
# PAE / pLDDT based metrics
# ---------------------------------------------------------------------------

def tm_func(d, d0):
    return 1.0 / (1.0 + (np.asarray(d, dtype=float) / float(d0)) ** 2)


def calc_d0(length, min_value=1.0):
    """d0 from Yang & Skolnick (2004); same as ipsae.py calc_d0."""
    length = float(length)
    d0 = 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8 if length > 27 else 1.0
    return max(min_value, d0)


def calc_d0_array(lengths, min_value=1.0):
    """Vectorised d0; same as ipsae.py calc_d0_array (length clipped at 26)."""
    lengths = np.maximum(26, np.asarray(lengths, dtype=float))
    return np.maximum(min_value, 1.24 * (lengths - 15) ** (1.0 / 3.0) - 1.8)


def interchain_iptm(pae, rows, cols, d0):
    """ipTM for one direction.

    PAE 규약(볼츠/AF 동일): pae[i][j] = i 를 기준으로 정렬했을 때 j 의 예상 오차.
    여기서는 rows=기준(frame) 체인, cols=평가(scored) 체인 원자로 계산한다:
    즉 "rows 체인 좌표계에서 본 cols 체인의 배치 신뢰도".
    """
    if len(rows) == 0 or len(cols) == 0:
        return 0.0, None
    block = pae[np.ix_(rows, cols)]
    per_row = tm_func(block, d0).mean(axis=1)
    idx = int(np.argmax(per_row))
    return float(per_row[idx]), int(rows[idx])


def ipsae_pure(pae, chain_ids, chain_a, chain_b, cutoff=10.0):
    """Independent re-implementation of the official ipSAE (fallback only).

    Mirrors DunbrackLab/IPSAE ipsae.py v4:
      ipSAE(A->B) = max_{i in A} mean_{j in B, PAE_ij < cutoff} 1/(1+(PAE_ij/d0_i)^2)
      d0_i from |V_i| (number of j in B with PAE_ij < cutoff),
      clipped: max(1.0, 1.24*(max(26,|V_i|)-15)^(1/3)-1.8)

    PAE convention: rows = aligned residue (frame), columns = scored residue.
    Returns (max value, per-residue array for chain A rows).
    """
    chain_ids = np.asarray(chain_ids)
    mask_a = chain_ids == chain_a
    mask_b = chain_ids == chain_b
    valid = np.outer(mask_a, mask_b) & (pae < cutoff)
    n0res = valid.sum(axis=1)
    d0res = calc_d0_array(n0res)
    byres = np.zeros(pae.shape[0])
    for i in np.where(mask_a)[0]:
        v = valid[i]
        if v.any():
            byres[i] = tm_func(pae[i], d0res[i])[v].mean()
    return float(byres[mask_a].max()), byres


def per_chain_plddt_stats(plddt, tokens):
    """Return {chain: {mean, median, min, max, n, frac_gt90, frac_gt70}}."""
    chains = token_chain_array(tokens)
    out = {}
    for ch in dict.fromkeys(chains):
        vals = plddt[chains == ch]
        if vals.size == 0:
            continue
        out[ch] = {
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "frac_gt90": float((vals > 90).mean()),
            "frac_gt70": float((vals > 70).mean()),
            "frac_lt50": float((vals < 50).mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Interface analysis
# ---------------------------------------------------------------------------

def contact_pairs(tokens, chain_a, chain_b, cutoff=8.0):
    """CB-CB (CA for Gly) contacts between two chains, cutoff in Angstrom.

    numpy 로 벡터화되어 있어 잔기 수가 큰 복합체에서도 빠르다.
    """
    idx_a = [i for i, t in enumerate(tokens) if t["chain"] == chain_a]
    idx_b = [i for i, t in enumerate(tokens) if t["chain"] == chain_b]
    if not idx_a or not idx_b:
        return []

    def _coord(pos_list):
        out = np.full((len(pos_list), 3), np.nan, dtype=np.float32)
        keep = []
        for k, i in enumerate(pos_list):
            c = tokens[i]["cb"] if tokens[i]["cb"] is not None else tokens[i]["ca"]
            if c is None:
                continue
            out[k] = c
            keep.append(k)
        return out, keep

    ca, keep_a = _coord(idx_a)
    cb, keep_b = _coord(idx_b)
    if not keep_a or not keep_b:
        return []
    ca, cb = ca[keep_a], cb[keep_b]
    ia = [idx_a[k] for k in keep_a]
    ib = [idx_b[k] for k in keep_b]
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    wi, wj = np.where(d <= cutoff)
    return [(ia[int(i)], ib[int(j)], float(d[i, j])) for i, j in zip(wi, wj, strict=True)]


def interface_report(tokens, plddt, pae, chain_a, chain_b, cutoff=8.0):
    """Interface residue tables + interface level PAE / pLDDT statistics."""
    pairs = contact_pairs(tokens, chain_a, chain_b, cutoff=cutoff)
    if not pairs:
        return {
            "n_contacts": 0, "cutoff": cutoff,
            "residues_a": [], "residues_b": [],
            "pae_mean_ab": None, "pae_mean_ba": None,
            "plddt_mean_a": None, "plddt_mean_b": None,
        }

    count_a, count_b = {}, {}
    pae_ab, pae_ba = [], []
    for i, j, _ in pairs:
        count_a[i] = count_a.get(i, 0) + 1
        count_b[j] = count_b.get(j, 0) + 1
        pae_ab.append(float(pae[i, j]))
        pae_ba.append(float(pae[j, i]))

    def table(counts):
        rows = []
        for idx, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            t = tokens[idx]
            rows.append(
                {
                    "token": int(idx),
                    "chain": t["chain"],
                    "resnum": int(t["resnum"]),
                    "icode": t.get("icode", ""),
                    "resname": t["resname"],
                    "aa": t["aa"],
                    "plddt": float(plddt[idx]),
                    "n_contacts": int(n),
                }
            )
        return rows

    res_a = table(count_a)
    res_b = table(count_b)
    return {
        "n_contacts": len(pairs),
        "cutoff": cutoff,
        "residues_a": res_a,
        "residues_b": res_b,
        "pae_mean_ab": float(np.mean(pae_ab)),
        "pae_mean_ba": float(np.mean(pae_ba)),
        "pae_median_ab": float(np.median(pae_ab)),
        "pae_median_ba": float(np.median(pae_ba)),
        # 평균 pLDDT 는 전달받은(해당 모델의) 잔기별 배열에서 계산한다.
        # tokens[*]["plddt"] 를 쓰면 모델 0 의 B-factor 가 섞인다.
        "plddt_mean_a": float(np.mean([plddt[i] for i in count_a])),
        "plddt_mean_b": float(np.mean([plddt[j] for j in count_b])),
    }


# ---------------------------------------------------------------------------
# Official ipSAE wrapper
# ---------------------------------------------------------------------------
# 로컬 패치(2026-09-18): vendored ipsae_official.py 는 confidence JSON 에
# pair_chains_iptm 키가 없을 때 {} 로 폴백을 선언한 뒤 곧바로 다시 그 키를 직접
# 조회해 KeyError 로 죽는다 (R16). 아래에서 그 부분만 방어적으로 바꾼다.
_IPSAE_PATCH_NEEDLE = "boltz_chain_pair_iptm_data=data_summary['pair_chains_iptm']"

def run_official_ipsae(ipsae_script, pae_npz, cif_path, pae_cutoff=10.0, dist_cutoff=15.0,
                       timeout=1800):
    """Run DunbrackLab ipsae.py on a Boltz prediction; parse the summary table.

    RR01: 임시 디렉터리에서 실행하고 성공했을 때만 실제 위치로 산출물을 복사한다.
    - timeout/실패 시 예외 대신 error dict 를 돌려준다 (호출부 fallback 이 동작하도록)
    - 실패해도 기존 txt/byres/pml 을 지우지 않는다
    """
    import shutil
    import tempfile

    pae_npz, cif_path = Path(pae_npz), Path(cif_path)
    # M-02(정밀): vendored ipsae 는 cutoff 를 int 로 절삭하고 10 미만이면 두 자리로
    # zero-padding 한다("09", "07"). 파일명도 같은 규칙으로 만들어야 비기본 cutoff
    # (예: 9/15, 7.5/15)에서 공식 결과를 읽고 fallback 으로 새지 않는다.
    def _cut(v):
        return ("0" + str(int(v))) if v < 10 else str(int(v))

    cut_tag = f"_{_cut(pae_cutoff)}_{_cut(dist_cutoff)}"
    real_stem = re.sub(r"\.(cif|pdb)$", "", str(cif_path)) + cut_tag

    # C-04: 보조 파일 경로는 '파일명'에서만 유도한다. 전체 경로에 replace 를 걸면
    # 상위 폴더 이름에 pae 가 있을 때 경로가 깨져 pLDDT=0 으로 조용히 오염된다.
    plddt_src = pae_npz.with_name(pae_npz.name.replace("pae", "plddt", 1))
    conf_src = pae_npz.with_name(pae_npz.name.replace("pae", "confidence", 1).replace(".npz", ".json"))

    missing = [p for p in (pae_npz, plddt_src, conf_src, cif_path) if not p.exists()]
    if missing:
        return {"error": "ipSAE 보조 파일 누락: " + ", ".join(p.name for p in missing),
                "pairs": {}, "max": {}}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # staging 은 경로 내용과 무관한 고정 이름으로 (Model 열도 안정적으로 남는다)
        shutil.copy2(pae_npz, tmp / "pae_m.npz")
        shutil.copy2(plddt_src, tmp / "plddt_m.npz")
        shutil.copy2(conf_src, tmp / "confidence_m.json")
        shutil.copy2(cif_path, tmp / "m.cif")
        tmp_pae, tmp_cif = tmp / "pae_m.npz", tmp / "m.cif"

        cmd = [sys.executable, str(Path(ipsae_script).resolve()), tmp_pae.name, tmp_cif.name,
               f"{pae_cutoff:g}", f"{dist_cutoff:g}"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(tmp),
                                  check=False)
        except subprocess.TimeoutExpired:
            return {"error": f"ipSAE 스크립트 시간 초과 ({timeout}s)", "pairs": {}, "max": {}}

        tmp_stem = re.sub(r"\.(cif|pdb)$", "", str(tmp_cif)) + cut_tag
        out_txt = Path(tmp_stem + ".txt")
        if proc.returncode != 0 or not out_txt.exists():
            return {"error": (proc.stderr or proc.stdout or "ipsae failed")[-2000:],
                    "pairs": {}, "max": {}}

        # 성공: 이번 실행이 실제로 만든 파일만 실제 위치로 교체 (R3-03)
        produced = {}
        for suffix in (".txt", "_byres.txt", ".pml"):
            src = Path(tmp_stem + suffix)
            if src.exists():
                shutil.copy2(src, Path(real_stem + suffix))
                produced[suffix] = str(real_stem + suffix)
        table_text = out_txt.read_text()

    header = None
    pairs, maxes = {}, {}
    for line in table_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Chn1":
            header = parts
            continue
        if header is None or len(parts) < 22:
            continue
        rec = {
            "chain1": parts[0], "chain2": parts[1],
            "pae_cutoff": float(parts[2]), "dist_cutoff": float(parts[3]),
            "type": parts[4],
            "ipsae": float(parts[5]), "ipsae_d0chn": float(parts[6]), "ipsae_d0dom": float(parts[7]),
            "iptm_af": float(parts[8]), "iptm_d0chn": float(parts[9]),
            "pdockq": float(parts[10]), "pdockq2": float(parts[11]), "lis": float(parts[12]),
            "n0res": int(parts[13]), "n0chn": int(parts[14]), "n0dom": int(parts[15]),
            "d0res": float(parts[16]), "d0chn": float(parts[17]), "d0dom": float(parts[18]),
            "nres1": int(parts[19]), "nres2": int(parts[20]),
            "dist1": int(parts[21]), "dist2": int(parts[22]) if len(parts) > 22 else None,
        }
        if rec["type"] == "asym":
            pairs[(rec["chain1"], rec["chain2"])] = rec
        else:
            maxes[(rec["chain1"], rec["chain2"])] = rec
    # 생성된 것만 경로를 노출한다 (없는 파일을 이전 결과에서 끌어오지 않도록)
    return {"pairs": pairs, "max": maxes,
            "out_txt": produced.get(".txt"),
            "pml": produced.get(".pml"),
            "byres": produced.get("_byres.txt")}


def parse_ipsae_byres(byres_path):
    """Parse *_byres.txt into a list of per-residue records.

    빈 문자열/None/디렉터리/없는 파일은 조용히 빈 목록을 돌려준다.
    (Path("") 는 현재 디렉터리라 exists()==True 가 되어 read_text() 에서 터진다 - R02)
    """
    rows = []
    if not byres_path or not isinstance(byres_path, (str, Path)):
        return rows
    p = Path(byres_path)
    if not p.is_file():
        return rows
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 16 or parts[0] in ("i", "#"):
            continue
        try:
            rows.append(
                {
                    "i": int(parts[0]), "align_chain": parts[1], "scored_chain": parts[2],
                    "resnum": int(parts[3]), "resname": parts[4], "plddt": float(parts[5]),
                    "n0res": int(parts[8]), "d0res": float(parts[11]),
                    "ptm_pae": float(parts[12]), "psae_d0chn": float(parts[13]),
                    "psae_d0dom": float(parts[14]), "psae_d0res": float(parts[15]),
                }
            )
        except (ValueError, IndexError):
            continue
    return rows


# ---------------------------------------------------------------------------
# DockQ
# ---------------------------------------------------------------------------

def find_executable(name, extra_dirs=()):
    exe = shutil.which(name)
    if exe:
        return exe
    dirs = [Path(sys.executable).resolve().parent]
    dirs += [Path(os.path.expanduser(d)) for d in extra_dirs if d]
    env_bin = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV")
    if env_bin:
        dirs.append(Path(env_bin) / "bin")
    for d in dirs:
        cand = d / name
        if cand.exists():
            return str(cand)
    return None


def run_dockq(model_cif, reference_cif, chain_map=None, dockq_exe=None, timeout=3600,
              allowed_mismatches=0):
    """Run DockQ (CLI) and return the parsed JSON results.

    allowed_mismatches: 모델 서열이 참조와 다를 때(예: CDR 변이체) 허용할 불일치 수.
    """
    if dockq_exe is None:
        dockq_exe = find_executable("DockQ") or find_executable("dockq")
    if dockq_exe is None:
        return {"error": "DockQ executable not found"}
    out_json = Path(model_cif).with_suffix("")
    out_json = Path(str(out_json) + "_dockq.json")
    if out_json.exists():
        out_json.unlink()  # 이전 실행 결과를 재사용하지 않도록 (실패 시 stale 값 방지)
    cmd = [dockq_exe, str(model_cif), str(reference_cif), "--json", str(out_json), "--n_cpu", "4"]
    if chain_map:
        cmd += ["--mapping", chain_map]
    if allowed_mismatches:
        cmd += ["--allowed_mismatches", str(int(allowed_mismatches))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"error": f"DockQ 시간 초과 ({timeout}s)"}
    if not out_json.exists():
        err = (proc.stderr or proc.stdout or "DockQ failed")[-2000:]
        if "identical corresponding chain" in err and not allowed_mismatches:
            err += ("\n[hint] 모델과 참조 서열이 다릅니다(CDR 변이체 등). "
                    "--dockq-mismatches 5 처럼 허용 불일치 수를 지정하세요.")
        return {"error": err}
    try:
        data = json.loads(out_json.read_text())
    except json.JSONDecodeError:
        return {"error": "could not parse DockQ JSON"}
    interfaces = data.get("best_result", {})
    data["interfaces"] = interfaces
    data.pop("best_result", None)
    if interfaces:
        vals = [float(v.get("DockQ", 0.0)) for v in interfaces.values()]
        data["n_interfaces"] = len(vals)
        # C-01: DockQ v2 의 best_dockq 는 '인터페이스별 합'이다. 평균은 GlobalDockQ.
        data["total_dockq"] = float(data.get("best_dockq") or sum(vals))
        data["global_dockq"] = float(data.get("GlobalDockQ") or (sum(vals) / len(vals)))
    if proc.returncode != 0:
        data["warning"] = (proc.stderr or "")[-500:]
    return data


def dockq_headline(d):
    """표시용 대표 DockQ: 인터페이스 평균(headline) 우선, 없으면 best_dockq."""
    if not d:
        return None
    return d.get("headline", d.get("best_dockq"))


def _finite_score(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def primary_interface_scores(model):
    """Return the headline nanobody/antigen interface scores for one model.

    Contract:
    - two-chain complexes keep the native Boltz directional pair ipTM and the
      official max ipSAE pair score.
    - multi-antigen-chain complexes use the antigen-union PAE ipTM and the
      antigen-union ipSAE score, so ranking, reports and plots describe the same
      interface.
    """
    antigen_chains = list(model.get("antigen_chains") or [])
    target_only = bool(model.get("target_only")) or not model.get("has_nanobody", True)
    if target_only:
        return {
            "scope": "target-only",
            "iptm": None,
            "ipsae": None,
            "iptm_label": "interface ipTM not applicable",
            "ipsae_label": "interface ipSAE not applicable",
            "iptm_source": "not-applicable",
            "ipsae_source": "not-applicable",
        }

    multichain = len(antigen_chains) > 1
    pae_iptm = (model.get("iptm_from_pae") or {}).get("nanobody_in_antigen_frame")
    pair_iptm = (model.get("boltz_pair_iptm") or {}).get("nanobody_in_antigen_frame")
    ipsae = model.get("ipsae") or {}
    pair_ipsae = (ipsae.get("max") or {}).get("ipsae")
    union_ipsae = (ipsae.get("union") or {}).get("ipsae")

    if multichain:
        iptm = _finite_score(pae_iptm)
        ipsae_score = _finite_score(union_ipsae)
        return {
            "scope": "antigen-union",
            "iptm": iptm,
            "ipsae": ipsae_score,
            "iptm_label": "ipTM (PAE antigen-union; nanobody scored in antigen frame)",
            "ipsae_label": "ipSAE (antigen-union)",
            "iptm_source": "pae_union_iptm",
            "ipsae_source": "builtin-union" if ipsae_score is not None else "missing",
        }

    iptm = _finite_score(pair_iptm)
    iptm_source = "boltz_pair_iptm"
    iptm_label = "ipTM (Boltz pair; nanobody scored in antigen frame)"
    if iptm is None:
        iptm = _finite_score(pae_iptm)
        iptm_source = "pae_pair_iptm" if iptm is not None else "missing"
        iptm_label = ("ipTM (PAE pair; nanobody scored in antigen frame)"
                      if iptm is not None else "ipTM unavailable")
    ipsae_score = _finite_score(pair_ipsae)
    src = ipsae.get("source") or "official"
    ipsae_label = (
        f"ipSAE ({'built-in fallback' if src == 'builtin-fallback' else src} max pair)"
        if ipsae_score is not None else "ipSAE unavailable"
    )
    return {
        "scope": "pair",
        "iptm": iptm,
        "ipsae": ipsae_score,
        "iptm_label": iptm_label,
        "ipsae_label": ipsae_label,
        "iptm_source": iptm_source,
        "ipsae_source": f"{src}:max" if ipsae_score is not None else "missing",
    }


def split_interface_key(key, chain_ids):
    """DockQ 인터페이스 키('AB')를 (chainA, chainB) 로 나눈다 (단일문자 체인 가정, 폴백 포함)."""
    key = "".join(str(key))
    ordered = sorted(set(chain_ids), key=len, reverse=True)
    for a in ordered:
        if key.startswith(a) and key[len(a):] in chain_ids:
            return a, key[len(a):]
    if len(key) == 2:
        return key[0], key[1]
    return key, ""


def _protein_chain(structure, chain_name):
    for ch in structure[0]:
        if ch.name == chain_name:
            return ch
    return None


def write_superposition(model_cif, reference_cif, out_cif, model_chain, reference_chain,
                        timeout=None):
    """예측 구조를 참조 구조에 겹쳐서(항원 체인 기준 정렬) 하나의 CIF 로 저장.

    모델 체인은 그대로(A, B), 참조 체인은 앞에 'R' 을 붙인다(RA, RB).
    정렬은 model_chain(모델) <-> reference_chain(참조) 의 CA 원자로 계산하고,
    변환은 모델 전체(모든 체인)에 적용한다.
    """
    import gemmi

    model = gemmi.read_structure(str(model_cif))
    ref = gemmi.read_structure(str(reference_cif))
    m_align = _protein_chain(model, model_chain)
    r_align = _protein_chain(ref, reference_chain)
    if m_align is None or r_align is None:
        return None
    res = gemmi.calculate_superposition(r_align.get_polymer(), m_align.get_polymer(),
                                        gemmi.PolymerType.PeptideL, gemmi.SupSelect.CaP)
    if res.count == 0:
        return None
    tr = res.transform

    out = gemmi.Structure()
    out.name = "boltz_overlay"
    out.spacegroup_hm = "P 1"
    out.cell = ref.cell if ref.cell.is_crystal() else gemmi.UnitCell(200, 200, 200, 90, 90, 90)
    new_model = gemmi.Model("1")

    for ch in ref[0]:
        info = gemmi.find_tabulated_residue(ch[0].name) if len(ch) else None
        if info is None or not info.is_amino_acid():
            continue
        c = ch.clone()
        c.name = ("R" + ch.name)[:4]
        new_model.add_chain(c)

    for ch in model[0]:
        if len(ch) == 0:
            continue
        info = gemmi.find_tabulated_residue(ch[0].name)
        if info is None or not info.is_amino_acid():
            continue
        c = ch.clone()
        for r in c:
            for a in r:
                v = tr.apply(gemmi.Vec3(a.pos.x, a.pos.y, a.pos.z))
                a.pos = gemmi.Position(v.x, v.y, v.z)
        new_model.add_chain(c)

    out.add_model(new_model)
    out.setup_entities()
    out.make_mmcif_document().write_file(str(out_cif))
    return {"rmsd": float(res.rmsd), "n_ca": int(res.count),
            "model_chain": model_chain, "reference_chain": reference_chain,
            "path": str(out_cif), "reference": str(reference_cif)}


def capri_class(dockq):
    if dockq is None:
        return None
    if dockq >= 0.80:
        return "High"
    if dockq >= 0.49:
        return "Medium"
    if dockq >= 0.23:
        return "Acceptable"
    return "Incorrect"


# ---------------------------------------------------------------------------
# CDR annotation (ANARCI, optional)
# ---------------------------------------------------------------------------

def annotate_cdrs_imgt(sequence, scheme="imgt", chain_type="H"):
    """Backward-compatible alias: annotate CDRs with the given numbering scheme."""
    return annotate_cdrs(sequence, scheme=scheme, chain_type=chain_type)


def annotate_cdrs(sequence, scheme="imgt", chain_type="H"):
    """Return {CDR1/2/3: [{resnum_in_seq, aa, imgt_pos}], ...} using ANARCI.

    Returns None if ANARCI (or hmmscan) is unavailable or numbering fails.
    """
    try:
        from anarci import run_anarci
    except Exception:  # noqa: BLE001 - ANARCI 미설치/로딩 실패는 정상적인 미지원 경로
        return None
    if find_executable("hmmscan") is None:
        return None
    try:
        result = run_anarci([("query", sequence)], scheme=scheme, allowed_species=None)
    except Exception:  # noqa: BLE001 - ANARCI 내부 실패 시 CDR 주석 생략
        return None
    if not result or len(result) < 3 or not result[1] or not result[1][0]:
        return None
    hits = result[2][0] if result[2] else []
    hit = hits[0] if hits else None

    if chain_type not in ("H", "K", "L"):
        raise ValueError(f"chain_type 은 H/K/L 중 하나여야 합니다: {chain_type}")
    ranges = CDR_RANGES_BY_SCHEME.get(scheme)
    if ranges is None:
        ranges = IMGT_CDR_RANGES
    out = {k: [] for k in ranges}
    mismatches = 0
    for _domain in result[1][0]:
        # C-02: ANARCI 도메인 튜플은 (numbered_items, start, end) 이다.
        # start(입력 서열에서의 도메인 시작 인덱스)를 무시하면 태그/비정렬 N-말단이 있는
        # 서열에서 CDR 위치가 통째로 밀린다.
        if isinstance(_domain, tuple):
            items, start = _domain[0], int(_domain[1])
        else:
            items, start = _domain, 0
        seq_index = start - 1
        for (pos, ins), aa in items:
            if aa == "-":
                continue
            seq_index += 1
            if not (0 <= seq_index < len(sequence)) or sequence[seq_index].upper() != aa.upper():
                mismatches += 1
                continue
            for name, (lo, hi) in ranges.items():
                if lo <= pos <= hi:
                    out[name].append({"seq_index": seq_index, "aa": aa,
                                      "imgt_pos": f"{pos}{ins}".strip()})
                    break
    total = sum(len(v) for v in out.values())
    if total and mismatches > total:
        return None   # 매핑이 어긋난 경우 잘못된 CDR 을 반환하지 않는다
    return {
        "scheme": scheme,
        "start_offset": start,
        "mismatches": mismatches,
        "hit": {"id": hit.get("id"), "evalue": hit.get("evalue"), "bitscore": hit.get("bitscore")} if hit else None,
        "cdrs": dict(out),
        "cdr_sequences": {k: "".join(x["aa"] for x in v) for k, v in out.items()},
    }


def find_hmmer_dir(extra=()):
    """Locate a directory that contains the hmmscan binary (for ANARCI)."""
    for d in extra:
        if d and (Path(d) / "hmmscan").exists():
            return str(d)
    exe = shutil.which("hmmscan")
    if exe:
        return str(Path(exe).parent)
    import glob

    for pat in ("~/miniforge3/envs/*/bin", "~/miniconda3/envs/*/bin", "/opt/conda/envs/*/bin",
                "/usr/local/bin", "/usr/bin"):
        for d in glob.glob(os.path.expanduser(pat)):
            if (Path(d) / "hmmscan").exists():
                return d
    return None


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

def json_safe(obj):
    """Recursively convert numpy types to plain python for JSON serialisation."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj
