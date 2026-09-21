#!/usr/bin/env python
"""PDB에서 나노바디-항원 복합체 테스트셋 자동 구성.

PDBe search API 로 후보를 찾고, CIF 를 받아
  - 단백질 체인이 정확히 2개 (나노바디 + 항원)
  - 한 체인은 VHH (길이/모티프/ANARCI 로 확인)
인 항목만 골라 `<outdir>/<pdbid>/` 에 target.fasta, nanobody.fasta, reference.cif 를 만든다.
체인은 리포트 규칙에 맞춰 A=항원, B=나노바디로 정리한다.

  python scripts/fetch_nanobody_complexes.py --count 10 --outdir examples/testset
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics
from fetch_complex_entry import best_chain_sequence

SEARCH_URL = "https://www.ebi.ac.uk/pdbe/search/pdb/select"
CIF_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/{pdbid}.cif"
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}
VHH_MOTIFS = ("WGQGT", "WGQG", "WGKGT", "RGQGT", "WGQGTQVTVSS")


def http_get(url: str, timeout: int = 60, data: bytes | None = None,
             headers: dict | None = None, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"HTTP 실패 ({url}): {last}")


def search_candidates(rows: int = 400):
    q = ('nanobody AND number_of_polymer_entities:2 AND '
         'experimental_method:"X-ray diffraction"')
    url = f"{SEARCH_URL}?" + urllib.parse.urlencode({"q": q, "wt": "json", "rows": rows})
    raw = http_get(url)
    docs = json.loads(raw)["response"]["docs"]
    out = {}
    for d in docs:
        eid = d.get("pdb_id")
        res = d.get("resolution")
        if not eid or res is None or float(res) > 2.5:
            continue
        plen = d.get("polymer_length")
        if plen is None or not (95 <= int(plen) <= 150):
            continue
        out.setdefault(eid.lower(), {"resolution": float(res),
                                     "title": str(d.get("title", ""))[:80]})
    return out


def analyse_entry(pdbid: str, cif_path: Path):
    """단백질 체인 2개 + VHH 1개 확인 -> (antigen_chain, vhh_chain, info) 또는 None."""
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    chains = []
    for ch in st[0]:
        poly = ch.get_polymer()
        if len(poly) == 0:
            continue
        # H-01: 결측 잔기를 포함한 엔티티(SEQRES) 서열을 우선 사용한다
        seq, mods = best_chain_sequence(st, ch.name)
        if not seq:
            seq = "".join(AA3.get(r.name.upper(), "X") for r in poly)
        if len(seq) < 30 or seq.count("X") > 0.1 * len(seq):
            continue
        chains.append({"chain": ch.name, "seq": seq, "mods": mods})
    if len(chains) != 2:
        return None

    def vhh_score(entry):
        seq, n = entry["seq"], len(entry["seq"])
        score = 0
        if 95 <= n <= 150:
            score += 2
        if any(m in seq for m in VHH_MOTIFS):
            score += 2
        if seq.count("C") >= 2:
            score += 1
        if re.search(r"C[A-Z]{2,4}G[^C]{0,20}W", seq):
            score += 1
        return score

    s0, s1 = vhh_score(chains[0]), vhh_score(chains[1])
    if abs(s0 - s1) < 2:
        return None
    vhh = chains[0] if s0 > s1 else chains[1]
    antigen = chains[1] if s0 > s1 else chains[0]

    if not metrics.find_hmmer_dir(extra=[os.environ.get("HMMER_DIR", "")]):
        raise SystemExit("hmmscan 을 찾지 못했습니다. ANARCI 로 VHH 를 확인할 수 없습니다 "
                         "(설치: bash setup.sh --with-hmmer, 또는 HMMER_DIR 지정)")
    ann = metrics.annotate_cdrs(vhh["seq"])
    if not ann or not ann["cdrs"]["CDR3"]:
        return None
    cdr3 = ann["cdr_sequences"]["CDR3"]
    if antigen.get("mods") or vhh.get("mods"):
        print(f"  [경고] 수정 잔기 발견: 항원 {antigen.get('mods')}, 나노바디 {vhh.get('mods')} "
              f"-> FASTA 에는 모 잔기로 기록")
    return antigen, vhh, {"cdr3": cdr3, "vhh_len": len(vhh["seq"]),
                          "ag_len": len(antigen["seq"]),
                          "domain_hit": (ann.get("hit") or {}).get("id")}


def write_entry(outdir: Path, pdbid: str, cif_path: Path, antigen, vhh, info):
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    out = gemmi.Structure()
    out.name = pdbid
    out.spacegroup_hm = "P 1"
    out.cell = gemmi.UnitCell(300, 300, 300, 90, 90, 90)
    model = gemmi.Model("1")
    for new_name, src in (("A", antigen["chain"]), ("B", vhh["chain"])):
        ch = st[0][src].clone()
        ch.name = new_name
        keep = gemmi.Chain(new_name)
        for r in ch:
            if r.name.upper() in AA3:
                keep.add_residue(r.clone())
        model.add_chain(keep)
    out.add_model(model)
    out.setup_entities()

    d = outdir / pdbid
    d.mkdir(parents=True, exist_ok=True)
    out.make_mmcif_document().write_file(str(d / "reference.cif"))
    for fname, header, seq in (
        ("target.fasta", f"{pdbid}_{antigen['chain']} antigen", antigen["seq"]),
        ("nanobody.fasta", f"{pdbid}_{vhh['chain']} nanobody(VHH)", vhh["seq"]),
    ):
        with (d / fname).open("w") as fh:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--outdir", type=Path, default=Path("examples/testset"))
    ap.add_argument("--max-tries", type=int, default=60)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache = args.outdir / "_cif_cache"
    cache.mkdir(parents=True, exist_ok=True)

    print("[fetch] PDBe 검색 (nanobody, X-ray, 2 polymer entities, resolution<=2.5)")
    cands = search_candidates()
    print(f"[fetch] 후보 {len(cands)}개")

    rows = ["name\tantigen\tnanobody\treference\tnotes"]
    accepted = []
    for i, (pdbid, meta) in enumerate(sorted(cands.items(), key=lambda kv: kv[1]["resolution"])):
        if len(accepted) >= args.count or i >= args.max_tries:
            break
        cif = cache / f"{pdbid}.cif"
        try:
            if not cif.exists() or cif.stat().st_size == 0:
                raw = http_get(CIF_URL.format(pdbid=pdbid), timeout=120)
                cif.write_bytes(raw)
            got = analyse_entry(pdbid, cif)
        except Exception as exc:  # noqa: BLE001
            print(f"  - {pdbid}: 건너뜀 ({exc})")
            continue
        if not got:
            print(f"  - {pdbid}: 조건 불충족 (체인 수/VHH 판정)")
            continue
        antigen, vhh, info = got
        write_entry(args.outdir, pdbid, cif, antigen, vhh, info)
        name = f"{pdbid}_wt"
        rows.append(f"{name}\t{(args.outdir / pdbid / 'target.fasta').as_posix()}\t"
                    f"{(args.outdir / pdbid / 'nanobody.fasta').as_posix()}\t"
                    f"{(args.outdir / pdbid / 'reference.cif').as_posix()}\t"
                    f"{meta['title'][:60]} ({meta['resolution']:.2f} A)")
        accepted.append((pdbid, meta, info))
        print(f"  + {pdbid}: 항원 {info['ag_len']} aa, 나노바디 {info['vhh_len']} aa, "
              f"CDR3 {len(info['cdr3'])} aa, {meta['resolution']:.2f} A | {meta['title'][:50]}")
        time.sleep(0.3)

    tsv = args.outdir / "testset.tsv"
    tsv.write_text("\n".join(rows) + "\n")
    print(f"\n[fetch] {len(accepted)}개 확보 -> {tsv}")
    if len(accepted) < args.count:
        print(f"[fetch] 주의: {args.count}개에 못 미침 (조건에 맞는 구조가 부족)")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
