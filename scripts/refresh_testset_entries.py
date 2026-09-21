#!/usr/bin/env python
"""기존 테스트셋 입력을 엔티티(SEQRES) 서열로 다시 만든다 (H-01 후속).

reference.cif 의 관측 서열과 원본 CIF 의 체인을 대조해 항원/바인더 체인을 찾고,
결측 잔기를 포함한 전체 서열(+수정 잔기)로 FASTA/YAML 을 재생성한다.
수정 잔기가 있는 복합체는 YAML 도 함께 저장한다.

  python scripts/refresh_testset_entries.py --dir examples/testset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_input as P
from fetch_complex_entry import AA1, best_chain_sequence

AA3 = set(AA1.keys())


def observed_seq(chain):
    return "".join(AA1.get(r.name.upper(), "") for r in chain.get_polymer() if r.name.upper() in AA1)


def find_original_chain(st, observed, exclude=()):
    for ch in st[0]:
        if ch.name in exclude:
            continue
        obs = observed_seq(ch)
        if not obs:
            continue
        if obs in observed or observed in obs:
            return ch.name
    return None


def write_fasta(path: Path, header: str, seq: str):
    with path.open("w") as fh:
        fh.write(f">{header}\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i + 60] + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, action="append", required=True)
    ap.add_argument("--cache", type=Path, default=None, help="CIF 캐시 위치 (기본 <dir>/_cif_cache)")
    args = ap.parse_args()

    for base in args.dir:
        cache = args.cache or (base / "_cif_cache")
        print(f"== {base} ==")
        for d in sorted(base.iterdir()):
            if not d.is_dir() or d.name.startswith("_") or not (d / "reference.cif").exists():
                continue
            pdb = d.name.split("_")[0]
            cif = cache / f"{pdb}.cif"
            if not cif.exists():
                print(f"  {d.name}: 원본 CIF 없음 ({cif}) - 건너뜀")
                continue
            ref = gemmi.read_structure(str(d / "reference.cif"))
            st = gemmi.read_structure(str(cif))
            st.setup_entities()
            ag_obs = observed_seq(ref[0]["A"]) if "A" in [c.name for c in ref[0]] else ""
            nb_obs = observed_seq(ref[0]["B"]) if "B" in [c.name for c in ref[0]] else ""
            # 역할 판정: 엔티티 서열에 ANARCI(VHH)가 성공하는 체인을 나노바디로 삼는다.
            # (Ig 도메인 항원과 VHH 를 구분해야 하며, 실패하면 기존 관측 서열 매칭으로 폴백)
            import metrics as _m
            vhh_cands = []
            for ch in st[0]:
                seq_i, _ = best_chain_sequence(st, ch.name)
                if seq_i and 90 <= len(seq_i) <= 160 and _m.annotate_cdrs(seq_i):
                    vhh_cands.append(ch.name)
            if len(vhh_cands) == 1:
                src_nb = vhh_cands[0]
                src_ag = next((c.name for c in st[0] if c.name != src_nb and observed_seq(c)), None)
            else:
                src_ag = find_original_chain(st, ag_obs)
                src_nb = find_original_chain(st, nb_obs, exclude=(src_ag,))
            if not src_ag or not src_nb:
                print(f"  {d.name}: 원본 체인 매칭 실패 (ag={src_ag}, nb={src_nb}, vhh_cands={vhh_cands})")
                continue
            ag_seq, ag_mods = best_chain_sequence(st, src_ag)
            nb_seq, nb_mods = best_chain_sequence(st, src_nb)
            old_ag = P.read_fasta(d / "target.fasta")[0]["sequence"] if (d / "target.fasta").exists() else ""
            old_nb = P.read_fasta(d / "nanobody.fasta")[0]["sequence"] if (d / "nanobody.fasta").exists() else ""
            write_fasta(d / "target.fasta", f"{pdb}_{src_ag} antigen (chain A)", ag_seq)
            write_fasta(d / "nanobody.fasta", f"{pdb}_{src_nb} nanobody(VHH) (chain B)", nb_seq)
            # 수정 잔기가 있으면 YAML 도 저장 (Boltz modifications 지원)
            lines = ["version: 1", "sequences:"]
            for cid, seq, mods in (("A", ag_seq, ag_mods), ("B", nb_seq, nb_mods)):
                lines += ["  - protein:", f"      id: {cid}", f"      sequence: {seq}"]
                if mods:
                    lines.append("      modifications:")
                    lines += [f"        - position: {p}\n          ccd: {c}" for p, c in mods]
            (d / f"{d.name}.yaml").write_text("\n".join(lines) + "\n")
            changed = (ag_seq != old_ag, nb_seq != old_nb)
            flag = "갱신" if any(changed) else "동일"
            print(f"  {d.name}: {flag} | 항원 {len(ag_seq)}aa(mods={ag_mods}) 나노바디 "
                  f"{len(nb_seq)}aa(mods={nb_mods}) | 변경(항원,나노바디)={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
