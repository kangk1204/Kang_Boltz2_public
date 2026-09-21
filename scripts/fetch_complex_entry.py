#!/usr/bin/env python
"""PDB 구조에서 타겟(들) + 바인더 체인을 뽑아 이 파이프라인 입력으로 정리한다.

  python scripts/fetch_complex_entry.py --pdb 1mel --binder-chain B \
      --antigen-chains A --name complex_demo --outdir examples/complexes

- 항원 체인들은 A, C, D, ... 로, 바인더는 B 로 이름을 바꿔 reference.cif 를 만든다.
- target.fasta (항원, 여러 레코드), nanobody.fasta (바인더 1개) 를 쓴다.
- `--target-only` 이면 항원 서열만 있는 YAML 을 만든다 (바인더 없음).
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gemmi

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
       "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE"}
AA1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
# H-01: 수정 잔기는 모(parent) 잔기로 매핑하고 Boltz modifications 로 따로 전달한다
MOD_PARENT = {"MSE": "M", "PTR": "Y", "SEP": "S", "TPO": "T", "CSO": "C", "KCX": "K", "MLY": "K",
              "HYP": "P", "PCA": "E", "FME": "M", "CSD": "C", "CME": "C", "SME": "M", "OCS": "C",
              "CAS": "C", "ALY": "K", "M3L": "K", "NEP": "H", "HIC": "H"}
# C-12: 25개 초과 시 IndexError 가 나던 것을 방지 (A-Z 에서 나노바디용 1개는 비워 둠)
ANTIGEN_NAMES = list("ACDEFGHIJKLMNOPQRSTUVWXYZ")[:24]


def _download(url: str, dst: Path, timeout: int = 180, retries: int = 3) -> None:
    """C-10/C-11: 원자적 다운로드 + 내용 검증 + 재시도."""
    import time

    last = None
    for attempt in range(1, retries + 1):
        tmp = dst.with_suffix(dst.suffix + f".part{os.getpid()}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "boltz2-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data or data.lstrip().startswith(b"<"):
                raise RuntimeError(f"응답이 비어 있거나 HTML 입니다 ({len(data)} bytes)")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            return
        except Exception as e:  # noqa: BLE001 - 네트워크/포맷/IO 오류를 한 번에 재시도한다
            last = e
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)
    raise SystemExit(f"다운로드 실패: {url} ({last})")


def fetch_cif(pdb: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{pdb.lower()}.cif"
    if not f.exists() or f.stat().st_size == 0:
        url = f"https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb.lower()}.cif"
        print(f"[fetch] {url}")
        _download(url, f)
        try:
            gemmi.read_structure(str(f))
        except Exception as e:
            f.unlink(missing_ok=True)
            raise SystemExit(f"내려받은 CIF 를 해석할 수 없습니다: {pdb} ({e})") from e
    return f


def chain_seq(structure, name):
    """관측 잔기 기반 서열 (수정 잔기는 X)."""
    return "".join(gemmi.find_tabulated_residue(r.name).one_letter_code
                   for r in structure[0][name].get_polymer() if r.name in AA3)


def _entity_seq_mods(ent):
    seq, mods = "", []
    for i, res in enumerate(ent.full_sequence, 1):
        if res in AA1:
            seq += AA1[res]
        elif res in MOD_PARENT:
            seq += MOD_PARENT[res]
            mods.append((i, res))
        elif res == "UNK":
            seq += "X"
        else:
            seq += "X"
    return seq, mods


def entity_sequence(structure, chain_name):
    """H-01(정밀): 엔티티(SEQRES) 전체 서열과 수정 잔기 목록.

    반환: (sequence, modifications) — modifications = [(1-based position, ccd code), ...]
    SEQRES 가 없으면 (None, []).

    auth 체인 이름(`_atom_site.auth_asym_id`)과 label subchain 식별자
    (`_struct_asym.id`)는 서로 다른 이름 공간이다. 예를 들어 8qf4 는
    auth `A`(=Arc 항원)의 label subchain 이 `B` 이고, auth `E`(=나노바디)의
    label subchain 이 `A` 다. 엔티티의 `subchains`(label)와 auth 이름을 직접
    비교하면 다른 분자의 서열을 집어 온다. 따라서 체인의 ResidueSpan 을
    `get_entity_of` 로 해석해 폴리머 엔티티를 찾는다.
    """
    names = [c.name for c in structure[0]]
    if chain_name not in names:
        return None, []
    chain = structure[0][chain_name]
    get_entity_of = getattr(structure, "get_entity_of", None)
    if get_entity_of is not None:
        try:
            spans = list(chain.subchains())
        except Exception:  # noqa: BLE001 - 구버전/비정상 체인은 폴백으로 넘긴다
            spans = []
        polymer = []
        for span in spans:
            try:
                ent = get_entity_of(span)
            except Exception:  # noqa: BLE001
                ent = None
            if ent is not None and ent.full_sequence:
                polymer.append(ent)
        if polymer:
            return _entity_seq_mods(polymer[0])
        # 폴리머 엔티티를 찾지 못하면(관측 서열만 있는 경우) 폴백으로 넘긴다
        return None, []
    # 아주 오래된 gemmi 폴백: label subchain 이 auth 이름과 같은 경우에만 유효
    for ent in structure.entities:
        if chain_name in (ent.subchains or []) and ent.full_sequence:
            return _entity_seq_mods(ent)
    return None, []


def best_chain_sequence(structure, chain_name):
    """엔티티(SEQRES) 전체 서열(없으면 관측 서열)과 수정 잔기를 돌려준다.

    정책: 발현/정제 태그를 포함한 엔티티 전체 서열을 쓴다. ANARCI 는 C-말단
    His-tag 등이 있어도 CDR 을 정상 검출하므로(4n9o 등 기존 입력과 일관) 태그를
    임의로 제거하지 않는다. 엔티티가 없을 때만 관측 서열로 폴백한다.
    """
    ent_seq, mods = entity_sequence(structure, chain_name)
    if ent_seq:
        return ent_seq, mods
    return chain_seq(structure, chain_name), []


def clean_chain(src_chain, new_name):
    ch = gemmi.Chain(new_name)
    for r in src_chain:
        if r.name in AA3:
            ch.add_residue(r.clone())
    return ch


def write_fasta(path: Path, header: str, seq: str):
    with path.open("w") as fh:
        fh.write(f">{header}\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i + 60] + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--binder-chain", default=None, help="두 번째 체인(바인더/쿼리) - 없으면 target-only")
    ap.add_argument("--antigen-chains", required=True, help="콤마 구분 (예: A,B,D)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", type=Path, default=Path("examples/complexes"))
    ap.add_argument("--cache-dir", type=Path, default=Path("examples/complexes/_cif_cache"))
    args = ap.parse_args()

    cif = fetch_cif(args.pdb, args.cache_dir)
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    src_chains = [c.strip() for c in args.antigen_chains.split(",") if c.strip()]

    d = args.outdir / args.name
    d.mkdir(parents=True, exist_ok=True)

    if args.binder_chain is None:
        # target-only: 항원 서열 FASTA + 단일/다중 체인 YAML
        lines = ["version: 1", "sequences:"]
        for i, ch in enumerate(src_chains):
            seq, mods = best_chain_sequence(st, ch)
            if not seq:
                raise SystemExit(f"chain {ch} 에 단백질이 없습니다")
            cid = ANTIGEN_NAMES[i]
            write_fasta(d / f"chain_{cid}.fasta", f"{args.pdb}_{ch} antigen", seq)
            lines += ["  - protein:", f"      id: {cid}", f"      sequence: {seq}"]
            if mods:
                lines.append("      modifications:")
                lines += [f"        - position: {p}\n          ccd: {c}" for p, c in mods]
        (d / f"{args.name}.yaml").write_text("\n".join(lines) + "\n")
        # canonical target.fasta (multi-record) - 체인 수와 무관하게 항상 작성 (R12)
        # H-02(정밀): 관측 서열(chain_seq)이 아니라 엔티티(SEQRES) 전체 서열을 쓴다.
        with (d / "target.fasta").open("w") as fh:
            for i, ch in enumerate(src_chains):
                seq, _mods = best_chain_sequence(st, ch)
                fh.write(f">{args.pdb}_{ch} antigen (chain {ANTIGEN_NAMES[i]})\n")
                for j in range(0, len(seq), 60):
                    fh.write(seq[j:j + 60] + "\n")
        print(f"[fetch] target-only -> {d} (chains: {[ANTIGEN_NAMES[i] for i in range(len(src_chains))]})")
        return 0

    # 항원 + 바인더: reference.cif (항원 A,C,D..., 바인더 B)
    binder_seq, binder_mods = best_chain_sequence(st, args.binder_chain)
    if not binder_seq:
        raise SystemExit(f"binder chain {args.binder_chain} 에 단백질이 없습니다")
    if binder_mods:
        print(f"[fetch] 경고: 바인더 {args.binder_chain} 에 수정 잔기 {binder_mods} "
              f"(FASTA 에는 모 잔기로 표기; 정확히 쓰려면 YAML modifications 사용)")

    out = gemmi.Structure()
    out.name = f"{args.pdb}_{args.name}"
    out.spacegroup_hm = "P 1"
    out.cell = gemmi.UnitCell(400, 400, 400, 90, 90, 90)
    model = gemmi.Model("1")
    mapping = {"B": args.binder_chain}
    for i, ch in enumerate(src_chains):
        cid = ANTIGEN_NAMES[i]
        model.add_chain(clean_chain(st[0][ch], cid))
        mapping[cid] = ch
    model.add_chain(clean_chain(st[0][args.binder_chain], "B"))
    out.add_model(model)
    out.setup_entities()
    out.make_mmcif_document().write_file(str(d / "reference.cif"))

    for i, ch in enumerate(src_chains):
        seq_i, mods_i = best_chain_sequence(st, ch)
        write_fasta(d / f"antigen_{ANTIGEN_NAMES[i]}.fasta", f"{args.pdb}_{ch} antigen", seq_i)
        if mods_i:
            print(f"[fetch] 참고: 항원 {ch} 수정 잔기 {mods_i} -> YAML 로 넣으려면 "
                  f"{d / (args.name + '.yaml')} 참고")
    write_fasta(d / "nanobody.fasta", f"{args.pdb}_{args.binder_chain} binder", binder_seq)
    # canonical target.fasta (multi-record) (R12)
    # H-02(정밀): 관측 서열이 아니라 엔티티(SEQRES) 전체 서열을 쓴다.
    with (d / "target.fasta").open("w") as fh:
        for i, ch in enumerate(src_chains):
            seq, _mods = best_chain_sequence(st, ch)
            fh.write(f">{args.pdb}_{ch} antigen (chain {ANTIGEN_NAMES[i]})\n")
            for j in range(0, len(seq), 60):
                fh.write(seq[j:j + 60] + "\n")

    print(f"[fetch] {args.name}: 항원 {[f'{ANTIGEN_NAMES[i]}<-{c}' for i, c in enumerate(src_chains)]}, "
          f"바인더 B<-{args.binder_chain} ({len(binder_seq)} aa) -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
