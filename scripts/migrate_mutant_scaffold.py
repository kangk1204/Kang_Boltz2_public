#!/usr/bin/env python
"""나노바디 scaffold(엔티티 서열)가 바뀐 테스트셋 항목의 변이 라이브러리를 이관한다.

H-01/H-02(정밀) 수정으로 일부 항목의 엔티티(SEQRES) 서열이 관측 서열과 달라지면
(예: 8qf4 의 N-말단 Gly, 9ho5/7oao 의 C-말단 tag, 내부 결측 잔기 복원), 기존 변이
FASTA 는 옛 scaffold 길이에 맞춰져 있어 그대로 쓸 수 없다.

이 스크립트는 옛 나노바디와 새 나노바디를 정렬해 **동일한 아미노산 치환을 새
scaffold 의 대응 위치에 그대로 옮긴다**. 즉 변이 집합(어떤 잔기를 무엇으로 바꿨는지)은
보존하고 좌표/번호만 새 서열에 맞춘다.

  python scripts/migrate_mutant_scaffold.py --dir examples/testset \
      --backup /tmp/ts_backup --entries 8qf4,9ho5,7oao
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_input as P


def build_mapping(old: str, new: str):
    """old 의 각 index -> new 의 index (동일 서열 블록). 정렬 실패 시 None."""
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    mapping = {}
    for tag, i1, i2, j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
    if len(mapping) != len(old):
        return None
    return mapping


def remap_mutant(old: str, new: str, mutant: str, mapping):
    assert len(mutant) == len(old), (len(mutant), len(old))
    out = list(new)
    subs = []
    for i, (o, m) in enumerate(zip(old, mutant, strict=False)):
        if m != o:
            assert new[mapping[i]] == o, f"정렬 불일치: {i} {o}->{new[mapping[i]]}"
            out[mapping[i]] = m
            subs.append((mapping[i], o, m))
    return "".join(out), subs


def write_fasta(path: Path, header: str, seq: str):
    with path.open("w") as fh:
        fh.write(f">{header}\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i + 60] + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("examples/testset"))
    ap.add_argument("--backup", type=Path, required=True, help="갱신 전 항목 디렉터리 스냅샷")
    ap.add_argument("--entries", required=True, help="콤마 구분 항목 이름")
    args = ap.parse_args()

    for entry in [e.strip() for e in args.entries.split(",") if e.strip()]:
        cur = args.dir / entry
        bak = args.backup / entry
        old_nb = P.read_fasta(bak / "nanobody.fasta")[0]["sequence"]
        new_nb = P.read_fasta(cur / "nanobody.fasta")[0]["sequence"]
        mapping = build_mapping(old_nb, new_nb)
        if mapping is None:
            raise SystemExit(f"{entry}: 옛/새 나노바디 정렬 실패 (자동 이관 불가)")
        tsv = bak / "mutants.tsv"
        rows = list(csv.DictReader(io.StringIO(tsv.read_text(encoding="utf-8")), delimiter="\t"))
        out_rows = []
        for row in rows:
            name = row["name"]
            mutant = P.read_fasta(bak / "mutants" / f"{name}.fasta")[0]["sequence"]
            new_seq, subs = remap_mutant(old_nb, new_nb, mutant, mapping)
            desc = ", ".join(f"{o}{i + 1}{m}" for i, o, m in sorted(subs))
            note = f"m{len(subs)}: {desc}" if subs else "m0: 야생형 대조"
            write_fasta(cur / "mutants" / f"{name}.fasta", f"{name} ({desc})", new_seq)
            new_row = dict(row)
            new_row["notes"] = note
            out_rows.append(new_row)
        cols = ["name", "antigen", "nanobody", "reference", "notes"]
        with (cur / "mutants.tsv").open("w") as fh:
            fh.write("\t".join(cols) + "\n")
            for row in out_rows:
                fh.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
        print(f"[migrate] {entry}: 변이 {len(out_rows) - 1}개 이관 "
              f"({len(old_nb)} -> {len(new_nb)} aa)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
