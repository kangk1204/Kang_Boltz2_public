#!/usr/bin/env python
"""Turn simple FASTA files into a Boltz-2 YAML input (+ sidecar metadata).

Single job mode:  --target antigen.fasta --nanobody nb.fasta
The first target record becomes chain A, the nanobody becomes chain B and every
additional target record becomes chain C, D, E, ... (chain B is reserved for the
nanobody so the report can label the chains automatically).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

from runtime_state import ensure_path_within, validate_run_name

# H-09: 알파벳 26자를 모두 허용하면 'lysozyme' 같은 단어가 조용히 서열로 해석된다.
# 표준 20종 + 모호/특수 코드만 허용한다.
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
TARGET_CHAIN_POOL = ["A", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                     "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
NANOBODY_CHAIN = "B"


def read_fasta(path: Path):
    """Return [{"header":..., "sequence":...}, ...] for a (possibly multi-record) FASTA.

    H-09: BOM(엑셀/윈도우 저장) 때문에 첫 레코드가 사라지지 않도록 utf-8-sig 로 읽고,
    첫 헤더 이전의 내용은 오류로 처리한다.
    """
    records, header, chunks = [], None, []
    with Path(path).open(encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append({"header": header, "sequence": "".join(chunks)})
                header, chunks = line[1:].strip(), []
            elif header is None:
                raise ValueError(f"{path}: 첫 '>' 헤더 이전에 서열/텍스트가 있습니다: {line[:40]!r}")
            else:
                chunks.append(line.replace(" ", "").upper())
    if header is not None:
        records.append({"header": header, "sequence": "".join(chunks)})
    if not records:
        raise ValueError(f"no sequences found in {path}")
    for rec in records:
        bad = sorted(set(rec["sequence"]) - VALID_AA)
        if bad:
            raise ValueError(f"{path}: invalid characters {bad} in '{rec['header']}'")
        if not rec["sequence"]:
            raise ValueError(f"{path}: empty sequence for '{rec['header']}'")
    return records


def parse_hotspot(spec: str, default_chain: str):
    """'45,67' 또는 'A:45,A:67' 또는 '45-60' -> [(chain, resnum), ...]"""
    if str(spec or "").strip() in ("", "-"):
        return []   # '-' 는 표에서 '없음'을 뜻하는 관례
    spots = []
    for chunk in str(spec or "").replace(" ", "").split(","):
        if not chunk:
            continue
        if ":" in chunk:
            chain, res = chunk.split(":", 1)
        else:
            chain, res = default_chain, chunk
        if "-" in res:
            lo, hi = (int(x) for x in res.split("-", 1))
            if lo > hi:
                raise ValueError(f"hotspot 범위가 거꾸로입니다: '{chunk}' (시작 <= 끝 이어야 함)")
            if lo < 1:
                raise ValueError(f"hotspot 잔기 번호는 1 이상이어야 합니다: '{chunk}'")
            spots += [(chain, n) for n in range(lo, hi + 1)]
        else:
            n = int(res)
            if n < 1:
                raise ValueError(f"hotspot 잔기 번호는 1 이상이어야 합니다: '{chunk}'")
            spots.append((chain, n))
    if str(spec or "").strip() and not spots:
        raise ValueError(f"hotspot 입력을 해석하지 못했습니다: '{spec}'")
    return spots


def build_yaml(chains, msa: str, hotspot=None, binder: str = NANOBODY_CHAIN):
    lines = ["version: 1", "sequences:"]
    for ch in chains:
        lines.append("  - protein:")
        lines.append(f"      id: {ch['id']}")
        lines.append(f"      sequence: {ch['sequence']}")
        if msa == "empty":
            lines.append("      msa: empty")
        elif msa and msa not in ("server", "cache"):
            lines.append(f"      msa: {msa}")
    if hotspot:
        contacts = ", ".join(f"[{c}, {n}]" for c, n in hotspot)
        lines.append("constraints:")
        lines.append("  - pocket:")
        lines.append(f"      binder: {binder}")
        lines.append(f"      contacts: [{contacts}]")
        lines.append("      max_distance: 6.0")
    return "\n".join(lines) + "\n"


def build_job(target_records, nanobody_records, name, msa, target_file=None, nanobody_file=None,
              hotspot_spec=None):
    if not nanobody_records:
        if hotspot_spec:
            raise ValueError("target-only 입력에는 hotspot(pocket 제약)을 쓸 수 없습니다")
        # target-only: 항원 체인만 (A, C, D, ...)
        chains = []
        for i, rec in enumerate(target_records):
            if i >= len(TARGET_CHAIN_POOL):
                raise ValueError("too many target chains (max 25)")
            chains.append({"id": TARGET_CHAIN_POOL[i], "role": "target",
                           "sequence": rec["sequence"], "header": rec["header"],
                           "source": str(target_file) if target_file else None})
        return build_yaml(chains, msa), {
            "name": name, "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "chains": chains, "antigen_chain": chains[0]["id"] if chains else None,
            "nanobody_chain": None, "msa": msa, "target_only": True,
        }

    chains = []
    for i, rec in enumerate(target_records):
        if i >= len(TARGET_CHAIN_POOL):
            raise ValueError("too many target chains (max 25)")
        ch = TARGET_CHAIN_POOL[i]
        if ch == NANOBODY_CHAIN:
            ch = TARGET_CHAIN_POOL[len(target_records) + i]
        chains.append({"id": ch, "role": "antigen", "sequence": rec["sequence"],
                       "header": rec["header"], "source": str(target_file) if target_file else None})
    for rec in nanobody_records[:1]:
        chains.append({"id": NANOBODY_CHAIN, "role": "nanobody", "sequence": rec["sequence"],
                       "header": rec["header"], "source": str(nanobody_file) if nanobody_file else None})
    if len(nanobody_records) > 1:
        raise ValueError("nanobody FASTA must contain exactly one sequence per job "
                         "(use batch mode for multiple nanobodies)")
    hotspot = None
    if hotspot_spec:
        hotspot = parse_hotspot(hotspot_spec, chains[0]["id"]) if chains else []
        roles = {x["id"]: x.get("role") for x in chains}
        for c, n in hotspot:
            if c not in roles:
                raise ValueError(f"hotspot 체인 '{c}' 이 입력 체인에 없습니다 "
                                 f"(사용 가능: {[x['id'] for x in chains]})")
            # M-06(정밀): hotspot 은 항원 epitope 지정이다. binder(나노바디) 자신의
            # 체인을 지정하면 자기-제약이 만들어지므로 오류로 막는다.
            if roles.get(c) == "nanobody":
                raise ValueError(f"hotspot 은 항원 체인에만 지정할 수 있습니다 "
                                 f"(체인 '{c}' 는 나노바디입니다)")
            target_lens = {x["id"]: len(x["sequence"]) for x in chains}
            if n < 1 or n > target_lens.get(c, 0):
                raise ValueError(f"hotspot 잔기 {c}:{n} 이 체인 길이"
                                 f"({target_lens.get(c)})를 벗어납니다")
    meta = {
        "name": name,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hotspot": [{"chain": c, "resnum": n} for c, n in hotspot] if hotspot else None,
        "chains": chains,
        "antigen_chain": chains[0]["id"] if chains else None,
        "nanobody_chain": NANOBODY_CHAIN,
        "msa": msa,
    }
    return build_yaml(chains, msa, hotspot=hotspot), meta


def print_summary(meta, yaml_path, meta_path):
    print(f"\n[prepare] job '{meta['name']}'")
    for ch in meta["chains"]:
        role = "나노바디" if ch["role"] == "nanobody" else "항원"
        print(f"  chain {ch['id']} ({role:6s}) {len(ch['sequence']):4d} aa  "
              f"{ch['header'][:60]}")
    print(f"  MSA: {meta['msa']}")
    if meta.get("hotspot"):
        print("  hotspot(pocket 제약): "
              + ", ".join(f"{h['chain']}:{h['resnum']}" for h in meta["hotspot"]))
    print(f"  YAML: {yaml_path}")
    print(f"  meta: {meta_path}")


def main():
    ap = argparse.ArgumentParser(description="FASTA -> Boltz-2 YAML")
    ap.add_argument("--target", type=Path, help="antigen FASTA (may contain multiple chains)")
    ap.add_argument("--nanobody", type=Path, help="nanobody (VHH) FASTA, single sequence")
    ap.add_argument("--target-only", action="store_true",
                    help="나노바디 없이 항원(들)만으로 예측 (타겟 단독/복합체 구조 예측)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", type=Path, default=Path("inputs/generated"))
    ap.add_argument("--hotspot", default=None,
                    help="항원의 epitope 잔기 (예: '45,67,101' 또는 'A:45-60'). pocket 제약으로 추가")
    ap.add_argument("--msa", default="server", choices=["server", "empty", "cache"],
                    help="'server' = use --use_msa_server at run time, 'empty' = single sequence")
    args = ap.parse_args()

    if not args.target:
        raise SystemExit("--target is required")
    if not args.target_only and not args.nanobody:
        raise SystemExit("--nanobody is required (또는 --target-only)")
    if args.target_only and args.hotspot:
        raise SystemExit(
            "--target-only 에서는 --hotspot 을 쓸 수 없습니다.\n"
            "  pocket 제약은 '바인더 체인'(B)이 있어야 의미가 있습니다.\n"
            "  특정 epitope 에 나노바디/바인더를 붙이는 예측이면 --nanobody 를 함께 주세요.")

    try:
        name = validate_run_name(args.name)
    except ValueError as exc:
        raise SystemExit(f"--name 오류: {exc}") from None

    target_records = read_fasta(args.target)
    nanobody_records = read_fasta(args.nanobody) if args.nanobody else []
    yaml_text, meta = build_job(target_records, nanobody_records, name, args.msa,
                                args.target, args.nanobody, hotspot_spec=args.hotspot)

    outdir = args.outdir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        yaml_path = ensure_path_within(outdir / f"{name}.yaml", outdir, label="YAML path")
        meta_path = ensure_path_within(outdir / f"{name}.meta.json", outdir, label="meta path")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    yaml_path.write_text(yaml_text, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print_summary(meta, yaml_path, meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
