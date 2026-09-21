#!/usr/bin/env python
"""CDR 변이 라이브러리 생성 -> 배치 실행용 TSV 만들기.

나노바디 CDR(IMGT 기준, ANARCI) 위치에 무작위/알라닌/보존적 변이를 도입한
FASTA 파일들을 만들고, `./run.sh --batch <tsv>` 로 바로 평가할 수 있는 표를 출력합니다.

예)
  # CDR1+2+3 에 변이 2개씩, 12개 variant (야생형 대조 포함)
  python scripts/make_cdr_library.py --cdrs CDR1,CDR2,CDR3 --n-mutations 2 --n-variants 12 --seed 1

  # CDR3만 알라닌 스캐닝, 8개
  python scripts/make_cdr_library.py --cdrs CDR3 --n-mutations 1 --n-variants 8 --mode ala

  # 생성된 표로 배치 실행 (야생형 포함)
  ./run.sh --batch cdr_library.tsv --name cdr_screen1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import random
import sys
from pathlib import Path

import metrics
import prepare_input as P
from runtime_state import validate_run_name

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_NO_CYS = [a for a in AA20 if a != "C"]
# 보존적 치환 그룹 (Murphy et al. 방식 단순화)
CONSERVATIVE = {
    "A": "GAS", "G": "AG", "S": "STA", "T": "TSA", "C": "CS",
    "V": "VILM", "I": "IVLM", "L": "LIVM", "M": "MILV",
    "D": "DE", "E": "ED", "N": "NQ", "Q": "QN",
    "K": "KR", "R": "RK", "H": "HY", "F": "FYW", "Y": "YFW", "W": "WYF",
}


def parse_n_mutations(spec: str):
    """'2' | '1-3' | '1,2,3' -> [2] / [1,2,3] / [1,2,3]

    R3-06: 중복 값('1,1')은 고유값으로 정규화한다 (용량/조합수/열거가 모두 같은 목록을 쓰도록).
    """
    spec = str(spec).strip()
    try:
        if "-" in spec:
            lo, hi = (int(x) for x in spec.split("-", 1))
            if lo < 1 or hi < lo:
                raise ValueError
            vals = list(range(lo, hi + 1))
        elif "," in spec:
            vals = [int(x) for x in spec.split(",")]
            if any(v < 1 for v in vals):
                raise ValueError
        else:
            v = int(spec)
            if v < 1:
                raise ValueError
            vals = [v]
        uniq = sorted(set(vals))
        if len(uniq) != len(vals):
            print(f"  [안내] --n-mutations 중복 값을 고유값으로 정규화: {vals} -> {uniq}")
        return uniq
    except ValueError:
        raise SystemExit(f"--n-mutations 형식 오류: '{spec}' (예: 2, 1-3, 1,2,3)") from None


def load_nanobody(path: Path):
    recs = P.read_fasta(path)
    if len(recs) != 1:
        raise SystemExit("나노바디 FASTA 에는 서열이 정확히 1개 있어야 합니다")
    return recs[0]


def get_cdr_positions(seq: str, scheme: str = "imgt"):
    hmmer_dir = metrics.find_hmmer_dir()
    if hmmer_dir:
        import os
        os.environ["PATH"] = hmmer_dir + os.pathsep + os.environ.get("PATH", "")
    try:
        import anarci  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "ANARCI Python package를 현재 Python에서 import할 수 없습니다.\n"
            f"  Python: {Path(sys.executable)}\n"
            f"  원인: {exc}\n"
            "  설치/실행은 run.sh 로 감싸 쓰면 편합니다:\n"
            "    bash setup.sh --reuse-env\n"
            "    ./run.sh --make-cdr-library --nanobody nb.fasta --target ag.fasta ..."
        ) from None
    if metrics.find_executable("hmmscan") is None:
        raise SystemExit(
            "ANARCI는 설치되어 있지만 hmmscan(HMMER)을 찾지 못했습니다.\n"
            "  설치: bash setup.sh --with-hmmer --reuse-env\n"
            "  또는 HMMER_DIR=/path/to/bin 을 설정하세요."
        )
    ann = metrics.annotate_cdrs(seq, scheme=scheme)
    if not ann:
        raise SystemExit(
            "ANARCI가 이 서열을 항체 가변 도메인(VHH/VH/VL)으로 번호매김하지 못했습니다.\n"
            "  서열이 가변 도메인인지, 태그/절단/비표준 문자가 섞였는지 확인하세요."
        )
    pos = {k: [x["seq_index"] for x in v] for k, v in ann["cdrs"].items()}
    return pos, ann


def verify_paratope_sequence(paratope, seq: str, source: str) -> None:
    """C-1: 파라토프/접촉 잔기 정보가 지금 쓰는 서열과 같은 나노바디인지 확인한다.

    results.json 에는 나노바디 서열이 함께 저장되므로, 불일치하면 조용히 엉뚱한
    위치에 변이를 넣는 대신 즉시 실패시킨다.
    """
    ref = (paratope.get("sequence") if isinstance(paratope, dict) else None) or None
    if not ref:
        print("[warn] --paratope-from 결과에 나노바디 서열이 없어 일치 검증을 건너뜁니다")
        return
    if ref.strip().upper() != seq.strip().upper():
        raise SystemExit(
            f"{source} 의 나노바디 서열이 현재 서열과 다릅니다 "
            f"(길이 {len(ref)} vs {len(seq)}). 올바른 나노바디로 다시 실행하세요.")


def load_paratope_positions(run_dir: Path, min_contacts: int):
    """results.json 에서 나노바디 인터페이스(파라토프) 잔기 위치(0-based)를 읽는다."""
    import json
    res_path = run_dir / "analysis" / "results.json"
    if not res_path.exists():
        raise SystemExit(f"파라토프 정보를 찾을 수 없습니다: {res_path} (먼저 해당 나노바디로 "
                         f"단일 예측을 실행하세요: ./run.sh --name <이름>)")
    r = json.loads(res_path.read_text())
    best = r["models"][0]
    iface = (best.get("interface_8A") or {}).get("residues_b") or []
    pos, contacts = {}, {}
    for rec in iface:
        if rec["n_contacts"] >= min_contacts:
            pos[rec["resnum"] - 1] = rec["resnum"]
            contacts[rec["resnum"]] = rec["n_contacts"]
    return pos, contacts, r


def choices_for(aa: str, mode: str, allow_cys: bool):
    """해당 잔기에서 가능한 치환 아미노산 목록 (pick_substitution 과 동일 규칙)."""
    if mode == "ala":
        return ["A"]
    if mode == "conservative":
        cand = [a for a in CONSERVATIVE.get(aa, "") if a != aa and (allow_cys or a != "C")]
        if cand:
            return cand
        # M-07(정밀): pick_substitution 은 conservative 에 정의가 없으면 일반 풀로
        # 폴백한다. 조합 수 계산도 같은 규칙을 써야 P(Pro) 하나 때문에 전체가
        # 예외로 중단되지 않는다.
        pool = AA20 if allow_cys else AA_NO_CYS
        return [a for a in pool if a != aa]
    pool = AA20 if allow_cys else AA_NO_CYS
    return [a for a in pool if a != aa]


def _counts_by_k(choice_counts, mut_values):
    """k 별로 만들 수 있는 조합 수 (정확값)."""
    e = [1]
    for c in choice_counts:
        new = [*e, 0]
        for k in range(1, len(new)):
            new[k] = (e[k] if k < len(e) else 0) + c * e[k - 1]
        e = new
    return [int(e[k]) if 0 <= k < len(e) else 0 for k in mut_values]


def count_library_size(choice_counts, mut_values):
    """가능한 서로 다른 variant 수 (정확값).

    위치별 치환 선택지 수 c_i 가 다를 수 있으므로, k개 변이 조합 수는
    prod(1 + c_i x) 의 x^k 계수(초등대칭다항식)로 계산한다 (R11).
    c_i 가 모두 같은 값이면 C(n,k) * c^k 와 일치한다.
    """
    e = [1]
    for c in choice_counts:
        new = [*e, 0]
        for k in range(1, len(new)):
            new[k] = (e[k] if k < len(e) else 0) + c * e[k - 1]
        e = new
    return int(sum(e[k] for k in mut_values if 0 <= k < len(e)))


def estimate_library_size(pool_size: int, choice_counts, mut_values):
    """(하위 호환) 정확한 조합 수를 돌려준다."""
    if not choice_counts:
        return 0
    return count_library_size(choice_counts, mut_values)


def build_exhaustive(pool, seq, mode, allow_cys, mut_values, rng, cap):
    """모든 조합을 열거(상한 cap)한 뒤 섞어서 반환. 완전한 스캐닝(예: Ala scan)용."""
    import itertools
    items = []
    for k in mut_values:
        for pos_combo in itertools.combinations(range(len(pool)), k):
            options = [list(choices_for(seq[pool[idx][1]], mode, allow_cys)) for idx in pos_combo]
            for subs_combo in itertools.product(*options):
                items.append({pool[idx][1]: aa for idx, aa in zip(pos_combo, subs_combo, strict=False)})
                if len(items) >= cap:
                    break
            if len(items) >= cap:
                break
        if len(items) >= cap:
            break
    rng.shuffle(items)
    return items


def pick_substitution(aa: str, mode: str, rng: random.Random, allow_cys: bool):
    if mode == "ala":
        return "A"
    pool = AA20 if allow_cys else AA_NO_CYS
    if mode == "conservative":
        cand = [a for a in CONSERVATIVE.get(aa, "") if a != aa and (allow_cys or a != "C")]
        if cand:
            return rng.choice(cand)
    choices = [a for a in pool if a != aa]
    return rng.choice(choices)


def write_fasta(path: Path, header: str, seq: str):
    with path.open("w") as fh:
        fh.write(f">{header}\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i + 60] + "\n")


def main():
    ap = argparse.ArgumentParser(description="CDR 변이 라이브러리 -> 배치 TSV 생성")
    ap.add_argument("--nanobody", type=Path, default=Path("inputs/nanobody.fasta"))
    ap.add_argument("--target", type=Path, default=Path("inputs/target.fasta"))
    ap.add_argument("--reference", default=None,
                    help="정답 복합체(cif/pdb) - 야생형 대조 job 에만 적용됩니다")
    ap.add_argument("--scheme", default="imgt", choices=["imgt", "kabat", "chothia", "martin"],
                    help="CDR 번호 체계 (기본 imgt)")
    ap.add_argument("--paratope-from", type=Path, default=None,
                    help="기존 실행 폴더(outputs/<name>)의 결과를 읽어 인터페이스(파라토프) 잔기만 변이")
    ap.add_argument("--paratope-min-contacts", type=int, default=1,
                    help="파라토프로 인정할 최소 접촉 수 (기본 1)")
    ap.add_argument("--exclude-positions", default="",
                    help="변이에서 제외할 잔기 번호(FASTA 1-based, 콤마 구분)")
    ap.add_argument("--cdrs", default="CDR1,CDR2,CDR3",
                    help="변이를 넣을 CDR (콤마 구분: CDR1,CDR2,CDR3)")
    ap.add_argument("--n-mutations", default="2",
                    help="variant 당 변이 개수. 범위/목록도 가능: '1-3', '1,2,3' (기본 2)")
    ap.add_argument("--n-variants", type=int, default=12, help="생성할 variant 수")
    ap.add_argument("--mode", choices=["random", "ala", "conservative"], default="random")
    ap.add_argument("--exhaustive", action="store_true",
                    help="가능한 조합을 모두 열거(중복 0, 빠짐 0) - 예: CDR3 Ala scan 전체. "
                         "조합이 200만개 이하일 때만 사용")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--allow-cys", action="store_true",
                    help="Cys 로의 치환 허용 (기본: 금지 - 이황화결합 보존)")
    ap.add_argument("--keep-cys-positions", dest="keep_cys_positions", action="store_true",
                    default=True, help="CDR 내 Cys 위치를 변이 대상에서 제외 (기본 켜짐)")
    ap.add_argument("--allow-cys-positions", dest="keep_cys_positions", action="store_false",
                    help="CDR 내 Cys 위치도 변이 대상에 포함")
    ap.add_argument("--outdir", type=Path, default=Path("examples/cdr_library"))
    ap.add_argument("--prefix", default=None, help="variant 이름 접두어 (기본: 나노바디 FASTA 헤더 첫 단어)")
    ap.add_argument("--batch-out", type=Path, default=Path("cdr_library.tsv"))
    args = ap.parse_args()

    rec = load_nanobody(args.nanobody)
    seq = rec["sequence"]
    if args.prefix:
        prefix = args.prefix
    else:
        import re as _re
        first = rec["header"].split()[0].replace(">", "")
        first = _re.split(r"_(?:VHH|vhh|nanobody|nb|VH|vh)(?=_|$)", first)[0]
        prefix = _re.sub(r"[^A-Za-z0-9._-]", "_", first)[:24] or "nb"
    try:
        prefix = validate_run_name(prefix)
    except ValueError as exc:
        raise SystemExit(f"--prefix 오류: {exc}") from None

    cdr_positions, _ann = get_cdr_positions(seq, scheme=args.scheme)
    exclude = set()
    for chunk in str(args.exclude_positions).replace(" ", "").split(","):
        if not chunk:
            continue
        if not chunk.lstrip("+-").isdigit():
            raise SystemExit(f"--exclude-positions 값이 정수가 아닙니다: '{chunk}'")
        exclude.add(int(chunk) - 1)
    wanted = list(dict.fromkeys(c.strip().upper() for c in args.cdrs.split(",") if c.strip()))
    bad = [c for c in wanted if c not in cdr_positions]
    if bad:
        raise SystemExit(f"알 수 없는 CDR: {bad} (사용 가능: {list(cdr_positions)})")

    paratope_pos = None
    if args.paratope_from:
        paratope_pos, _contacts, _res = load_paratope_positions(
            Path(args.paratope_from), args.paratope_min_contacts)
        nb_meta = ((_res.get("models") or [{}])[0].get("cdr") or {})
        verify_paratope_sequence(nb_meta, seq, f"--paratope-from {args.paratope_from}")
        print(f"파라토프 필터: {args.paratope_from} 의 인터페이스 잔기 "
              f"{len(paratope_pos)}개 (접촉 >= {args.paratope_min_contacts})")

    pool = []
    for cdr in wanted:
        for i in cdr_positions[cdr]:
            if args.keep_cys_positions and seq[i] == "C":
                continue
            if args.mode == "ala" and seq[i] == "A":
                continue  # Ala 스캐닝에서 이미 Ala 인 위치는 의미 없음
            if i in exclude:
                continue
            if paratope_pos is not None and i not in paratope_pos:
                continue
            pool.append((cdr, i))
    mut_values = parse_n_mutations(args.n_mutations)
    if max(mut_values) > len(pool):
        raise SystemExit(f"변이 가능한 위치({len(pool)}) 보다 변이 개수({max(mut_values)}) 가 많습니다")

    if args.n_variants < 1:
        raise SystemExit("--n-variants 는 1 이상이어야 합니다")
    rng = random.Random(args.seed)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"나노바디: {prefix} ({len(seq)} aa) | CDR 체계: {args.scheme.upper()}")
    for cdr in wanted:
        s = "".join(seq[i] for i in cdr_positions[cdr])
        print(f"  {cdr}: {len(cdr_positions[cdr])} aa, 서열 {s}")
    dist = {v: sum(1 for i in range(args.n_variants)
                   if mut_values[i % len(mut_values)] == v)
            for v in sorted(set(mut_values))}
    choice_counts = [len(choices_for(seq[i], args.mode, args.allow_cys)) for _, i in pool]
    total_possible = estimate_library_size(len(pool), choice_counts, mut_values)
    print(f"  변이 가능 위치 {len(pool)}개 (Cys 제외), mode={args.mode}, seed={args.seed}")
    print("  변이 개수 분포: " + ", ".join(f"{v}개×{n}" for v, n in dist.items()))
    print(f"  서로 다른 variant 조합: {total_possible:,}개 "
          f"(요청 {args.n_variants}개 = 점유율 {100.0 * args.n_variants / max(total_possible, 1):.1f}%)")
    if args.n_variants > total_possible and args.exhaustive:
        print(f"  exhaustive: 요청 {args.n_variants}개 > 가능한 조합 {total_possible:,}개 "
              f"-> 가능한 것 전부({total_possible:,}개)를 만듭니다")
        args.n_variants = int(total_possible)
    elif args.n_variants > total_possible:
        raise SystemExit(
            f"요청한 {args.n_variants}개를 만들 수 없습니다 (가능한 조합 {total_possible:,}개).\n"
            "  - 변이 개수를 늘리거나(--n-mutations 1-3), CDR 범위를 확장하거나 "
            "--exhaustive 로 가능한 것만 만드세요.")
    if args.n_variants > 0.25 * total_possible:
        print("  [주의] 가능한 조합의 25%를 넘게 요청했습니다 - 중복 재추첨이 늘어 느려질 수 있습니다.")

    exhaustive_items = None
    if args.exhaustive:
        if total_possible > 2_000_000:
            raise SystemExit(
                f"--exhaustive 는 조합이 200만개 이하일 때만 사용하세요 (현재 약 {total_possible:,}개). "
                "변이 개수를 줄이거나 random 모드를 쓰세요.")
        # total_possible <= 2e6 이 보장되므로 전부 열거한다.
        # (cap 을 요청 수 근처로 잡으면 앞쪽 k 값만 뽑히는 편향이 생긴다)
        exhaustive_items = build_exhaustive(pool, seq, args.mode, args.allow_cys, mut_values, rng,
                                            cap=int(total_possible) + 1)
        print(f"  exhaustive: 전체 조합 {total_possible:,}개 중 {len(exhaustive_items):,}개 열거(셔플)")
    print()

    rows = [("name", "antigen", "nanobody", "reference", "notes")]
    # 야생형 대조
    wt_fasta = outdir / f"{prefix}_WT.fasta"
    write_fasta(wt_fasta, f"{prefix}_WT wild-type", seq)
    rows.append((f"{prefix}_WT", str(args.target), str(wt_fasta),
                 args.reference or "-", "m0: 야생형 대조"))
    print(f"  {prefix}_WT  (변이 없음)")

    seen = {seq}  # 야생형과 동일한 variant, 중복 variant 금지
    made = 0
    attempts = 0
    max_attempts = max(200, 100 * args.n_variants)

    # RR07: k 별 남은 생성 용량 (e_k) 을 추적해 소진된 k 만 계속 고르는 문제를 막는다
    capacity = dict(zip(mut_values, _counts_by_k(choice_counts, mut_values), strict=False))
    while made < args.n_variants:
        attempts += 1
        if attempts > max_attempts:
            print(f"  [경고] 서로 다른 variant 를 {args.n_variants}개 만들지 못했습니다 "
                  f"(가능 조합 부족). 지금까지 만든 {made}개로 표를 작성합니다.")
            break
        if exhaustive_items is not None:
            if not exhaustive_items:
                print(f"  (가능한 variant 를 모두 생성했습니다: {made}개)")
                break
            subs = exhaustive_items.pop()
            n_mut = len(subs)
        else:
            avail = [k for k in mut_values if capacity.get(k, 0) > 0]
            if not avail:
                print(f"  [경고] 요청한 변이 개수 조합을 모두 소진했습니다 (생성 {made}개)")
                break
            n_mut = avail[made % len(avail)]
            picks = rng.sample(pool, n_mut)
            subs = {i: pick_substitution(seq[i], args.mode, rng, args.allow_cys) for _, i in picks}
        mutant = list(seq)
        for i, aa in subs.items():
            mutant[i] = aa
        mutant = "".join(mutant)
        if mutant in seen:
            continue
        seen.add(mutant)
        capacity[n_mut] = max(0, capacity.get(n_mut, 0) - 1)
        name = f"{prefix}_v{made + 1:02d}"
        desc = ", ".join(f"{seq[i]}{i + 1}{subs[i]}" for i in sorted(subs))
        note = f"m{n_mut}: {desc}"
        fasta = outdir / f"{name}.fasta"
        write_fasta(fasta, f"{name} ({desc})", mutant)
        rows.append((name, str(args.target), str(fasta), "-", note))
        print(f"  {name:22s} m{n_mut}  {desc}")
        made += 1

    tsv = Path(args.batch_out)
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    shortfall = args.n_variants - made
    print(f"\n[CDR library] FASTA {made + 1}개 -> {outdir}")
    print(f"[CDR library] 배치 표 -> {tsv}")
    print("\n다음 명령으로 평가하세요 (예측 + 정렬 가능한 표 리포트):")
    print(f"  ./run.sh --batch {tsv} --name cdr_screen --samples 3 --msa server")
    print("  ./run.sh --serve outputs/batch_cdr_screen/report")
    print("\n해석 팁: ipTM/ipSAE/pDockQ2 하락 = 구조 예측상 인터페이스 신뢰도 하락,\n"
          "          나노바디/CDR3 pLDDT 하락 = 루프 모델링 신뢰도 하락. 둘 다 실제 결합력이\n"
          "          아니라 예측 신뢰도이므로, 결합 여부는 SPR/BLI 등 실험으로 확인하세요.")
    print(f"생성 시각: {_dt.datetime.now().astimezone().isoformat(timespec='seconds')}")
    if shortfall > 0 and args.exhaustive:
        # exhaustive 는 '가능한 것 전부'가 계약이지만, 요청 수를 못 채운 사실을 분명히 알린다
        print(f"[안내] exhaustive: 요청 {args.n_variants}개 중 가능한 {made}개만 생성했습니다 "
              f"(중복/공간 한계). 표에는 {made}개가 들어 있습니다.")
        return 0
    if shortfall > 0 and not args.exhaustive:
        print(f"[오류] 요청 {args.n_variants}개 중 {made}개만 생성했습니다 (부족 {shortfall}개). "
              f"가능 조합 수를 확인하세요. TSV 는 부분 라이브러리로 저장했습니다: {tsv}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
