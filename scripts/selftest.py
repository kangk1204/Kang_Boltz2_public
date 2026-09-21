#!/usr/bin/env python
"""Self-test for the analysis stack.

  python scripts/selftest.py                 # 수학/파싱 단위 테스트
  python scripts/selftest.py --run-dir outputs/1MEL_demo   # 실제 실행 결과 일관성 검증
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics
from metrics import calc_d0, calc_d0_array, tm_func


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def test_d0():
    assert approx(calc_d0(19), 1.0), "d0 for small length must be clamped to 1.0"
    assert approx(calc_d0(27), 1.0)
    # Yang & Skolnick: d0 = 1.24*(L-15)^(1/3) - 1.8
    assert approx(calc_d0(100), 1.24 * (85) ** (1 / 3) - 1.8)
    arr = calc_d0_array([5, 26, 100])
    assert approx(arr[0], 1.0) and approx(arr[1], 1.0)
    assert approx(arr[2], 1.24 * (85) ** (1 / 3) - 1.8)
    print("  [ok] d0 functions")


def test_tm_func():
    assert approx(tm_func(0, 5), 1.0)
    assert approx(tm_func(5, 5), 0.5)
    print("  [ok] tm_function")


def test_ipsae_pure_handcase():
    """Hand-computed ipSAE for a tiny 2-chain case (cutoff=10, d0=1.0)."""
    n_a, n_b = 30, 30
    pae = np.full((n_a + n_b, n_a + n_b), 25.0)
    # rows 0..2 (chain A) see 26 residues of chain B with PAE = 2.0 -> d0 clamp 1.0
    for i in range(3):
        pae[i, n_a:n_a + 26] = 2.0
    ids = np.array(["A"] * n_a + ["B"] * n_b)
    value, byres = metrics.ipsae_pure(pae, ids, "A", "B", cutoff=10.0)
    n0res = 26
    d0 = max(1.0, 1.24 * (max(26, n0res) - 15) ** (1 / 3) - 1.8)
    expected = 1.0 / (1.0 + (2.0 / d0) ** 2)
    assert approx(value, expected, 1e-6), (value, expected)
    assert byres[0] == value
    print(f"  [ok] ipsae_pure hand case (value={value:.4f})")


def test_contact_and_interface():
    tokens = []
    for i in range(3):
        tokens.append({"chain": "A", "resnum": i + 1, "resname": "ALA", "aa": "A",
                       "ca": np.array([0.0, 0.0, 0.0]), "cb": np.array([0.0, 0.0, 0.0]),
                       "plddt": 90.0, "n_atoms": 1})
    for i in range(3):
        tokens.append({"chain": "B", "resnum": i + 1, "resname": "ALA", "aa": "A",
                       "ca": np.array([0.0, 0.0, 3.0]), "cb": np.array([0.0, 0.0, 3.0]),
                       "plddt": 80.0, "n_atoms": 1})
    pairs = metrics.contact_pairs(tokens, "A", "B", cutoff=8.0)
    assert len(pairs) == 9, len(pairs)
    pae = np.full((6, 6), 4.0)
    plddt = np.array([t["plddt"] for t in tokens])
    rep = metrics.interface_report(tokens, plddt, pae, "A", "B", cutoff=8.0)
    assert rep["n_contacts"] == 9
    assert approx(rep["pae_mean_ab"], 4.0) and approx(rep["plddt_mean_b"], 80.0)
    print("  [ok] contacts / interface report")


def test_fasta_and_job(tmp=Path("/tmp/opencode/selftest")):
    import prepare_input as P

    tmp.mkdir(parents=True, exist_ok=True)
    t = tmp / "ag.fasta"
    n = tmp / "nb.fasta"
    t.write_text(">ag chain1\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n>ag chain2\nGAVLIPES\n", encoding="utf-8")
    n.write_text(">nb\nQVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQ\n", encoding="utf-8")
    ag = P.read_fasta(t)
    nb = P.read_fasta(n)
    yaml_text, meta = P.build_job(ag, nb, "job1", "server", t, n)
    assert [c["id"] for c in meta["chains"]] == ["A", "C", "B"], meta["chains"]
    assert meta["nanobody_chain"] == "B"
    assert "msa:" not in yaml_text
    yaml_empty, _ = P.build_job(ag, nb, "job2", "empty")
    assert yaml_empty.count("msa: empty") == 3
    print("  [ok] FASTA parsing / YAML generation (multi-chain)")


def test_role_detection():
    import analyze

    tokens = ([{"chain": "A", "resnum": i, "resname": "GLY", "aa": "G"} for i in range(1, 130)]
              + [{"chain": "B", "resnum": i, "resname": "SER", "aa": "S"} for i in range(1, 121)])
    seq = ("VQLQASGGGSVQAGGSLRLSCAASGYTIGPYCMGWFRQAPGKEREGVAAINMGGGITYYADSVKGRFTISQDNAKNTVYLL"
           "MNSLEPEDTAIYYCAADSTIYASYYECGHGLSTGGYGYDSWGQGTQVTVSS")
    derived = seq + "S" * (120 - len(seq))
    called = []

    def fake_seq(tks, ch):
        called.append(ch)
        return derived if ch == "B" else "M" * 129

    orig = analyze.sequence_of
    analyze.sequence_of = fake_seq
    try:
        det = analyze.detect_nanobody_chain(tokens)
    finally:
        analyze.sequence_of = orig
    assert det == "B", det
    print("  [ok] VHH auto-detection")


def test_run_dir(run_dir: Path):
    """Consistency checks against a finished Boltz-2 run."""
    import analyze as A

    pred = A.find_predictions_dir(run_dir)
    models = A.model_files(pred)
    print(f"  run dir: {run_dir} ({len(models)} models)")
    for m in models:
        conf = json.loads(m["confidence"].read_text(encoding="utf-8"))
        plddt = np.load(m["plddt"])["plddt"].astype(float)
        plddt = plddt * 100 if plddt.max() <= 1.0 else plddt
        assert approx(plddt.mean(), conf["complex_plddt"] * 100, 0.05), \
            (plddt.mean(), conf["complex_plddt"])
        tokens = metrics.read_structure_tokens(m["cif"])
        assert len(tokens) == len(plddt), (len(tokens), len(plddt))
        pae = np.load(m["pae"])["pae"].astype(float)
        assert pae.shape[0] == len(tokens)
        chains = metrics.token_chain_array(tokens)
        ids = list(dict.fromkeys(chains.tolist()))
        res_path = run_dir / "analysis" / "results.json"
        nb = ag = None
        if res_path.exists():
            r = json.loads(res_path.read_text(encoding="utf-8"))
            nb, ag = r["nanobody_chain"], r["antigen_chain"]
        # M-13(정밀): target-only(나노바디 없음)에서는 인터페이스 지표가 '해당 없음'이다.
        # 체인쌍 ipTM/ipSAE 비교를 건너뛰고 pTM 만 conf 와 교차 확인한다.
        if not nb:
            ptm = conf.get("ptm")
            if ptm is not None:
                assert 0.0 <= float(ptm) <= 1.0, ptm
            print(f"    model_{m['index']}: [target-only] pLDDT mean {plddt.mean():.2f} | pTM {ptm}")
            continue
        # official ipSAE vs pure re-implementation
        official = metrics.run_official_ipsae(Path(__file__).parent / "vendor" / "ipsae_official.py",
                                              m["pae"], m["cif"], 10.0, 15.0)
        assert not official.get("error"), official.get("error")
        nb = nb or ids[-1]
        ag = ag or ids[0]
        d0 = metrics.calc_d0(max(len(tokens), 19))
        pure, _ = metrics.interchain_iptm(pae, np.where(chains == ag)[0], np.where(chains == nb)[0], d0)
        boltz_json = conf["pair_chains_iptm"][str(ids.index(nb))][str(ids.index(ag))]
        # Boltz는 PAE bin별 TM 값을 확률로 평균하고, proxy는 평균 PAE에 TM을 적용한다.
        # TM(E[PAE]) != E[TM(PAE)]: 두 추정량 사이에 고정 허용 오차를 보장할 수 없다.
        assert 0.0 <= pure <= 1.0, pure
        assert 0.0 <= boltz_json <= 1.0, boltz_json
        if res_path.exists():
            saved = next(e for e in r["models"] if e["index"] == m["index"])
            recorded = (saved.get("boltz_pair_iptm") or {}).get("nanobody_in_antigen_frame")
            assert recorded is not None and approx(recorded, boltz_json), \
                f"recorded pair ipTM differs from confidence JSON: {recorded} != {boltz_json}"
        # M-13(정밀): 공식 wrapper 값이 '공식 출력을 제대로 읽었는지'뿐 아니라
        # 독립 재구현(ipsae_pure)과도 일치하는지 검증한다. (값 999 주입 같은 오염 차단)
        pair = (official.get("max") or {}).get((min(ag, nb), max(ag, nb)))
        assert pair and pair.get("ipsae") is not None, official.get("max")
        pure_ab, _ = metrics.ipsae_pure(pae, chains, ag, nb, cutoff=10.0)
        pure_ba, _ = metrics.ipsae_pure(pae, chains, nb, ag, cutoff=10.0)
        pure_max = max(pure_ab, pure_ba)
        assert abs(float(pair["ipsae"]) - pure_max) < 0.05, (pair["ipsae"], pure_max)
        # R3-01/R4(테스트 공백) 회귀: 그림이 하나라도 등록된 run 이면
        #   (a) 등록된 경로가 실제로 존재하고
        #   (b) 정상(token_dims_ok) 모델은 최소 pae/plddt 그림을 가져야 한다.
        res_path = run_dir / "analysis" / "results.json"
        if res_path.exists():
            rr = json.loads(res_path.read_text(encoding="utf-8"))
            models_rr = rr.get("models", [])
            any_figs = any(e.get("figures") for e in models_rr)
            for e in models_rr:
                for key, fp in (e.get("figures") or {}).items():
                    assert Path(fp).is_file(), f"등록된 그림이 없음: {key} {fp}"
            if any_figs:
                for e in models_rr:
                    if not e.get("token_dims_ok", True):
                        continue
                    figs = e.get("figures") or {}
                    missing = [k for k in ("pae", "plddt") if k not in figs]
                    assert not missing, (
                        f"model_{e['index']} 에 필수 그림 누락 {missing} (그림 삭제/미등록 회귀). "
                        "분석 시 --no-figures 를 쓰지 않았다면 실패가 맞습니다")
                print(f"    그림 검사: 모델 {len(models_rr)}개, 필수 그림(pae/plddt) 존재 확인")
            elif rr.get("settings", {}).get("no_figures"):
                # 이번 분석이 --no-figures 로 실행됐다면 등록이 없는 게 정상이다 (R5-03)
                print("    그림 검사: settings.no_figures=true 로 실행된 결과라 건너뜀")
            else:
                # 등록은 없는데 PNG 파일이 남아 있으면 등록 누락(회귀)이다.
                # PNG 자체가 없으면 --no-figures 실행으로 보고 건너뛴다.
                figdir = run_dir / "analysis" / "figures"
                n_png = len(list(figdir.glob("*.png"))) if figdir.exists() else 0
                assert n_png == 0, (
                    f"figures 디렉터리에 PNG {n_png}개가 있는데 results.json 에 등록된 그림이 없습니다 "
                    "(등록 누락 회귀)")
                print("    그림 검사: 그림 파일이 없어 건너뜀 (--no-figures 실행일 수 있음)")
        print(f"    model_{m['index']}: pLDDT mean {plddt.mean():.2f} (json ok) | "
              f"PAE-derived ipTM proxy {pure:.3f} | Boltz pair ipTM {boltz_json:.3f} | "
              f"ipSAE {pair['ipsae']:.3f} | DockQ-ready")
    print("  [ok] run-dir consistency")


def test_chain_pair_table():
    """3체인 합성 케이스로 체인 쌍 표(ipTM 방향/접촉 수) 검증."""
    import analyze

    n = 30
    tokens = []
    for ch in "ABC":
        for i in range(10):
            tokens.append({"chain": ch, "resnum": i + 1, "resname": "ALA", "aa": "A",
                           "ca": np.array([0.0, 0.0, 0.0]), "cb": np.array([0.0, 0.0, 0.0]),
                           "plddt": 90.0, "n_atoms": 1})
    chains = metrics.token_chain_array(tokens)
    pae = np.full((n, n), 3.0)
    conf = {"pair_chains_iptm": {str(i): {str(j): 0.5 + 0.1 * i for j in range(3)}
                                 for i in range(3)}}
    rows = analyze.chain_pair_table(conf, tokens, pae, chains, d0=5.0)
    assert len(rows) == 3, len(rows)          # A-B, A-C, B-C
    assert [r["chain_a"] + r["chain_b"] for r in rows] == ["AB", "AC", "BC"]
    assert all(r["n_contacts"] > 0 for r in rows)
    assert approx(rows[0]["iface_pae_a_frame_b_scored"], 3.0)
    # 볼츠 규약: pair_chains_iptm[scored][frame]
    assert approx(rows[0]["iptm_boltz_a_scored_in_b_frame"], 0.5 + 0.1 * 0)   # A scored, B frame
    assert approx(rows[0]["iptm_boltz_b_scored_in_a_frame"], 0.5 + 0.1 * 1)   # B scored, A frame
    print("  [ok] chain_pair_table (방향 규약 포함)")


def test_target_only_guards():
    import prepare_input as P

    yaml_text, meta = P.build_job([{"header": "t", "sequence": "MKT" * 30}], [], "t", "server")
    assert meta["nanobody_chain"] is None and meta["target_only"] is True
    assert "constraints:" not in yaml_text
    try:
        P.build_job([{"header": "t", "sequence": "MKT" * 30}], [], "t", "server", hotspot_spec="5")
    except ValueError:
        pass
    else:
        raise AssertionError("target-only + hotspot 이 차단되지 않음")
    print("  [ok] target-only 가드 (hotspot 차단)")


def test_cdr_library_exhaustive(tmp=Path("/tmp/opencode/selftest_ex")):
    """exhaustive 모드가 특정 k 에 편향되지 않는지 (직접 생성 없이 열거 함수만)."""
    import make_cdr_library as L

    class R:
        def shuffle(self, x):
            pass

    pool = [("CDR1", i) for i in range(6)]
    seq = "A" * 20
    items = L.build_exhaustive(pool, seq, "ala", False, [1, 2], R(), cap=10 ** 6)
    ks = sorted({len(x) for x in items})
    assert ks == [1, 2], ks
    from math import comb
    assert len(items) == comb(6, 1) + comb(6, 2), len(items)
    print("  [ok] exhaustive 열거 편향 없음")


def test_interface_plddt_uses_passed_array():
    """R01 회귀: 인터페이스 평균 pLDDT 는 tokens(B-factor) 가 아니라 전달된 배열을 써야 한다."""
    tokens = []
    for ch in ("A", "B"):
        for i in range(3):
            tokens.append({"chain": ch, "resnum": i + 1, "resname": "ALA", "aa": "A",
                           "ca": np.array([0.0, 0.0, 0.0]), "cb": np.array([0.0, 0.0, 0.0]),
                           "plddt": 90.0, "n_atoms": 1})   # tokens 에는 90 이 들어있다
    pae = np.full((6, 6), 4.0)
    plddt = np.full(6, 10.0)                                # 전달 배열은 10
    rep = metrics.interface_report(tokens, plddt, pae, "A", "B", cutoff=8.0)
    assert approx(rep["plddt_mean_b"], 10.0), rep["plddt_mean_b"]
    print("  [ok] 인터페이스 pLDDT 는 전달 배열 사용 (R01)")


def test_ipsae_byres_guard():
    """R02 회귀: 빈 경로/디렉터리를 byres 로 주면 예외 없이 빈 목록."""
    assert metrics.parse_ipsae_byres("") == []
    assert metrics.parse_ipsae_byres(".") == []
    assert metrics.parse_ipsae_byres("/tmp") == []
    assert metrics.parse_ipsae_byres(None) == []
    print("  [ok] parse_ipsae_byres 가드 (R02)")


def test_hotspot_validation():
    """R10 회귀: 역방향/0 범위는 오류, 정상 범위는 파싱."""
    import prepare_input as P

    for bad in ("60-45", "0-5", "0"):
        try:
            P.parse_hotspot(bad, "A")
        except ValueError:
            continue
        raise AssertionError(f"잘못된 hotspot 이 통과됨: {bad}")
    assert P.parse_hotspot("45-47", "A") == [("A", 45), ("A", 46), ("A", 47)]
    assert P.parse_hotspot("A:5,B:7", "A") == [("A", 5), ("B", 7)]
    print("  [ok] hotspot 범위 검증 (R10)")


def test_library_size_exact():
    """R11 회귀: 선택지 수가 달라도 정확한 조합 수."""
    import make_cdr_library as L

    assert L.count_library_size([1, 2, 3], [3]) == 6          # 1*2*3
    assert L.count_library_size([19] * 25, [1]) == 475
    assert L.count_library_size([19] * 25, [2]) == 108300
    assert L.count_library_size([19] * 25, [1, 2, 3]) == 475 + 108300 + 15775700
    print("  [ok] variant 조합 수 정확 계산 (R11)")


def test_ipsae_max_key_normalization():
    """R04 회귀: 항원/나노바디 역할이 뒤바뀌어도 정렬된 키로 max ipSAE 를 찾는다."""
    from pathlib import Path as _P

    import analyze

    fake = {"max": {("B", "C"): {"ipsae": 0.9, "pdockq2": 0.5}},
            "pairs": {("C", "B"): {"ipsae": 0.8}}, "out_txt": None, "byres": None, "pml": None}
    orig = analyze.run_official_ipsae
    analyze.run_official_ipsae = lambda *a, **k: dict(fake)
    try:
        class Args:
            ipsae_script = "unused"
            pae_cutoff = 10.0
            dist_cutoff = 15.0
        model = {"pae": _P("/dev/null"), "cif": _P("/dev/null")}
        out = analyze._ipsae_from_official(model, Args(), None, None, ("C", "B"))
        assert out["max"]["ipsae"] == 0.9, out["max"]
        assert approx(out["asym_nb_frame_ag"]["ipsae"], 0.8)
    finally:
        analyze.run_official_ipsae = orig
    print("  [ok] max ipSAE 키 정규화 (R04)")


def test_cdr_library_partial_generation(tmp=Path("/tmp/opencode/selftest_cdr")):
    """RR07 회귀: mixed-k random 생성이 요청 수를 못 채우면 부분 성공으로 끝나지 않아야 한다."""
    import make_cdr_library as L

    tmp.mkdir(parents=True, exist_ok=True)
    orig_load, orig_pos = L.load_nanobody, L.get_cdr_positions
    L.load_nanobody = lambda p: {"header": "nb_test", "sequence": "G" * 20}
    L.get_cdr_positions = lambda seq, scheme="imgt": ({"CDR3": [0, 1, 2, 3]}, {})
    try:
        def run(n_variants, outdir):
            out = Path(outdir)
            out.mkdir(parents=True, exist_ok=True)
            sys.argv = ["make_cdr_library.py", "--nanobody", "/dev/null", "--target", "/dev/null",
                        "--cdrs", "CDR3", "--mode", "ala", "--n-mutations", "1,2",
                        "--n-variants", str(n_variants), "--seed", "1",
                        "--outdir", str(out), "--batch-out", str(out / "b.tsv")]
            try:
                rc = L.main()
            except SystemExit as exc:            # 사전검증 실패는 SystemExit 로 종료된다
                rc = exc.code if isinstance(exc.code, int) else 1
            rows = (out / "b.tsv").read_text(encoding="utf-8").strip().splitlines() if (out / "b.tsv").exists() else []
            return rc, len(rows)

        # 가능한 전체(4 + 6 = 10) 요청 -> 10개 모두 생성, 성공 종료
        rc, n_rows = run(10, tmp / "full")
        assert n_rows == 1 + 1 + 10, n_rows        # header + WT + 10
        assert rc == 0, rc
        # 공간(10)보다 1개 더 요청 -> 사전검증에서 실패(비정상 종료), 파일 미생성
        rc2, n_rows2 = run(11, tmp / "over")
        assert rc2 != 0, rc2
        assert n_rows2 == 0, n_rows2
    finally:
        L.load_nanobody, L.get_cdr_positions = orig_load, orig_pos
        sys.argv = sys.argv[:1]
    print("  [ok] mixed-k random 부분 생성 방지 (RR07)")


def test_display_formatter():
    """R6-01 회귀: 공통 표시 포맷터가 raw HTML 을 만들지 않아야 한다."""
    import make_report as M

    assert M.fmt("<b>x</b>") == "&lt;b&gt;x&lt;/b&gt;", M.fmt("<b>x</b>")
    assert M.fmt("<span data-x=\"1\">y</span>").count("<") == 0
    for bad in (float("nan"), float("inf"), float("-inf"), None, "nan", "inf", "abc<script>"):
        out = M.fmt(bad)
        assert "<" not in out, (bad, out)
    for bad in (float("nan"), "inf", "<i>1</i>", None):
        assert M.fmt_num(bad) == "-", (bad, M.fmt_num(bad))
    assert M.fmt(0.936) == "0.936"
    assert M.fmt_num(54, 0) == "54"
    print("  [ok] 표시 포맷터 HTML 경계 (R6-01)")


def test_render_sweep(run_dir, tmp=Path("/tmp/opencode/selftest_sweep")):
    """R7 회귀: 결과 JSON 의 모든 문자열에 marker 를 넣어 렌더링해도
    HTML 태그로 해석되는 값이 없어야 한다 (직접 보간 경로 검출)."""
    import shutil
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.hits = 0

        def handle_starttag(self, tag, attrs):
            if any(a.startswith("data-sweep") for a, _ in attrs):
                self.hits += 1

    marker = '<span data-sweep="1">X</span>'
    # R8-02: 문자열 키 + 수치 키 + 데이터 사전 키까지 치환해 직접 보간 경로를 넓게 검출한다.
    keys_str = {"id", "chain", "resname", "role", "label", "sub", "source", "job", "aa",
                "best_mapping_str", "capri", "name", "antigen_chain", "nanobody_chain",
                "chain_a", "chain_b", "notes", "headline", "message", "reference"}
    # 주의: index / best_model_index 는 표시 문자열이 아니라 구조적 식별자다.
    # make_report.validate_results() 가 예측 산출물과 모델 index 의 정합성을 int 로
    # 검증하므로, 렌더 스윕에서 마커로 치환하지 않는다.
    keys_num = {"n", "n_res", "n_contacts", "resnum", "ptm", "confidence_score",
                "best_dockq", "DockQ", "F1", "iRMSD", "LRMSD", "fnat", "fnonnat",
                "nat_correct", "nat_total", "clashes", "ipsae", "ipsae_d0chn", "ipsae_d0dom",
                "iptm_d0chn", "pdockq", "pdockq2", "lis", "pae_mean_ab", "pae_mean_ba",
                "plddt", "mean", "median", "min", "max", "frac_gt90", "frac_lt50",
                "rmsd", "auto_allowed_mismatches", "iptm_boltz_b_scored_in_a_frame",
                "iptm_boltz_a_scored_in_b_frame", "iface_pae_a_frame_b_scored",
                "iface_pae_b_frame_a_scored", "n_tokens"}
    # 키 자체를 marker 로 바꿀 데이터 사전 (키와 값을 같은 marker 로 맞춰 조회 일관성 유지)
    keyed_dicts = {"per_chain", "cdr_sequences"}

    def sweep(obj, parent_key=None):
        if isinstance(obj, dict):
            if parent_key in keyed_dicts:
                items = [(marker, v) for v in obj.values()]
                obj.clear()
                obj.update(dict(items))
            for k, v in list(obj.items()):
                if (isinstance(v, str) and k in keys_str) or (isinstance(v, (int, float)) and not isinstance(v, bool) and k in keys_num):
                    obj[k] = marker
                else:
                    sweep(v, k)
        elif isinstance(obj, list):
            for v in obj:
                sweep(v, parent_key)

    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(run_dir, tmp / "run")
    res = tmp / "run" / "analysis" / "results.json"
    data = json.loads(res.read_text(encoding="utf-8"))
    sweep(data)
    res.write_text(json.dumps(data), encoding="utf-8")
    out = tmp / "report"
    rc = subprocess.run([sys.executable, str(Path(__file__).parent / "make_report.py"),
                         "--results", str(res), "--outdir", str(out)],
                        capture_output=True, text=True, check=False)
    assert rc.returncode == 0, rc.stderr[-300:]
    html = (out / "index.html").read_text(encoding="utf-8")
    parser = _P()
    parser.feed(html)
    assert parser.hits == 0, f"marker 가 HTML 태그로 해석됨: {parser.hits}개"
    assert marker not in html, "marker 원문이 남아 있음"
    print(f"  [ok] 렌더 스윕 (문자열 marker -> 태그 0, escape {html.count('&lt;span data-sweep')}곳)")


def test_contact_vectorized_matches_bruteforce():
    """B-10: 벡터화된 contact_pairs 가 순수 python 결과와 동일해야 한다."""
    import numpy as np
    from metrics import contact_pairs
    n = 60
    tokens = [{"chain": "A" if i < n else "B",
               "ca": np.array([i * 0.7, 0.0, 0.0], dtype=np.float32),
               "cb": np.array([i * 0.7 + 0.3, 0.3, 0.0], dtype=np.float32)}
              for i in range(2 * n)]
    fast = {(i, j) for i, j, _ in contact_pairs(tokens, "A", "B", 8.0)}
    slow = set()
    for i, ti in enumerate(tokens):
        if ti["chain"] != "A":
            continue
        for j, tj in enumerate(tokens):
            if tj["chain"] != "B":
                continue
            if float(np.linalg.norm(ti["cb"] - tj["cb"])) <= 8.0:
                slow.add((i, j))
    assert fast == slow, "contact_pairs 벡터화 결과가 brute-force 와 다름"
    # 빈 체인도 조용히 빈 리스트
    assert contact_pairs(tokens, "A", "Z", 8.0) == []
    print(f"  [ok] contact_pairs 벡터화 == brute-force ({len(fast)}쌍)")


def test_json_safe_nonfinite_numpy():
    """B-7: np.floating NaN/inf 가 JSON 표준 null 로 변환되어야 한다."""
    import json

    import numpy as np
    from metrics import json_safe
    r = json_safe({"nan": np.float32("nan"), "inf": np.float64("inf"),
                   "ok": np.float32(0.25), "i": np.int32(7)})
    assert r["nan"] is None and r["inf"] is None
    assert r["ok"] == 0.25 and r["i"] == 7
    json.dumps(r)  # NaN 토큰이 있으면 여기서 실패
    print("  [ok] json_safe: np NaN/inf -> null")


def test_cdr_scheme_boundaries():
    """B-11: CDR 범위 정의가 스킴별로 단조 증가하고 AbM CDR2 경계가 맞아야 한다."""
    from metrics import CDR_RANGES_BY_SCHEME
    for scheme, rngs in CDR_RANGES_BY_SCHEME.items():
        for name, (a, b) in rngs.items():
            assert a < b, f"{scheme}/{name} 범위 역전"
        assert rngs["CDR1"][0] < rngs["CDR3"][0], scheme
    assert CDR_RANGES_BY_SCHEME["martin"]["CDR2"] == (50, 58)
    print("  [ok] CDR 스킴 경계 (martin CDR2=50-58)")


def test_chain_role_validation():
    """B-17 / M-13(정밀): 입력에 없는 체인을 지정하면 검증 코드에 실제로 도달해
    오류로 끝나야 한다. (기존 테스트는 지원하지 않는 --pred-dir 때문에 argparse
    단계에서 실패했고, usage 의 'BOLTZ_DIR' 에 포함된 'Z' 로 통과하는 위양성이었다.)"""
    from types import SimpleNamespace

    from analyze import resolve_roles

    tokens = [{"chain": "A", "aa": "E"} for _ in range(120)]
    tokens += [{"chain": "B", "aa": "G"} for _ in range(120)]
    args = SimpleNamespace(no_nanobody=False, nanobody_chain="Z", antigen_chain=None)
    try:
        resolve_roles(tokens, None, args)
    except SystemExit as e:
        assert "Z" in str(e), f"오류 메시지에 체인명이 없음: {e}"
    else:
        raise AssertionError("없는 체인 지정이 통과됨")
    print("  [ok] 체인 역할 검증 (없는 체인 -> SystemExit)")


def test_vhh_detection_min_score():
    """B-18: VHH 모티프가 없는 서열은 자동 판별하지 않아야 한다 (항원 오인 방지)."""
    from analyze import detect_nanobody_chain
    tokens = [{"chain": "A", "aa": "E"} for _ in range(80)]
    tokens += [{"chain": "B", "aa": "G"} for _ in range(120)]
    # A: 길이 미달(80), B: 길이만 맞고 VHH 특징(모티프/2C/WGQG) 없음 -> None
    assert detect_nanobody_chain(tokens) is None, "특징 없는 체인이 나노바디로 판별됨"
    print("  [ok] VHH 자동 판별 최소 점수")


def test_hotspot_rejects_nanobody_chain():
    """M-06(정밀): hotspot 은 항원에만. binder 자신의 체인을 지정하면 오류."""
    from prepare_input import build_job

    targets = [{"header": ">t", "sequence": "A" * 50}]
    nanobodies = [{"header": ">n", "sequence": ("EVQLV" * 24) + "WGQG"}]
    try:
        build_job(targets, nanobodies, "x", "server", hotspot_spec="B:1")
    except ValueError as e:
        assert "나노바디" in str(e), e
    else:
        raise AssertionError("나노바디 체인 hotspot 이 통과됨")
    # 항원 체인 hotspot 은 정상 통과
    yaml_text, meta = build_job(targets, nanobodies, "x", "server", hotspot_spec="1-3")
    assert "binder: B" in yaml_text and meta["hotspot"], meta
    print("  [ok] hotspot 항원 전용 검증 (binder 차단)")


def test_missing_user_msa_path_detected():
    """M-05(정밀): 경로처럼 보이는 MSA 지정이 없으면 조용히 캐시로 대체하지 않는다."""
    from msa_cache import is_user_msa, looks_like_msa_path

    assert looks_like_msa_path("definitely_missing_custom.csv")
    assert looks_like_msa_path("sub/dir/msa.a3m")
    assert not looks_like_msa_path("empty")
    assert not looks_like_msa_path("")
    assert not is_user_msa("definitely_missing_custom.csv")
    print("  [ok] 누락 사용자 MSA 경로 탐지")


def test_conservative_choices_match_selection():
    """M-07(정밀): conservative 정의가 없는 잔기(P)도 일반 풀로 폴백해 계산/선택 규칙이 일치."""
    import random

    from make_cdr_library import AA_NO_CYS, choices_for, pick_substitution

    c = choices_for("P", "conservative", allow_cys=False)
    assert c and "P" not in c and "C" not in c, c
    assert len(c) == len(set(c))
    assert set(c) <= set(AA_NO_CYS)
    for _ in range(10):
        assert pick_substitution("P", "conservative", random.Random(0), False) in c
    print("  [ok] conservative 폴백 규칙 일치 (P)")


def test_report_same_file_guard():
    """A-20: results.json 이 이미 outdir 안에 있어도 리포트 생성이 실패하지 않아야 한다."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "report"
        out.mkdir(parents=True)
        res = out / "results.json"
        pred = Path(td) / "predictions"
        pred.mkdir()
        stem = "selftest_model_0"
        cif = pred / f"{stem}.pdb"
        pae = pred / f"pae_{stem}.npz"
        plddt = pred / f"plddt_{stem}.npz"
        conf = pred / f"confidence_{stem}.json"
        cif.write_text(
            "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 80.00           C\n"
            "END\n",
            encoding="utf-8",
        )
        np.savez(pae, pae=np.array([[1.0]], dtype=float))
        np.savez(plddt, plddt=np.array([80.0], dtype=float))
        conf.write_text(json.dumps({"confidence_score": 0.5, "ptm": 0.4, "iptm": 0.7}),
                        encoding="utf-8")
        empty_model = {
            "model": "model_0", "stem": stem, "index": 0, "cif": str(cif),
            "pae_npz": str(pae), "confidence_json": str(conf), "chain_roles": {},
            "boltz": {"ptm": None, "confidence_score": None},
            "plddt": {"per_chain": {}, "mean": None},
            "pae": {"mean": None}, "boltz_pair_iptm": {"nanobody_in_antigen_frame": None,
                                          "antigen_in_nanobody_frame": None},
            "chain_pairs": [], "iptm_from_pae": {}, "interface_8A": {"residues_a": [], "residues_b": []},
            "interface_10A": {"residues_a": [], "residues_b": []}, "cdr": {"available": False},
            "ipsae": {"max": {}}, "dockq": None, "warnings": [],
        }
        res.write_text(json.dumps({"models": [empty_model], "sequences": {}, "chains": [],
                                   "settings": {"run_params": {"samples": 1}},
                                   "target_only": False, "warnings": [], "run_dir": td,
                                   "predictions_dir": str(pred),
                                   "run_name": "selftest",
                                   "nanobody_chain": None, "antigen_chain": None,
                                   "role_source": "selftest",
                                   "best_model_index": 0, "best_model_stem": stem,
                                   "antigen_chains": [],
                                   "created": "2026-01-01T00:00:00"}), encoding="utf-8")
        rc = subprocess.run([sys.executable, str(Path(__file__).parent / "make_report.py"),
                             "--results", str(res), "--outdir", str(out)],
                            capture_output=True, text=True, check=False)
        assert rc.returncode == 0, rc.stderr[-300:]
    print("  [ok] results.json 자기복사(SameFileError) 가드")


def main():
    if not __debug__:
        raise SystemExit("selftest requires assertions: run Python without -O/-OO or PYTHONOPTIMIZE")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()
    print("== Boltz-2 pipeline self-test ==")
    test_d0()
    test_tm_func()
    test_ipsae_pure_handcase()
    test_contact_and_interface()
    test_fasta_and_job()
    test_role_detection()
    test_chain_pair_table()
    test_interface_plddt_uses_passed_array()
    test_ipsae_byres_guard()
    test_hotspot_validation()
    test_library_size_exact()
    test_ipsae_max_key_normalization()
    test_cdr_library_partial_generation()
    test_display_formatter()
    test_target_only_guards()
    test_cdr_library_exhaustive()
    test_contact_vectorized_matches_bruteforce()
    test_json_safe_nonfinite_numpy()
    test_cdr_scheme_boundaries()
    test_chain_role_validation()
    test_vhh_detection_min_score()
    test_hotspot_rejects_nanobody_chain()
    test_missing_user_msa_path_detected()
    test_conservative_choices_match_selection()
    test_report_same_file_guard()
    if args.run_dir:
        test_run_dir(args.run_dir)
        test_render_sweep(args.run_dir)
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
