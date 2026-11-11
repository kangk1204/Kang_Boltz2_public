import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import batch_report
import metrics
from msa_cache import read_msa_summary


def prepare(source, *options):
    return subprocess.run([sys.executable, str(ROOT / "scripts/prepare_batch.py"),
                           "--batch", str(source), "--outdir", str(source.parent / "gen"),
                           *options], text=True, capture_output=True, check=False)


def test_batch_inputs_and_failure_preserve_prepared_files(tmp_path):
    pairs = tmp_path / "pairs.fa"
    pairs.write_text(">WT\nQQQQ:AAAA\n>mut|binder=1\nCCCC:GGGG\n")
    args = ["--layout", "complex", "--check-sites", "A:2", "--samples", "3"]
    plan = prepare(pairs, *args, "--plan")
    assert plan.returncode == 0 and "요청 모델 6개" in plan.stdout
    assert not (tmp_path / "gen").exists()
    assert prepare(pairs, *args).returncode == 0
    root = tmp_path / "gen/batch_pairs"
    before = {p: p.read_bytes() for p in root.iterdir()}
    assert "sequence: CCCC" in (root / "mut.yaml").read_text()
    assert "constraints:" not in (root / "WT.yaml").read_text()
    assert json.loads((root / "mut.meta.json").read_text())["chains"][1]["sequence"] == "GGGG"
    pairs.write_text(">WT\nCCCC:AAAA\n>WT\nAA:GG\n")
    assert prepare(pairs, *args).returncode != 0
    assert all(p.read_bytes() == v for p, v in before.items())


def test_table_long_antigen_and_empty_input(tmp_path):
    source = tmp_path / "batch.tsv"
    header = "name\tantigen\tnanobody\n"
    source.write_text(header + "a\t" + "A" * 300 + "\t" + "Q" * 120 + "\n")
    assert prepare(source, "--plan").returncode == 0
    assert prepare(source, "--check-sites", "A:1-1000000000", "--plan").returncode != 0
    source.write_text(header)
    assert prepare(source).returncode != 0


def test_shared_target_and_large_library(tmp_path):
    target = tmp_path / "target.fa"
    target.write_text(">ag\n" + "A" * 300)
    lib = tmp_path / "lib.fa"
    lib.write_text("".join(f">nb_{i}\nGGG\n" for i in range(1000)))
    args = ["--layout", "binders", "--target", str(target), "--plan"]
    assert "요청 모델 1000개" in prepare(lib, *args).stdout
    assert prepare(lib, *args, "--check-sites", "B:1").returncode != 0


def test_msa_requires_query_first_and_correct_alignment(tmp_path):
    path = tmp_path / "msa.csv"
    for rows, expected in [("0,AAC\n1,AAA", False), ("0,AAA\n1,AA", False),
                           ("0,AAA\n1,AcA-", True)]:
        path.write_text("key,sequence\n" + rows)
        assert read_msa_summary(path, "AAA")[1] is expected


def test_site_contacts_use_input_position_and_distinguish_missing(monkeypatch):
    seq = "GC"
    tokens = [{"chain":"A", "aa":a, "resnum":45+i, "cb":1} for i, a in enumerate(seq)]
    tokens.append({"chain":"B", "aa":"Q", "resnum":1, "cb":1})
    monkeypatch.setattr(metrics, "read_structure_tokens", lambda _:tokens)
    result = {"antigen_chains":["A"], "nanobody_chain":"B", "chains":[{"id":"A", "sequence":seq}],
              "settings":{"effective_msa_policy":{"chains":[{"ids":["A"], "sequence_hash":
                  hashlib.sha256(seq.encode()).hexdigest()[:16]}]}}}
    model = {"cif":"saved", "pae_metrics_applicable":True, "interface_8A":{"residues_a":[{"token":1}]}}
    sites = batch_report.checked_contacts(result, model, "1,2,2")
    assert (sites["site_hits"], sites["contacted_sites"]) == (1, "A:2")
    assert batch_report.checked_contacts(result, model, "1")["site_hits"] == 0
    model["pae_metrics_applicable"] = False
    assert batch_report.checked_contacts(result, model, "1")["site_hits"] is None


def test_quoted_tsv_and_spreadsheet_safe_text(tmp_path):
    p = tmp_path / "summary.tsv"
    row = {"job":"=unsafe", "status":"미완료", "delta_reason":"a\tb\nc"}
    batch_report.write_summary_tsv(p, [row], [("job", "Job", "str", None, True)])
    with p.open(newline="") as f:
        saved, = csv.DictReader(f, delimiter="\t")
    assert saved == {"job":"'=unsafe","status":"미완료","delta_reason":"a\tb\nc"}
