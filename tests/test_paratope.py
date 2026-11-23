from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze import analyze_model
from metrics import read_structure_tokens


@pytest.fixture
def cdr_annotation():
    return {
        "source": "ANARCI/IMGT",
        "available": True,
        "domain_span": [0, 4],
        "cdrs": {
            "CDR1": [{"seq_index": 0, "aa": "A"}],
            "CDR2": [{"seq_index": 2, "aa": "A"}],
            "CDR3": [{"seq_index": 3, "aa": "A"}],
        },
    }


def write_model(root, *, antigen_y, index=0):
    # Three equally buried surface patches on B: a CDR, framework, and tail.
    # Each contacting antigen chain covers a different patch; X is a spectator.
    coords = [("B", i, 20 * (i - 1), 0) for i in range(1, 6)]
    coords += [("A", 1, 0, antigen_y), ("C", 1, 20, antigen_y),
               ("C", 2, 80, antigen_y), ("X", 1, 40, 4)]
    stem = f"job_model_{index}"
    pdb = root / f"{stem}.pdb"
    pdb.write_text("".join(
        f"ATOM  {serial:5d}  CA  ALA {chain}{res:4d}    "
        f"{x:8.3f}{y:8.3f}{0:8.3f}  1.00 80.00           C\n"
        for serial, (chain, res, x, y) in enumerate(coords, 1)
    ) + "END\n")
    conf = root / f"confidence_{stem}.json"
    conf.write_text(json.dumps({"confidence_score": 0.7, "ptm": 0.6, "iptm": 0.5}))
    pae = root / f"pae_{stem}.npz"
    plddt = root / f"plddt_{stem}.npz"
    np.savez(pae, pae=np.ones((len(coords), len(coords))))
    np.savez(plddt, plddt=np.full(len(coords), 80.0))
    return {"index": index, "stem": stem, "cif": pdb, "confidence": conf, "pae": pae, "plddt": plddt}


def run_analysis(tmp_path, model, cdr, tokens=None):
    args = Namespace(reference=None, ipsae_script=None, pae_cutoff=10, dist_cutoff=15)
    if tokens is None:
        tokens = read_structure_tokens(model["cif"])
    # PAE scoring is deliberately inapplicable; geometry uses coordinates only.
    provenance = {"available": True, "pae_metrics_eligible": False}
    return analyze_model(model, tokens, ("A", "B", ["A", "C"]), args, cdr, tmp_path, provenance)


def test_paratope_partitions_one_vhh_side_across_union_of_antigen_chains(tmp_path, cdr_annotation):
    model = write_model(tmp_path, antigen_y=4)
    result = run_analysis(tmp_path, model, cdr_annotation).get("paratope_burial", {})
    assert result.get("status") == "ok"
    assert result["area_A2"]["CDR2"] == 0
    assert result["area_A2"]["CDR3"] == 0
    assert result["total_A2"] > 0
    for name in ("cdr_pct", "framework_pct", "unassigned_pct"):
        assert result[name] == pytest.approx(100 / 3)


def test_paratope_uses_each_models_coordinates_and_keeps_zero_area_unrated(tmp_path, cdr_annotation):
    model_0 = write_model(tmp_path, antigen_y=4, index=0)
    model_1 = write_model(tmp_path, antigen_y=30, index=1)
    tokens = read_structure_tokens(model_0["cif"])
    result = run_analysis(tmp_path, model_1, cdr_annotation, tokens).get("paratope_burial", {})
    assert result.get("status") == "no_interface"
    assert result["total_A2"] == 0
    assert result.get("cdr_pct") is None


def test_paratope_does_not_infer_framework_when_imgt_mapping_is_unavailable(tmp_path, cdr_annotation):
    model = write_model(tmp_path, antigen_y=4)
    cdr_annotation["cdrs"]["CDR1"][0]["aa"] = "G"
    result = run_analysis(tmp_path, model, cdr_annotation).get("paratope_burial", {})
    assert result.get("status") == "cdr_unavailable"
    assert result.get("framework_pct") is None
