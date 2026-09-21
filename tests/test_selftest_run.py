import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import selftest


class RunDirectorySelftestTests(unittest.TestCase):
    def run_fixture(self, recorded_iptm):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model = {"index": 0, "cif": run / "model.cif", "confidence": run / "confidence.json",
                     "pae": run / "pae.npz", "plddt": run / "plddt.npz"}
            model["confidence"].write_text(json.dumps({
                "complex_plddt": 0.8, "pair_chains_iptm": {"1": {"0": 0.8}},
            }))
            np.savez(model["pae"], pae=np.full((60, 60), 25.0))
            np.savez(model["plddt"], plddt=np.full(60, 0.8))
            (run / "analysis").mkdir()
            (run / "analysis/results.json").write_text(json.dumps({
                "nanobody_chain": "B", "antigen_chain": "A", "settings": {"no_figures": True},
                "models": [{"index": 0, "boltz_pair_iptm": {"nanobody_in_antigen_frame": recorded_iptm}}],
            }))
            tokens = [{"chain": chain} for chain in ["A"] * 30 + ["B"] * 30]
            with (
                patch("analyze.find_predictions_dir", return_value=run),
                patch("analyze.model_files", return_value=[model]),
                patch("selftest.metrics.read_structure_tokens", return_value=tokens),
                patch("selftest.metrics.run_official_ipsae", return_value={"max": {("A", "B"): {"ipsae": 0.0}}}),
            ):
                selftest.test_run_dir(run)

    def test_pae_proxy_need_not_equal_native_boltz_iptm(self):
        self.run_fixture(recorded_iptm=0.8)

    def test_recorded_native_iptm_must_match_confidence_json(self):
        with self.assertRaisesRegex(AssertionError, "recorded pair ipTM"):
            self.run_fixture(recorded_iptm=0.7)


if __name__ == "__main__":
    unittest.main()
