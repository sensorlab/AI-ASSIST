import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.service import bootstrap_risk_coverage_selected as brs


def _synthetic_df_covered(rng: np.random.Generator, n: int = 60) -> pd.DataFrame:
    # Heavy ties on purpose (few distinct values across many rows), so a coverage cutoff
    # is very likely to land inside a tied group and exercise the fractional path.
    states = [f"s{i}" for i in range(10)]
    return pd.DataFrame(
        {
            "state": rng.choice(states, size=n),
            "err": rng.random(n) * 5.0,
            "fake_metric": rng.choice([1.0, 2.0, 3.0], size=n),
        }
    )


class RunMetricSetTiePolicyOutputTests(unittest.TestCase):
    def _run(self, tie_policy: str) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        df_covered = _synthetic_df_covered(rng)
        metrics = (("fake_metric", False),)
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "out.csv"
            with (
                patch.object(brs, "N_BOOTSTRAP", 5),
                patch.object(brs, "_output_path", return_value=output_path),
            ):
                brs._run_metric_set(
                    df_covered,
                    dataset_name="unittest",
                    metric_set_name="main",
                    metrics=metrics,
                    tie_policy=tie_policy,
                )
            return pd.read_csv(output_path)

    def test_fractional_output_has_no_rmse_rows_and_all_finite_mae_naurc(self):
        out = self._run("fractional")
        self.assertNotIn("rmse", set(out["quantity"]))
        finite_quantities = out[out["quantity"].isin(["mae", "nAURC(MAE)"])]
        self.assertTrue(len(finite_quantities) > 0)
        self.assertTrue(np.isfinite(finite_quantities["point_estimate"]).all())
        self.assertTrue(np.isfinite(finite_quantities["ci_low"]).all())
        self.assertTrue(np.isfinite(finite_quantities["ci_high"]).all())
        self.assertTrue((out["tie_policy"] == "fractional").all())

    def test_hard_output_still_has_finite_rmse_rows(self):
        out = self._run("hard")
        rmse_rows = out[out["quantity"] == "rmse"]
        self.assertTrue(len(rmse_rows) > 0)
        self.assertTrue(np.isfinite(rmse_rows["point_estimate"]).all())


if __name__ == "__main__":
    unittest.main()
