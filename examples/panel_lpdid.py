from pathlib import Path
import pandas as pd
from didsuitepy import didsuite

data = pd.read_csv(Path(__file__).with_name("school_panel.csv"))
result = didsuite(data, y="test_score", time="year", unit="school", cluster="school",
                  gvar="first_treat", covariates=["baseline_score"], data_mode="panel",
                  estimators=["lpdid"], event_window=(0, 1), inference="standard")
columns = ["estimator", "estimate", "native_se", "jackknife_se", "status"]
print(result.summary[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
assert (result.summary.status == "PASS").all(), result.summary.failure_reason.to_list()
