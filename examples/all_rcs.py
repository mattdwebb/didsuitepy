from pathlib import Path
import pandas as pd
from didsuitepy import didsuite

data = pd.read_csv(Path(__file__).with_name("school_policy.csv"))
result = didsuite(data, y="test_score", time="year", cluster="school",
                  gvar="first_treat", covariates=["baseline_score"], data_mode="rcs",
                  estimators=["twfe", "did2s", "bjs", "flex", "csdid", "undid", "didint", "stacked"],
                  inference="standard",
                  estimator_options={"csdid": {"method": "reg"},
                                     "stacked": {"event_window": (-1, 1)}})
columns = ["estimator", "estimate", "native_se", "jackknife_se", "status"]
print(result.summary[columns].to_string(index=False, na_rep="N/A", float_format=lambda x: f"{x:.6f}"))
assert (result.summary.status == "PASS").all(), result.summary.failure_reason.to_list()
for name in ["undid", "didint"]:
    assert result.details[name]["methods"]["native"]["status"] == "UNSUPPORTED"
