from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from didsuitepy import didsuite

@pytest.mark.integration
@pytest.mark.parametrize("method",["reg","dr"])
@pytest.mark.parametrize("control",["never","not_yet"])
@pytest.mark.parametrize("weighting",["cohort","comparison"])
def test_weighted_unbalanced_csdid_and_surviving_cohort(method,control,weighting):
    pytest.importorskip("fastcsdidpy")
    root=Path(__file__).parent;d=pd.read_csv(root/"csdid_weighting_data.csv");ref=pd.read_csv(root/"csdid_weighting_reference.csv")
    ref=ref[(ref.method==method)&(ref.control==control)&(ref.weighting==weighting)]
    z=didsuite(d,y="y",time="time",cluster="cluster",gvar="g",weights="w",covariates=["x","z"],data_mode="rcs",estimators="csdid",estimator_options={"csdid":dict(method=method,control_group=control,aggregation_weights=weighting)})
    row=z.summary.iloc[0];assert row.status=="PASS",row.failure_reason
    np.testing.assert_allclose([row.estimate,row.native_se,row.jackknife_se],ref.iloc[0][["estimate","native_se","jk_se"]].to_numpy(float),atol=1e-8,rtol=0)
    jk=z.details["csdid"]["methods"]["jackknife"]
    np.testing.assert_allclose(jk["delete_estimates"],ref.set_index("cluster").loc[[int(h) for h in jk["delete_clusters"]],"estimate"],atol=1e-8,rtol=0)
    assert row.jackknife_df==11 and sum(jk["support_changed"])==2
