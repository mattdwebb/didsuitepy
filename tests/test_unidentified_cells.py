"""Full-sample failure is distinct from supported deletion aggregation."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from didsuitepy import didsuite
from didsuitepy.data import prepare
from didsuitepy.inference import jackknife

ROOT = Path(__file__).parent

def run(data, method, weighting, **kwargs):
    return didsuite(data,y="y",time="time",cluster="cluster",gvar="g",weights="w",
        covariates=["x","z"],data_mode="rcs",estimators="csdid",
        estimator_options={"csdid":dict(method=method,aggregation_weights=weighting,
            jackknife_support=kwargs.pop("support","changing"))},**kwargs)

@pytest.mark.integration
@pytest.mark.parametrize("method",["reg","dr"])
@pytest.mark.parametrize("weighting",["cohort","comparison"])
@pytest.mark.parametrize("inference",["point","standard"])
def test_unidentified_full_sample_does_not_pass(method,weighting,inference):
    pytest.importorskip("fastcsdidpy")
    out=run(pd.read_csv(ROOT/"unidentified_cell.csv"),method,weighting,inference=inference)
    row=out.summary.iloc[0]
    assert row.status=="IDENTIFICATION_FAILURE",row.to_dict()
    assert row.estimate is None or not np.isfinite(row.estimate)

@pytest.mark.integration
@pytest.mark.parametrize("method",["reg","dr"])
@pytest.mark.parametrize("weighting",["cohort","comparison"])
def test_partial_support_is_allowed_only_for_deletions(method,weighting):
    pytest.importorskip("fastcsdidpy")
    data=pd.read_csv(ROOT/"partial_deletion.csv")
    result=run(data,method,weighting)
    row=result.summary.iloc[0]
    assert row.status=="PASS",row.failure_reason
    jk=result.details["csdid"]["methods"]["jackknife"]
    assert len(jk["delete_estimates"])==13 and np.isfinite(jk["delete_estimates"]).all()
    assert jk["support_changed"][jk["delete_clusters"].index("13")]
    assert row.jackknife_df==12
    assert row.jackknife_se==pytest.approx(np.sqrt(12/13*np.sum(
        (np.array(jk["delete_estimates"])-np.mean(jk["delete_estimates"]))**2)))
    # The same reduced data, submitted as a full-sample fit, must fail.
    physical=run(data[data.cluster!=13],method,weighting,inference="point")
    assert physical.summary.iloc[0].status=="IDENTIFICATION_FAILURE"
    fixed=run(data,method,weighting,support="fixed")
    assert fixed.summary.iloc[0].status=="IDENTIFICATION_FAILURE"

def test_generic_jackknife_accepts_two_estimable_deletions():
    data=pd.DataFrame(dict(y=[0.,1.,2.,4.],time=[1,2,1,2],cluster=[1,1,2,2],g=[2,2,0,0]))
    p=prepare(data,y="y",time="time",cluster="cluster",gvar="g",data_mode="rcs")
    result=jackknife(p,lambda x,native:{"estimate":float(x.data.y.mean())},1.75)
    assert result["status"]=="PASS" and result["jackknife_df"]==1
