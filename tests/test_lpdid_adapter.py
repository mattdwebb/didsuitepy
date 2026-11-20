import numpy as np
import pytest
from didsuitepy import didsuite
from test_contract import fixture


@pytest.mark.integration
@pytest.mark.parametrize("weighting", ["variance", "reweighted"])
@pytest.mark.parametrize("covariates", [[], ["x"]])
def test_lpdid_panel_adapter(weighting, covariates):
    lp = pytest.importorskip("fastlpdidpy")
    data = fixture()
    args = dict(y="y", time="year", unit="id", cluster="state", gvar="g",
                data_mode="panel", covariates=covariates)
    expected = lp.fit(data, **args, event_window=(0,1), weighting=weighting)
    result = didsuite(data, **args, estimators="lpdid", inference="full", ri_reps=2,
                      estimator_options={"lpdid": {"event_window": (0,1), "weighting": weighting}})
    row = result.summary.iloc[0]
    assert row.status == "PASS", row.failure_reason
    assert row.estimate == pytest.approx(expected["estimate"], abs=1e-11)
    assert row.native_se == pytest.approx(expected["native_se"], abs=1e-11)
    assert len(result.events) == 2
    for name in ["ri_beta_p", "ri_t_native_p"]:
        assert 0 < row[name] <= 1


@pytest.mark.integration
def test_lpdid_rcs_does_not_fake_panel():
    pytest.importorskip("fastlpdidpy")
    result = didsuite(fixture(), y="y", time="year", cluster="state", gvar="g",
                      data_mode="rcs", estimators="lpdid", inference="point",
                      estimator_options={"lpdid": {"event_window": (0,1)}})
    assert result.summary.iloc[0].status == "UNSUPPORTED"
