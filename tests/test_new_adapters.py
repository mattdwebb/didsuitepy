import numpy as np
import pytest
from didsuitepy import didsuite
from test_contract import fixture


@pytest.mark.integration
def test_stacked_adapter_zero_never_translation_and_window():
    module = pytest.importorskip("faststackedpy")
    data = fixture()
    kwargs = dict(y="y", time="year", cluster="state", gvar="g", data_mode="rcs")
    expected = module.fit(data.assign(g=data.g.replace(0, np.nan)), **kwargs, event_window=(-1,1))
    result = didsuite(data, **kwargs, estimators="stacked", inference="standard",
                      estimator_options={"stacked": {"event_window": (-1,1)}})
    row = result.summary.iloc[0]
    assert row.status == "PASS", row.failure_reason
    assert row.estimate == pytest.approx(expected["estimate"], abs=1e-11)
    assert row.native_se == pytest.approx(expected["native_se"], abs=1e-11)
    assert row.nobs == expected["n_used"]


@pytest.mark.integration
def test_csdid_adapter_default_and_covariates():
    module = pytest.importorskip("fastcsdidpy")
    data = fixture()
    kwargs = dict(y="y", time="year", cluster="state", gvar="g", data_mode="rcs", covariates=["x"])
    expected = module.fit(data, **kwargs, method="dripw", inference="point")
    result = didsuite(data, **kwargs, estimators="csdid", inference="native")
    row = result.summary.iloc[0]
    assert row.status == "PASS", row.failure_reason
    assert row.estimate == pytest.approx(expected["estimate"], abs=1e-11)
    assert row.native_se == pytest.approx(expected["native_se"], abs=1e-11)
    assert row.covariates_used == ["x"]


@pytest.mark.integration
def test_nonzero_null_twfe_native_and_sharp_ri():
    data = fixture()
    kwargs = dict(y="y", time="year", cluster="state", gvar="g", estimators="twfe", inference="full", ri_reps=2, seed=33)
    shifted = data.assign(y=data.y-0.7*data.D)
    a = didsuite(data, **kwargs, null=0.7)
    b = didsuite(shifted, **kwargs)
    assert a.summary.iloc[0].status == "PASS"
    for method in ("ri_beta", "ri_t_native"):
        np.testing.assert_allclose(a.details["twfe"]["methods"][method]["statistics"],
                                   b.details["twfe"]["methods"][method]["statistics"])
