import numpy as np
import pytest
from didsuitepy import didsuite
from didsuitepy.data import prepare
from didsuitepy.inference import jackknife
from test_contract import fixture

pytestmark = pytest.mark.integration


def run(data, method="none", **options):
    estimator_options = dict(method=method, **options.pop("csdid_options", {}))
    return didsuite(data, y="y", time="year", cluster="state", gvar="g", data_mode="rcs",
                    weights="w", covariates=[] if method == "none" else ["x"],
                    estimators="csdid", estimator_options={"csdid": estimator_options}, **options)


def standalone(data, method="none", **options):
    from fastcsdidpy import fit
    return fit(data, "y", "year", gvar="g", cluster="state", weights="w", method=method,
               covariates=[] if method == "none" else ["x"], data_mode="rcs", **options)


def singleton():
    data = fixture()
    data = data[data.state != "s1"].copy()
    return data.assign(w=1 + 0.2 * (data.year % 3) + 0.1 * (data.x + 1))


@pytest.mark.parametrize("method", ["none", "reg", "ipw", "dr"])
def test_default_matches_standalone_and_physical_refits(method):
    data = singleton()
    result = run(data, method)
    explicit = run(data, method, csdid_options={"jackknife_support": "changing"})
    fixed = run(data, method, csdid_options={"jackknife_support": "fixed"})
    row = result.summary.iloc[0]
    assert row.status == "PASS", row.failure_reason
    assert row.jackknife_support == "changing"
    assert fixed.summary.iloc[0].status == "IDENTIFICATION_FAILURE"
    for field in ("estimate", "native_se", "native_p"):
        assert row[field] == fixed.summary.iloc[0][field] == explicit.summary.iloc[0][field]
    jk = result.details["csdid"]["methods"]["jackknife"]
    assert sum(jk["support_changed"]) == 1
    expected = [standalone(data[data.state != h], method, inference="point")["estimate"]
                for h in jk["delete_clusters"]]
    np.testing.assert_allclose(jk["delete_estimates"], expected, atol=1e-12)
    native = standalone(data, method, inference="jackknife")
    assert row.jackknife_se == pytest.approx(native["jackknife"]["se"], abs=1e-12)
    assert row.jackknife_p == pytest.approx(native["jackknife"]["p"], abs=1e-12)
    assert row.jackknife_df == data.state.nunique() - 1




@pytest.mark.parametrize("policy", ["changing", "fixed"])
def test_undefined_deletions_fail_without_success_only_denominator(policy):
    data = fixture().assign(w=1.0)
    data = data[(data.g != 0) | (data.state == "s4")]
    result = run(data, csdid_options={"jackknife_support": policy})
    row = result.summary.iloc[0]
    assert row.status == "IDENTIFICATION_FAILURE"
    jk = result.details["csdid"]["methods"]["jackknife"]
    assert len(jk["delete_estimates"]) == data.state.nunique()
    assert jk["jackknife_df"] == data.state.nunique() - 1
    assert any(f["cluster"] == "s4" for f in jk["failures"])
    assert row.jackknife_p is None or np.isnan(row.jackknife_p)


def test_nonfinite_custom_deletion_is_a_recorded_failure():
    data = fixture()
    prepared = prepare(data, y="y", time="year", cluster="state", gvar="g")
    result = jackknife(prepared, lambda p, native: {"estimate": np.nan}, 1, support="changing")
    assert result["status"] == "IDENTIFICATION_FAILURE"
    assert len(result["failures"]) == 8


def test_invalid_policy_is_not_silently_ignored():
    result = run(singleton(), inference="point", csdid_options={"jackknife_support": "adaptive"})
    assert result.summary.iloc[0].status != "PASS"
    assert "jackknife_support" in result.summary.iloc[0].failure_reason




def test_fixed_compares_actual_support_not_omitted_cohort_labels():
    data = fixture().assign(w=1.0)
    data = data[data.state != "s3"]
    # Cohort 4 is now singleton but lies entirely outside event window (2, 2).
    # Its deletion does not change any required comparison.
    result = run(data, event_window=(2, 2), csdid_options={"jackknife_support": "fixed"})
    jk = result.details["csdid"]["methods"]["jackknife"]
    assert result.summary.iloc[0].status == "PASS"
    assert not jk["failures"]
    assert not any(jk["support_changed"])
