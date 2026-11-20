import numpy as np
import pytest
from didsuitepy import didsuite
from didsuitepy.data import prepare
from didsuitepy.inference import jackknife, randomized
from didsuitepy import assignment_set
from test_contract import fixture


def test_nearzero_jackknife_is_structured_failure():
    p = prepare(fixture(), y="y", time="year", cluster="state", gvar="g")
    def evaluator(data, native):
        return {"estimate": 2.0 + len(data.data)*1e-18}
    result = jackknife(p, evaluator, 2.0)
    assert result["status"] == "IDENTIFICATION_FAILURE"
    assert "roundoff" in result["failure_reason"]


def test_nonfinite_custom_deletion_is_recorded_without_optional_backend():
    prepared = prepare(fixture(), y="y", time="year", cluster="state", gvar="g")
    result = jackknife(prepared, lambda p, native: {"estimate": np.nan}, 1,
                       support="changing")
    assert result["status"] == "IDENTIFICATION_FAILURE"
    assert len(result["failures"]) == 8


def test_failed_randomization_is_not_dropped():
    p = prepare(fixture(), y="y", time="year", cluster="state", gvar="g")
    draws = assignment_set(p.cohorts, units=p.units, reps=3, seed=99)
    failed = draws.rows[1]
    def evaluator(data, native):
        if data.cohorts == failed:
            raise ValueError("required assignment fails")
        return {"estimate": sum((i+1)*g for i,g in enumerate(data.cohorts))}
    result = randomized(p, evaluator, draws, ["ri_beta"])["ri_beta"]
    assert result["p"] is None
    assert len(result["statistics"]) == 3
    assert len(result["failures"]) == 1


def test_numeric_assignment_order_is_numeric():
    data = fixture()
    data["state"] = data.state.str[1:].astype(int) + 7
    p = prepare(data, y="y", time="year", cluster="state", gvar="g")
    assert p.units == tuple(range(7,15))


@pytest.mark.integration
def test_harmonized_unsupported_does_not_look_compatible():
    result = didsuite(fixture(), y="y", time="year", cluster="state", gvar="g",
                      estimators="twfe", target_mode="harmonized", event_window=(0,1))
    assert result.summary.iloc[0].status == "UNSUPPORTED"
    assert result.summary.iloc[0].estimate is None
