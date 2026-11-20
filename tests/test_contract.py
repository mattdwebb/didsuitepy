from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import math
import os
import subprocess
import numpy as np
import pandas as pd
import pytest
from didsuitepy import assignment_set, didsuite
from didsuitepy.data import prepare
from didsuitepy.probability import student_two_sided


def fixture():
    rows = []
    for s, g in enumerate([3, 3, 4, 4, 0, 0, 0, 0]):
        for t in range(1, 6):
            for individual in range(2 + s % 3):
                d = int(g > 0 and t >= g)
                x = math.cos(0.7 * s + t + individual)
                y = 0.3 * s + 0.2 * t + d * (1.2 + 0.05 * s) + 0.15 * math.sin(s*t + individual)
                rows.append(dict(state=f"s{s}", year=t, id=f"s{s}-{individual}", g=g, D=d, y=y, x=x))
    return pd.DataFrame(rows)


def run_twfe(_=None):
    return didsuite(fixture(), y="y", time="year", cluster="state", gvar="g",
                    estimators="twfe", inference=["ri_beta"], ri_reps=3, seed=87).summary.to_json()


def test_assignments_exact_against_bruteforce():
    observed = (3, 3, 0, 0)
    result = assignment_set(observed, exact=True)
    expected = set(itertools.permutations(observed)) - {observed}
    assert set(result.rows) == expected
    assert result.space == 6


def test_large_assignment_space_deterministic():
    observed = (3,) * 30 + (4,) * 30 + (0,) * 40
    a = assignment_set(observed, reps=101, seed=41)
    b = assignment_set(observed, reps=101, seed=41)
    assert a.rows == b.rows
    assert a.space > 2**64
    assert len(set(a.rows)) == 101


@pytest.mark.parametrize("rows", [[(3, 3, 0, 0)], [(3, 0, 3, 0)] * 2, [(3, 0, 0, 0)], []])
def test_invalid_supplied_assignments(rows):
    with pytest.raises(ValueError):
        assignment_set((3, 3, 0, 0), supplied=rows)


def test_assignment_csv_roundtrip(tmp_path):
    a = assignment_set((3, 3, 0, 0), units=("A", "B", "C", "D"), exact=True)
    path = tmp_path / "assignments.csv"
    a.to_csv(path)
    b = assignment_set(a.observed, units=a.units, supplied=pd.read_csv(path))
    assert a.hash == b.hash
    with pytest.raises(ValueError, match="columns"):
        assignment_set(a.observed, units=a.units, supplied=b.to_frame().iloc[:, ::-1])


def test_treatment_and_cohort_equivalent():
    kwargs = dict(y="y", time="year", cluster="state")
    a = prepare(fixture(), gvar="g", **kwargs)
    b = prepare(fixture(), treatment="D", **kwargs)
    assert a.cohorts == b.cohorts
    pd.testing.assert_frame_equal(a.data, b.data)


@pytest.mark.parametrize("change", ["reversal", "inconsistent", "missing", "nonbinary", "varying_cohort"])
def test_invalid_histories(change):
    data = fixture()
    if change == "reversal":
        data.loc[(data.state == "s0") & (data.year == 5), "D"] = 0
    elif change == "inconsistent":
        data.loc[data.state == "s0", "g"] = 4
    elif change == "missing":
        data.loc[0, "y"] = np.nan
    elif change == "nonbinary":
        data.loc[0, "D"] = 2
    else:
        data.loc[0, "g"] = 4
    with pytest.raises(ValueError):
        prepare(data, y="y", time="year", cluster="state", treatment="D", gvar="g")


def test_panel_validation_and_legacy_scope():
    p = prepare(fixture(), y="y", time="year", cluster="state", unit="id", gvar="g")
    assert p.mode == "panel"
    result = didsuite(fixture(), y="y", time="year", cluster="state", unit="id", gvar="g", estimators="twfe")
    assert result.summary.iloc[0].status == "UNSUPPORTED"
    with pytest.raises(ValueError):
        prepare(fixture(), y="y", time="year", cluster="state", gvar="g", data_mode="panel")


@pytest.mark.integration
def test_twfe_against_independent_dummy_ols():
    data = fixture()
    X = np.column_stack([pd.get_dummies(data.state, dtype=float),
                         pd.get_dummies(data.year, drop_first=True, dtype=float), data.D])
    beta = np.linalg.lstsq(X, data.y, rcond=None)[0]
    error = data.y - X @ beta
    direction = np.linalg.solve(X.T @ X, np.eye(X.shape[1])[:, -1])
    scores = pd.Series((X @ direction) * error).groupby(data.state).sum().to_numpy()
    G, N, K = data.state.nunique(), len(data), X.shape[1]
    se = np.linalg.norm(scores) * np.sqrt(G/(G-1) * (N-1)/(N-K))
    result = didsuite(data, y="y", time="year", cluster="state", gvar="g", estimators="twfe")
    row = result.summary.iloc[0]
    assert row.status == "PASS", row.failure_reason
    assert row.estimate == pytest.approx(beta[-1], abs=1e-11)
    assert row.native_se == pytest.approx(se, abs=1e-11)
    deletes = []
    for state in sorted(data.state.unique()):
        d = data.loc[data.state != state]
        z = np.column_stack([pd.get_dummies(d.state, dtype=float), pd.get_dummies(d.year, drop_first=True, dtype=float), d.D])
        deletes.append(np.linalg.lstsq(z, d.y, rcond=None)[0][-1])
    expected = np.sqrt((G-1)/G * np.sum((np.array(deletes) - np.mean(deletes))**2))
    assert row.jackknife_se == pytest.approx(expected, abs=1e-11)
    assert row.jackknife_df == G - 1


@pytest.mark.parametrize("name", ["twfe", "did2s", "bjs", "flex", "undid", "didint"])
@pytest.mark.integration
def test_historical_point_and_jk_parity(name):
    from didsuitepy.adapters import PACKAGES
    import importlib
    data = fixture()
    design = importlib.import_module(PACKAGES[name]).Design(data, outcome="y", time="year", cluster="state", assignment=data.groupby("state").g.first().to_dict())
    result = didsuite(data, y="y", time="year", cluster="state", gvar="g", estimators=name)
    row = result.summary.iloc[0]
    assert row.status == "PASS", row.failure_reason
    assert row.estimate == pytest.approx(design.point(), abs=1e-10)
    assert row.jackknife_se == pytest.approx(design.jackknife()["standard_error"], abs=1e-10)


@pytest.mark.integration
def test_every_randomized_native_se_recomputed():
    import fasttwfepy
    data = fixture()
    result = didsuite(data, y="y", time="year", cluster="state", gvar="g", estimators="twfe", inference="full", ri_reps=7, seed=2026)
    assert result.summary.iloc[0].status == "PASS"
    native_stats, beta_stats = [], []
    for row in result.assignments.rows:
        design = fasttwfepy.Design(data, outcome="y", cluster="state", time="year", assignment=dict(zip(result.assignments.units, row)))
        native = design.native_cluster()
        native_stats.append(native["estimate"] / native["native_cluster_se"])
        beta_stats.append(design.point())
    for method, stats in [("ri_beta", beta_stats), ("ri_t_native", native_stats)]:
        detail = result.details["twfe"]["methods"][method]
        np.testing.assert_allclose(detail["statistics"], stats, atol=1e-9, rtol=1e-9)
        hits = sum(abs(t) >= abs(detail["observed_statistic"]) - 1e-12 for t in stats)
        assert detail["p"] == (1+hits)/(len(stats)+1)


@pytest.mark.integration
def test_singleton_cohort_deletion_refits_the_surviving_att():
    data = fixture()
    data.loc[data.state == "s1", "g"] = 0
    opts = dict(y="y", time="year", cluster="state", gvar="g", estimators="twfe")
    result = didsuite(data, **opts)
    jk = result.details["twfe"]["methods"]["jackknife"]
    assert jk["status"] == "PASS" and jk["jackknife_df"] == 7
    expected = [didsuite(data.loc[data.state != state], inference="point", **opts).summary.estimate.iloc[0]
                for state in jk["delete_clusters"]]
    np.testing.assert_allclose(jk["delete_estimates"], expected)
    assert jk["support_changed"][jk["delete_clusters"].index("s0")]
    assert np.isclose(jk["jackknife_se"], np.sqrt(7/8)*np.linalg.norm(expected-np.mean(expected)))


@pytest.mark.integration
def test_failure_isolation_and_no_silent_weights():
    data = fixture().assign(w=1.0)
    result = didsuite(data, y="y", time="year", cluster="state", gvar="g", weights="w", estimators=["twfe", "did2s"], inference="point")
    assert list(result.summary.status) == ["UNSUPPORTED", "PASS"]
    assert json.loads(result.to_json())[0]["estimate"] is None


@pytest.mark.integration
def test_row_order_and_seed_invariance():
    data = fixture()
    kwargs = dict(y="y", time="year", cluster="state", gvar="g", estimators="twfe", inference="full", ri_reps=3, seed=21)
    a = didsuite(data, **kwargs)
    b = didsuite(data.sample(frac=1, random_state=99), **kwargs)
    assert a.assignments.hash == b.assignments.hash
    for column in ["estimate", "native_se", "jackknife_se", "ri_beta_p", "ri_t_native_p"]:
        assert a.summary.iloc[0][column] == pytest.approx(b.summary.iloc[0][column], abs=1e-10)


@pytest.mark.integration
def test_process_isolation():
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_twfe, range(2)))
    rows = [pd.read_json(__import__("io").StringIO(value)).iloc[0] for value in results]
    for column in ("estimate", "ri_beta_p", "assignment_hash"):
        assert rows[0][column] == rows[1][column]


def test_student_t_against_r():
    rscript = os.environ.get("RSCRIPT")
    if not rscript:
        pytest.skip("RSCRIPT not configured for independent probability oracle")
    df = [1, 2, 3, 7, 30, 127, 1000]
    ts = [0.01, 0.5, 1, 2, 5, 20]
    expression = "cat(format(as.vector(outer(c(.01,.5,1,2,5,20),c(1,2,3,7,30,127,1000),function(t,d) 2*pt(-abs(t),d))),digits=17),sep='\\n')"
    output = subprocess.check_output([rscript, "--vanilla", "-e", expression], text=True)
    expected = np.array([float(v) for v in output.split()])
    actual = np.array([student_two_sided(t, d) for d in df for t in ts])
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-14)


@pytest.mark.integration
@pytest.mark.parametrize('estimator', ['twfe', 'did2s', 'bjs', 'flex'])
def test_native_nonzero_null_recomputes_statistic_and_p(estimator):
    row=didsuite(fixture(), y='y', time='year', cluster='state', gvar='g',
                 estimators=estimator, inference='native', null=1).summary.iloc[0]
    assert row.status == 'PASS'
    statistic=(row.estimate-1)/row.native_se
    expected=(math.erfc(abs(statistic)/math.sqrt(2)) if pd.isna(row.native_df)
              else student_two_sided(statistic,row.native_df))
    assert row.native_stat == pytest.approx(statistic)
    assert row.native_p == pytest.approx(expected)
