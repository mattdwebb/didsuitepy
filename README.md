# DiD Suite for Python

**Run several difference-in-differences estimators and compare their estimates
and uncertainty in one table.**

Suppose some states adopt a policy earlier than others. Different DiD estimators
use those observations in different ways, and their reported uncertainty can
also differ. DiD Suite gives them a common calling interface, reports their
results together, and can use the same treatment assignments for randomization
inference. Each estimator runs natively in Python; R and Stata are not required.

The suite does not make different estimands identical. Check the treatment
timing, comparison group and aggregation used by each method before interpreting
differences between rows.

## Install and get a first result

Use Python 3.10 or newer in a virtual environment. Install directly from GitHub; this package is not distributed on PyPI. Download the candidate source or clone the repository, then run:

```sh
git clone https://github.com/mattdwebb/didsuitepy.git
cd didsuitepy
python -m pip install "didsuitepy @ git+https://github.com/mattdwebb/didsuitepy.git"
python tools/install_estimators.py --minimal
python examples/quickstart.py
```

The helper installs the three backends needed for the first example, using the
exact revisions in [estimator-lock.json](estimator-lock.json). Omit `--minimal`
to install all nine. For an offline installation, first download each repository in the lock file
and put the checkouts in one directory, including `fastcsdid/python`. Then run:

```sh
python tools/install_estimators.py --from-siblings ..
```

That directory must contain the estimator repository folders. Use `--dry-run`
to inspect the installer’s arguments. A missing backend produces an explicit
unsupported row; installing the suite alone does not install every estimator.

## A school-policy example

The bundled data contain **960 pupil observations in 12 schools over eight
years**. Four schools adopt in 2013, four in 2015, and four never adopt. Pupils
are repeated cross sections, so the school is the inference cluster and there
is no individual panel ID. The outcome is a test score; the adjustment variable
is a predetermined baseline score. These are fictional teaching data.

The [data guide](examples/DATA_GUIDE.md) defines every column. In particular,
`first_treat` contains the adoption year on every row for a school, with zero
for never treated. The artificial policy adds two score points, alongside
school/year effects and varying disturbances.

```python
import pandas as pd
from didsuitepy import didsuite

data = pd.read_csv("examples/school_policy.csv")
result = didsuite(data, y="test_score", time="year", cluster="school",
                  gvar="first_treat", covariates=["baseline_score"],
                  data_mode="rcs", estimators=["twfe", "did2s", "bjs"],
                  inference="standard")
columns = ["estimator", "estimate", "native_se", "jackknife_se", "status"]
print(result.summary[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
```

Actual output:

```text
estimator  estimate  native_se  jackknife_se status
     twfe  2.012599   0.367339      0.401294   PASS
    did2s  2.316869   0.285375      0.336814   PASS
      bjs  2.316869   0.290170      0.336814   PASS
```

TWFE estimates an adjusted effect of **2.01 score points**; the two-stage and
imputation estimates are **2.32 points**. Here did2s and BJS use the same
untreated model and scalar aggregation, so their point estimates agree. Their
native covariance calculations differ. This equality is not promised for every
possible specification.

`native_se` is the estimator’s own clustered standard error. `jackknife_se`
comes from deleting schools one at a time and refitting. In this example all
12 deletions are estimable; standalone jackknife tests use t(11). The generator’s
known effect does not turn this small example into a test of inferential coverage.

Save the full result table with:

```python
result.to_csv("school_policy_comparison.csv")
```

`result.summary` is a pandas DataFrame. `result.details` contains estimator and
inference diagnostics; `print(result)` gives a wider comparison including p-values.

## Compare more estimators

After installing all backends, run `python examples/all_rcs.py`. It uses the same
data and covariates, regression adjustment for CSDID, and a stacked window from
one year before through one year after adoption:

```text
estimator  estimate  native_se  jackknife_se status
     twfe  2.012599   0.367339      0.401294   PASS
    did2s  2.316869   0.285375      0.336814   PASS
      bjs  2.316869   0.290170      0.336814   PASS
     flex  2.297807   0.303068      0.328889   PASS
    csdid  2.461347   0.501400      0.575392   PASS
    undid  2.391928        N/A      0.523918   PASS
   didint  2.394744        N/A      0.519843   PASS
  stacked  1.771243   0.251554      0.285132   PASS
```

UN-DID and DID-INT have no separate native analytic standard error for these
aggregates. Their N/A entries are intentional; the jackknife remains available.
For any requested inference method, inspect its status in
`result.details[estimator]["methods"]`. A summary row marked PASS can still have
N/A for a method that the estimator does not supply. Failed estimates and failed
required deletions are reported rather than removed from the comparison.

The supported families are:

| Name in `estimators` | What it computes | Main scope point |
|---|---|---|
| `twfe` | One regression with school/year fixed effects | Its coefficient need not be a common weighted ATT. |
| `did2s` | Untreated-outcome regression followed by an adjusted-outcome regression | Static scalar target. |
| `bjs` | Imputed untreated outcomes, then averaged treated effects | Static imputation aggregate. |
| `flex` | Flexible fitted treatment effects, then aggregation | School-group specification by default. |
| `csdid` | Cohort-time comparisons and their aggregate | Repeated-cross-section equations; none/reg/ipw/dripw methods. |
| `undid` | Adjusted school changes and comparisons | No separate native analytic covariance for this aggregate. |
| `didint` | School-year models standardized to treated covariates | No separate native analytic covariance for this aggregate. |
| `stacked` | Corrective-weighted cohort stacks | Choose an explicit window containing event time −1. |
| `lpdid` | Local projections around adoption | Genuine panels only; choose an explicit pooled window. |

LP-DiD is deliberately absent from the repeated-cross-section example. Run
`python examples/panel_lpdid.py` for a panel of the same schools’ annual means:

```text
estimator  estimate  native_se  jackknife_se status
    lpdid  1.601279   0.331234      0.365699   PASS
```

This panel example averages horizons zero and one. The panel unit is the school,
not a pupil whose identifier has been reused across years. CSDID’s supported
RCS implementation should likewise not be relabelled as a panel estimator.

## Covariates and inference

Omit `covariates` for an unadjusted specification. Estimator-specific choices
belong in `estimator_options`, for example
`{"csdid": {"method": "dripw"}, "stacked": {"event_window": (-1, 1)}}`.
Covariates are passed through or the unsupported request is reported; they are
not silently ignored. LP-DiD uses lag-one covariates by default.

Use `inference="standard"` for native and delete-cluster jackknife results.
Use `"point"` for estimates alone, or request coefficient and native-studentized
randomization tests explicitly. For example, using the already loaded data:

```python
randomized = didsuite(data, y="test_score", time="year", cluster="school",
                     gvar="first_treat", covariates=["baseline_score"],
                     estimators="twfe", inference=["ri_beta", "ri_t_native"],
                     ri_reps=99, seed=2026)
print(randomized.summary[["estimator", "ri_beta_p", "ri_t_native_p", "status"]])
```

The methods share cohort-count-preserving assignments. Studentized methods
recompute their standard error under every assignment. Use more assignments for
finer p-value resolution; the example’s 99 alternatives give a minimum p-value
of 0.01. Reassignment is justified only when the chosen assignment mechanism is
appropriate for the study. Studentization does not guarantee correct finite-sample
rejection rates.

When deleting a school removes a cohort, the supported aggregate and its weights
are recomputed over surviving estimable effects. All required school deletions
must produce an estimate. CSDID also offers an explicit fixed-support policy.
The [inference guide](INFERENCE.md) explains null hypotheses, ties, assignment
ordering, failure handling and reproducibility.

## Scope, help and reproducibility

The [validation record](RELEASE_VALIDATION.md) distinguishes local numerical tests, hosted CI, and optional integrations.

Use finite, nonmissing required input columns and absorbing treatment histories.
The common interface uses zero for never treated and translates backend-specific
coding internally. Separate assignment/inference units, automatic common-sample
intersection and general common-target harmonization are not supported. Not
every estimator supports every weighting rule or data structure. Native p-value
reference distributions are preserved rather than standardized by relabelling.

See the [method contract](METHOD_SPECIFICATION.md), [validation record](VALIDATION.md),
and linked backend guides for estimator references. Cite the estimators used and
record the suite and backend commits from the installation lock. Submit small,
reproducible examples through [GitHub issues](https://github.com/mattdwebb/didsuitepy/issues).

Core checks do not require optional estimators:

```sh
python -m pip install "didsuitepy[test] @ git+https://github.com/mattdwebb/didsuitepy.git"
python -m pytest -q -m "not integration"
```

After installing all nine pinned backends, `python -m pytest -q` also checks
their integration and all three README examples. Optional integration CI is separate from the local run with all nine backends installed; a skipped integration workflow is not numerical validation.

Original suite code is distributed under the [MIT license](LICENSE). Backend
licenses and third-party notices remain separate.


## Inference scope and deletion support

The suite provides coefficient RI (`ri_beta`) and native-studentized RI
(`ri_t_native`) in addition to native and standalone jackknife inference.
These tests are implemented by the suite. Standalone Fast CSDID intentionally
has a narrower public inference API, so its API checker rejects those option
names; that restriction does not remove the suite's separately tested methods.
See [jackknife support and identification](JACKKNIFE_SUPPORT.md) for surviving
ATT aggregation, full-sample failures, overlap limits, and interface differences.
