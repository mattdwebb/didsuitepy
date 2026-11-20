# Validation

Current checkpoint: 39 tests passed on Python 3.14, NumPy 2.5.3, pandas 3.0.6.

Tests cover multiset enumeration against brute force, huge-space deterministic
sampling, supplied-matrix rejection, CSV/hash round trips, absorbing-history gates,
panel restrictions, independent explicit-dummy OLS/CR1/jackknife, historical six-package
point/JK parity, assignment-by-assignment native and JK studentization, singleton
cohort failures, failure isolation, row-order invariance, and two independent processes.
Focused integration tests also cover CSDID, corrective Stacked, and both LP-DiD
weighting choices with and without covariates. LP-DiD uses genuine panel data;
RCS requests are explicitly unsupported. Near-zero jackknife denominators produce
an explicit inference failure instead of an unstable studentized statistic.

Student t probabilities were compared with base R `pt` for 42 t/df combinations.
The R process is validation-only; there is no production R dependency.

Run `python -m pytest -q`; set `RSCRIPT` to an installed executable for the probability
oracle. Install all nine estimator packages for the full integration tests.
Full native/harmonized support and cross-language harness
certification remain pending and are not implied by this checkpoint.
