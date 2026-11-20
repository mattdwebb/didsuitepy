# Adapter specifications

The harness contains no substitute DiD estimator. It calls each same-language
package, preserving its target. Six historical packages use `Design(...).point()`
and `.native_cluster()` where defined. Point and inference definitions are pinned
by package provenance in the master certification registry, not by a shared VCE.

- TWFE: ordinary treatment coefficient with state/time FE; state CR1 and t(G-1).
- did2s: untreated first stage and corrected two-stage native sandwich; normal
  historical primary convention.
- BJS: imputation ATT and historical imputation clustered VCE; normal convention.
- FLEX: historical FLEX with unconditional composition and coefficient uncertainty.
- UN-DID/DID-INT: historical aggregation and covariate logic; native VCE N/A.
- CSDID/Stacked/LP-DiD: dedicated native `fit` adapters under integration validation.

The data gate verifies absorbing assignments and stable cohorts at assignment-unit
level, positive weights, complete requested fields, and panel nesting when applicable.
RCS mode never takes individual longitudinal differences. `auto` selects panel only
when a supplied panel identifier repeats; otherwise RCS.

Covariate transformations are estimator-specific and must be recorded. The harness
does not freeze time-varying covariates or infer a scientifically equivalent target.
