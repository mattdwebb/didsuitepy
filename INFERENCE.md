# Inference contract

Jackknife deletes each complete inference cluster, rebuilds the design, requires
all G results, and uses mean-centered CV3J: `(G-1)/G * sum((b[-g]-mean(b[-g]))^2)`.
Student t has G-1 degrees of freedom. CSDID uses `jackknife_support="changing"`
by default: cohort masses and aggregation weights are recomputed, and a singleton
cohort may disappear if the reduced aggregate remains defined. Set
`estimator_options={"csdid": {"jackknife_support": "fixed"}}` to reject changes
in required group/base/post/event comparisons. Already-omitted cohorts are not
required comparison support. Neither policy freezes weights. Undefined deletions
invalidate inference under both policies. The selected policy and per-deletion
`support_changed` diagnostics are retained. Other routes also re-estimate the
supported aggregate on the remaining sample. A disappearing cohort alone is
not a failure when a surviving subaggregate is estimable; its weights are
recomputed. Every required scalar deletion must remain defined.

RI uses unique fixed-cohort-count assignments, excludes the observed assignment,
and samples without replacement via multiset ranks and Floyd's algorithm.
Exact enumeration is bounded at 100000 total assignments. Python integer ranks
permit spaces beyond 64-bit integers. Same seeds reproduce Python draws; supplied
matrices are the cross-language contract, not equal RNG seeds.

Tests use inclusive absolute tails and `(1+hits)/(R+1)`. Roundoff ties use
`64*machine_epsilon*max(1,abs(T_r),abs(T_obs))`, recorded in result details.
Any required invalid draw invalidates the method's p-value. No successful-draw-only
denominator is allowed. Nonzero RI nulls subtract the homogeneous sharp-null effect
times observed treatment from outcomes before observed and randomized refitting.

The assignment hash is SHA-256 of ASCII compact JSON for the array
`["didsuite-assignments-v1", unit_strings, observed_numeric_strings, row_numeric_strings]`.
Numbers use 17 significant digits; JSON uses ASCII escaping and no extra spaces.
CSV columns must match the recorded assignment-unit order exactly.
