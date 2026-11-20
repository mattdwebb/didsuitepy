import math
import numpy as np
from .errors import IdentificationError
from .probability import student_two_sided


def jackknife(prepared, evaluator, estimate, *, null=0, support="changing", full_support=None):
    if support not in ("changing", "fixed"):
        raise ValueError("jackknife support must be changing or fixed")
    clusters = tuple(prepared.data[prepared.cluster].unique())
    count = len(clusters)
    if count < 2:
        raise IdentificationError("jackknife requires at least two clusters")
    required_cohorts = set(prepared.cohorts) - {0.0}
    deleted, failures, support_changed = [], [], []
    for cluster in clusters:
        changed = None
        try:
            reduced = prepared.without_cluster(cluster)
            changed = (set(reduced.cohorts) - {0.0}) != required_cohorts
            if support == "fixed" and full_support is None and changed:
                raise IdentificationError("deletion removes a requested treatment cohort")
            if full_support is not None:
                changed = None  # Actual comparison support is unknown until the refit succeeds.
            fitted = evaluator(reduced, False)
            if full_support is not None:
                changed = fitted["support"] != full_support
            if support == "fixed" and changed:
                raise IdentificationError("deletion changed required ATT support under fixed policy")
            value = float(fitted["estimate"])
            if not math.isfinite(value):
                raise IdentificationError("deletion aggregate is unidentified")
            deleted.append(value)
        except Exception as exc:
            deleted.append(float("nan"))
            failures.append({"cluster": str(cluster), "reason": str(exc)})
        support_changed.append(changed)
    result = {"delete_clusters": list(map(str, clusters)), "delete_estimates": deleted,
              "jackknife_df": count - 1, "failures": failures,
              "jackknife_support": support, "support_changed": support_changed}
    if failures:
        result.update(status="IDENTIFICATION_FAILURE", failure_reason="one or more required deletions failed")
        return result
    variance = (count - 1) / count * np.sum((np.array(deleted) - np.mean(deleted)) ** 2)
    se = math.sqrt(float(variance))
    noise_floor = 64 * np.finfo(float).eps * max(1.0, abs(estimate), float(np.max(np.abs(deleted))))
    if se <= noise_floor or not math.isfinite(se):
        result.update(status="IDENTIFICATION_FAILURE", failure_reason="zero, roundoff-scale, or nonfinite jackknife variance")
        return result
    statistic = (estimate - null) / se
    result.update(status="PASS", jackknife_se=se, jackknife_t=statistic,
                  jackknife_p=student_two_sided(statistic, count - 1))
    return result


def randomized(prepared, evaluator, assignments, methods, *, null=0, jackknife_support="changing"):
    # A nonzero sharp null changes outcomes once, before every reassignment/refit.
    adjusted = prepared.sharp_null_adjusted(null) if null else prepared
    methods = tuple(methods)
    observed = {}
    stats = {method: [] for method in methods}
    failures = {method: [] for method in methods}

    def evaluate(p, method):
        result = evaluator(p, method == "ri_t_native")
        if method == "ri_beta":
            return result["estimate"]
        if method == "ri_t_native":
            return result["estimate"] / result["native_se"]
        raise ValueError("Unsupported randomization statistic")

    for method in methods:
        try:
            observed[method] = evaluate(adjusted, method)
        except Exception as exc:
            failures[method].append({"assignment": "observed", "reason": str(exc)})
    for index, row in enumerate(assignments.rows):
        reassigned = adjusted.reassigned(row)
        for method in methods:
            if method not in observed:
                continue
            try:
                value = float(evaluate(reassigned, method))
                if not math.isfinite(value):
                    raise IdentificationError("nonfinite randomized statistic")
                stats[method].append(value)
            except Exception as exc:
                stats[method].append(float("nan"))
                failures[method].append({"assignment": index, "reason": str(exc)})
    results = {}
    for method in methods:
        values = np.asarray(stats[method])
        if failures[method]:
            results[method] = dict(status="IDENTIFICATION_FAILURE", p=None,
                                   statistics=stats[method], failures=failures[method])
        else:
            threshold = abs(observed[method])
            tolerance = 64 * np.finfo(float).eps * np.maximum(1, np.maximum(np.abs(values), threshold))
            hits = int(np.count_nonzero(np.abs(values) >= threshold - tolerance))
            results[method] = dict(status="PASS", p=(1 + hits) / (len(values) + 1),
                                   hits=hits, observed_statistic=observed[method],
                                   statistics=stats[method], failures=[],
                                   tail="inclusive absolute; numerical ties within 64 eps * max(1, magnitudes)")
    return results
