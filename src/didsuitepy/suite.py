from dataclasses import dataclass
import json
import math
import time as clock
import numpy as np
import pandas as pd
from . import adapters
from .assignments import assignment_set
from .data import prepare
from .errors import IdentificationError, UnsupportedError
from .inference import jackknife, randomized
from .probability import student_two_sided

FIELDS = "estimator language estimator_version target_label target_mode estimate native_method native_se native_stat native_df native_p jackknife_se jackknife_t jackknife_df jackknife_p jackknife_support ri_beta_p ri_t_native_p ri_reps assignment_space assignment_hash nobs nclusters n_treated_clusters cohorts_used control_group data_mode covariate_mode weighting_scheme event_window runtime_setup runtime_point runtime_native runtime_jk runtime_ri runtime_total status failure_reason".split()


@dataclass
class SuiteResult:
    summary: pd.DataFrame
    details: dict
    assignments: object = None
    events: pd.DataFrame | None = None

    def to_csv(self, path):
        self.summary.to_csv(path, index=False)

    def to_json(self, path=None):
        # pandas normalizes NaN to JSON null in machine-readable summaries.
        result = self.summary.to_json(orient="records", indent=2)
        if path is not None:
            from pathlib import Path
            Path(path).write_text(result + "\n", encoding="utf-8")
        return result

    def __str__(self):
        columns = ["estimator", "estimate", "native_p", "jackknife_p", "ri_beta_p", "ri_t_native_p", "status", "target_label"]
        table = self.summary[columns].astype(object)
        return table.where(pd.notna(table), "N/A").to_string(index=False)


def _methods(inference):
    modes = {"point": [], "standard": ["native", "jackknife"],
             "full": ["native", "jackknife", "ri_beta", "ri_t_native"]}
    if isinstance(inference, str):
        inference = modes.get(inference, [inference])
    result = tuple(dict.fromkeys(inference))
    if set(result) - set(modes["full"]):
        raise ValueError("unknown inference method")
    return result


def _failure(exc):
    reason = getattr(exc, "reason", None)
    if reason in ("numerical_failure", "rank_deficient", "zero_standard_error"):
        return "NUMERICAL_FAILURE"
    if reason in ("invalid_input", "invalid_timing_assignment"):
        return "INVALID_INPUT"
    if isinstance(exc, (UnsupportedError, NotImplementedError)) or str(exc).startswith("UNSUPPORTED"):
        return "UNSUPPORTED"
    if isinstance(exc, IdentificationError) or hasattr(exc, "reason") or str(exc).startswith("IDENTIFICATION_FAILURE"):
        return "IDENTIFICATION_FAILURE"
    return "NUMERICAL_FAILURE"


def didsuite(data, *, y, time, unit=None, group=None, gvar=None, treatment=None,
             cluster=None, assignment_unit=None, covariates=(), weights=None,
             data_mode="auto", estimators="all", target_mode="native",
             sample_mode="native", inference="standard", event_window=None,
             control_group="native", estimator_options=None, ri_reps=999, seed=0,
             assignments=None, exact=False, null=0):
    start = clock.perf_counter()
    selected = list(adapters.PACKAGES) if estimators == "all" else ([estimators] if isinstance(estimators, str) else list(estimators))
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(adapters.PACKAGES):
        raise ValueError("select unique estimator names from the nine supported families")
    methods = _methods(inference)
    if target_mode not in ("native", "harmonized") or sample_mode not in ("native", "common"):
        raise ValueError("unknown target/sample mode")
    if not np.isfinite(null):
        raise ValueError("null must be finite")
    if target_mode == "harmonized" and event_window is None:
        raise ValueError("harmonized mode requires an explicit event window")
    options = estimator_options or {}
    if set(options) - set(selected):
        raise ValueError("options supplied for an unselected estimator")
    prepared = prepare(data, y=y, time=time, unit=unit, group=group, cluster=cluster,
                       assignment_unit=assignment_unit, treatment=treatment, gvar=gvar,
                       covariates=covariates, weights=weights, data_mode=data_mode)
    ri_methods = tuple(m for m in methods if m in ("ri_beta", "ri_t_native"))
    draws = assignment_set(prepared.cohorts, units=prepared.units, reps=ri_reps, seed=seed,
                           exact=exact, supplied=assignments) if ri_methods else None
    setup = clock.perf_counter() - start
    rows, details, events = [], {}, []
    for name in selected:
        tick = clock.perf_counter()
        row = dict.fromkeys(FIELDS)
        row.update(estimator=name, language="Python", estimator_version=adapters.package_version(name),
                   target_label=adapters.TARGETS[name], target_mode=target_mode,
                   sample_mode=sample_mode, target_compatible=None, nobs=len(prepared.data),
                   nclusters=prepared.data[prepared.cluster].nunique(),
                   n_treated_clusters=prepared.data.loc[prepared.data["__didsuite_treatment__"] == 1, prepared.cluster].nunique(),
                   cohorts_used=sorted(set(prepared.cohorts) - {0.0}), control_group=control_group,
                   data_mode=prepared.mode, event_window=event_window, runtime_setup=setup,
                   covariates_requested=list(prepared.covariates), covariates_used=[],
                   assignment_unit=prepared.assignment_unit, inference_cluster=prepared.cluster,
                   assignment_cluster_distinct=prepared.assignment_unit != prepared.cluster,
                   ri_reps=len(draws.rows) if draws else 0,
                   assignment_space=draws.space if draws else None,
                   assignment_hash=draws.hash if draws else None, status="PASS", failure_reason=None)
        detail = {"methods": {}, "failures": []}
        native_options = dict(options.get(name, {}))
        routed_window = native_options.pop("event_window", event_window)
        if name == "stacked" and target_mode == "harmonized" and event_window[0] == 0:
            routed_window = (-1, event_window[1])
        if target_mode == "native":
            row["event_window"] = routed_window
        evaluator = lambda p, native: adapters.fit(name, p, native_options, native=native, event_window=routed_window)
        try:
            if target_mode == "harmonized" and "event_window" in options.get(name, {}):
                raise UnsupportedError("per-estimator window overrides are incompatible with a harmonized request")
            if name == "stacked" and target_mode == "harmonized" and event_window[0] != 0:
                raise UnsupportedError("stacked harmonized post window must start at event time zero")
            if sample_mode == "common":
                raise UnsupportedError("cross-estimator support intersection not yet certified")
            if control_group != "native":
                raise UnsupportedError("common control-group translation not yet certified; use explicit native estimator options")
            point_start = clock.perf_counter()
            point = evaluator(prepared, False)
            jk_support = point.get("jackknife_support", "changing")
            row["jackknife_support"] = jk_support
            row.update({key: point[key] for key in ("estimate", "target_label", "nobs", "nclusters", "cohorts_used", "covariate_mode", "weighting_scheme", "cells_used", "cells_eligible", "min_controls", "max_controls") if key in point})
            row["covariates_used"] = list(prepared.covariates)
            row["runtime_point"] = clock.perf_counter() - point_start
            row["target_compatible"] = point.get("target_compatible", False) if target_mode == "harmonized" else None
            detail["point"] = point
            if "events" in point:
                for event in point["events"]:
                    events.append(dict(estimator=name, **event))
            if "native" in methods:
                method_start = clock.perf_counter()
                try:
                    native = evaluator(prepared, True)
                    if null != 0:
                        statistic = (native["estimate"] - null) / native["native_se"]
                        distribution = native.get("native_distribution")
                        if distribution == "normal":
                            native["native_p"] = math.erfc(abs(statistic) / math.sqrt(2))
                        elif distribution == "student_t" and native.get("native_df"):
                            native["native_p"] = student_two_sided(statistic, native["native_df"])
                        else:
                            raise UnsupportedError("native nonzero-null reference distribution not specified")
                        native["native_stat"] = statistic
                    row.update({key: value for key, value in native.items() if key in FIELDS and key.startswith("native_")})
                    detail["methods"]["native"] = {"status": "PASS", "result": native}
                except Exception as exc:
                    detail["methods"]["native"] = {"status": _failure(exc), "reason": str(exc)}
                    if not isinstance(exc, UnsupportedError):
                        detail["failures"].append(str(exc))
                row["runtime_native"] = clock.perf_counter() - method_start
            if "jackknife" in methods:
                method_start = clock.perf_counter()
                jk = jackknife(prepared, evaluator, point["estimate"], null=null,
                               support=jk_support, full_support=point.get("support"))
                detail["methods"]["jackknife"] = jk
                row.update({key: value for key, value in jk.items() if key.startswith("jackknife_")})
                if jk["status"] != "PASS":
                    detail["failures"].append(jk["failure_reason"])
                row["runtime_jk"] = clock.perf_counter() - method_start
            for method in ri_methods:
                method_start = clock.perf_counter()
                # Unsupported native inference is N/A, never a fabricated p-value.
                if method == "ri_t_native":
                    try:
                        evaluator(prepared, True)
                    except UnsupportedError as exc:
                        detail["methods"][method] = {"status": "UNSUPPORTED", "reason": str(exc)}
                        continue
                result = randomized(prepared, evaluator, draws, (method,), null=null,
                                    jackknife_support=jk_support)[method]
                detail["methods"][method] = result
                row[method + "_p"] = result["p"]
                key = "runtime_ri"
                row[key] = (row[key] or 0) + clock.perf_counter() - method_start
                if result["status"] != "PASS":
                    detail["failures"].append(f"{method}: one or more required assignments failed")
            if detail["failures"]:
                row.update(status="IDENTIFICATION_FAILURE", failure_reason="; ".join(detail["failures"]))
        except Exception as exc:
            row.update(status=_failure(exc), failure_reason=str(exc))
        row["runtime_total"] = setup + clock.perf_counter() - tick
        rows.append(row)
        details[name] = detail
    return SuiteResult(pd.DataFrame(rows), details, draws, pd.DataFrame(events))
