import importlib
from importlib.metadata import version, PackageNotFoundError
import math
import numpy as np
from .data import COHORT
from .errors import UnsupportedError, IdentificationError

PACKAGES = {
    "twfe": "fasttwfepy", "did2s": "fastdid2spy", "bjs": "fastbjspy",
    "flex": "fastflexpy", "csdid": "fastcsdidpy", "undid": "undidpy",
    "didint": "didintpy", "stacked": "faststackedpy", "lpdid": "fastlpdidpy",
}
TARGETS = {
    "twfe": "TWFE coefficient", "did2s": "Static two-stage treated-observation effect",
    "bjs": "Imputation ATT", "flex": "FLEX aggregate effect",
    "csdid": "Group-time ATT aggregation", "undid": "UN-DID aggregate ATT",
    "didint": "Intersection parallel trends aggregate ATT",
    "stacked": "Trimmed aggregate ATT", "lpdid": "Pooled LP-DiD ATT",
}
LEGACY = {"twfe", "did2s", "bjs", "flex", "undid", "didint"}


def package_version(name):
    try:
        return version(PACKAGES[name])
    except PackageNotFoundError:
        return "not_installed"


def fit(name, prepared, options, *, native=False, event_window=None):
    try:
        module = importlib.import_module(PACKAGES[name])
    except ModuleNotFoundError as exc:
        raise UnsupportedError(f"{PACKAGES[name]} or a dependency is not installed: {exc.name}") from exc
    p = prepared
    if name in LEGACY:
        if event_window is not None:
            raise UnsupportedError("historical scalar adapter has no event-window restriction")
        if p.assignment_unit != p.cluster:
            raise UnsupportedError("historical Design conflates assignment and inference-cluster units")
        if p.mode == "panel" and p.unit != p.cluster:
            raise UnsupportedError("historical adapter uses cluster FE, not distinct individual-panel FE")
        if p.weights and name not in ("did2s", "flex"):
            raise UnsupportedError("historical adapter does not support observation weights")
        allowed = {"engine"} | ({"grouping"} if name == "flex" else set())
        if name == "undid":
            allowed.add("aggregation_weights")
        if set(options) - allowed:
            raise UnsupportedError(f"unsupported estimator options: {sorted(set(options) - allowed)}")
        kwargs = dict(outcome=p.y, cluster=p.cluster, time=p.time,
                      assignment=dict(zip(p.units, p.cohorts)), covariates=p.covariates)
        if p.weights:
            kwargs["weights"] = p.weights
        if "grouping" in options:
            kwargs["grouping"] = options["grouping"]
        if "aggregation_weights" in options:
            kwargs["aggregation_weights"] = options["aggregation_weights"]
        design = module.Design(p.data, **kwargs)
        result = {"estimate": float(design.point(engine=options.get("engine", "fast"))),
                  "target_label": TARGETS[name], "covariate_mode": "historical documented adjustment",
                  "cohorts_used": sorted(set(p.cohorts) - {0.0}),
                  "weighting_scheme": "native", "nobs": len(p.data),
                  "nclusters": p.data[p.cluster].nunique()}
        if hasattr(design, "last_support"):
            result["support"] = design.last_support
            result["cells_used"] = len(design.last_support)
            result["cells_eligible"] = sum(int(np.sum(design.times >= g)) for g in design.assignment if g>0)
            controls = [cell[2] for cell in design.last_support]
            result["min_controls"], result["max_controls"] = min(controls), max(controls)
            result["weighting_scheme"] = options.get("aggregation_weights", "post")
        if native:
            if not hasattr(design, "native_cluster"):
                raise UnsupportedError("no canonical native analytic VCE in historical implementation")
            vce = design.native_cluster(engine=options.get("engine", "fast"))
            for key in ("method", "se", "stat", "df", "p", "distribution"):
                result["native_" + key] = vce.get("native_cluster_" + key)
            # Historical FLEX calls its Student reference "t". Normalize the
            # adapter metadata so nonzero-null inference uses that same law.
            if result.get("native_distribution") == "t":
                result["native_distribution"] = "student_t"
            result["native_details"] = vce
    else:
        if not hasattr(module, "fit"):
            raise UnsupportedError("native package fit adapter is not available")
        options = dict(options)
        if set(options) & {"inference", "assignments", "ri_assignments", "reps", "ri_reps", "seed", "ri_seed", "null"}:
            raise UnsupportedError("inference and assignment options belong at harness level")
        csdid_weights = options.pop("aggregation_weights", "cohort") if name == "csdid" else None
        if name == "csdid" and csdid_weights not in ("cohort", "comparison"):
            raise UnsupportedError("CSDID aggregation_weights must be cohort or comparison")
        options["inference"] = "native" if native else "point"
        frame = p.data
        if name == "stacked":
            frame = p.data.copy()
            frame[COHORT] = frame[COHORT].replace(0, np.nan)
        kwargs = dict(y=p.y, time=p.time, unit=p.unit, gvar=COHORT,
                      cluster=p.cluster, covariates=list(p.covariates), data_mode=p.mode,
                      event_window=event_window, **options)
        if p.weights:
            kwargs["weights"] = p.weights
        if p.assignment_unit != p.cluster:
            raise UnsupportedError("separate assignment-unit routing not validated for this adapter")
        result = module.fit(frame, **kwargs)
        if name == "csdid":
            allow_partial = p.cluster_deletion and options.get("jackknife_support", "changing") == "changing"
            incomplete = not np.isfinite(result["group_time"].estimate).all()
            if incomplete and not allow_partial:
                raise IdentificationError("One or more required CSDID ATT(g,t) cells are unidentified")
            if csdid_weights == "comparison" or (incomplete and allow_partial):
                result = _comparison_csdid(result, frame, p, allow_partial=allow_partial,
                                           weighting=csdid_weights)
        if not isinstance(result, dict):
            raise TypeError("native fit must return a dictionary")
        metadata = result.get("metadata", {})
        result = dict(result)
        result.setdefault("nobs", result.get("n_used", result.get("n_obs", len(frame))))
        result.setdefault("nclusters", result.get("n_clusters", frame[p.cluster].nunique()))
        result.setdefault("cohorts_used", result.get("cohorts", sorted(set(p.cohorts)-{0.0})))
        result.setdefault("covariate_mode", metadata.get("covariate_timing", "see estimator specification"))
        result.setdefault("weighting_scheme", metadata.get("aggregation_weighting", "native"))
        result.setdefault("target_compatible", metadata.get("target_compatible", False))
        result.setdefault("native_method", metadata.get("native_vce"))
        result.setdefault("native_df", result.get("df"))
        if name == "csdid":
            result["jackknife_support"] = metadata["jackknife_support"]
            result["cells_used"] = int(np.isfinite(result["group_time"].estimate).sum())
            result["cells_eligible"] = len(result["group_time"])
            columns = ["group", "base", "post", "event"]
            result["support"] = tuple(result["group_time"].loc[np.isfinite(result["group_time"].estimate),columns].itertuples(index=False, name=None))
        if name == "stacked":
            result.setdefault("native_distribution", "student_t")
        if name == "lpdid":
            result["native_method"] = result.get("native_vce")
            result["native_distribution"] = "student_t" if result.get("native_df") else "normal"
            result["covariate_mode"] = result.get("covariate_timing")
            result["weighting_scheme"] = result.get("weighting")
            result["events"] = [dict(event_time=row["horizon"], **{k:v for k,v in row.items() if k != "horizon"})
                                for row in result.get("horizons", [])]
        if native and result.get("native_se") is not None and result["native_se"] > 0:
            result.setdefault("native_stat", result["estimate"] / result["native_se"])
        if native and result.get("native_se") is None:
            raise UnsupportedError("native package does not return a validated analytic SE")
    if not math.isfinite(result["estimate"]):
        raise IdentificationError("nonfinite estimate")
    if native and (not math.isfinite(result["native_se"]) or result["native_se"] <= 0):
        raise IdentificationError("zero or nonfinite native standard error")
    return result


def _comparison_csdid(result, frame, prepared, *, allow_partial=False, weighting="comparison"):
    """Reaggregate public CSDID cells using comparison-period treated mass.

    Include the estimated weights' contribution to the influence function.
    Nuisance fits and cell influence functions come from the canonical backend.
    """
    result = dict(result)
    cells = result["group_time"]
    ytime = frame[prepared.time].to_numpy()
    cohort = frame[COHORT].to_numpy()
    weight = frame[prepared.weights].to_numpy(dtype=float) if prepared.weights else np.ones(len(frame))
    mass = np.column_stack([weight * (cohort == cell.group) *
                           np.isin(ytime, [cell.base, cell.post])
                           for cell in cells.itertuples()])
    if weighting == "cohort":
        mass = np.column_stack([weight * (cohort == cell.group) for cell in cells.itertuples()])
    totals = mass.sum(axis=0)
    theta_cells = cells.estimate.to_numpy()
    valid = np.isfinite(theta_cells) & (totals > 0)
    if not valid.all() and not allow_partial:
        raise IdentificationError("One or more required CSDID ATT(g,t) cells are unidentified")
    if not valid.any():
        raise IdentificationError("No estimable CSDID cells remain in the deleted sample")
    records, influences, weights = [], [], []
    for row in result["aggregation"].itertuples():
        selected = np.ones(len(cells), dtype=bool)
        column = {"group": "group", "calendar": "post", "dynamic": "event"}.get(row.type)
        if column:
            selected = cells[column].to_numpy() == row.index
        active = valid & selected
        if not active.any():
            records.append(dict(label=row.label,type=row.type,index=row.index,estimate=np.nan))
            influences.append(np.full(len(frame),np.nan)); weights.append(np.full(len(cells),np.nan))
            continue
        raw = np.where(active, totals, 0.)
        w = raw/raw.sum()
        theta = w[active] @ theta_cells[active]
        influence = result["influence_att"][:,active] @ w[active]
        influence += len(frame)*(mass[:,active] @ (theta_cells[active]-theta))/raw.sum()
        records.append(dict(label=row.label,type=row.type,index=row.index,estimate=theta))
        influences.append(influence);weights.append(w)
    import pandas as pd
    aggregate = pd.DataFrame(records)
    influence = np.column_stack(influences)
    joint = np.column_stack((result["influence_att"],influence))
    clusters = frame[prepared.cluster].to_numpy()
    cluster_if = np.stack([joint[clusters==c].sum(axis=0) for c in result["cluster_ids"]])
    covariance = cluster_if.T@cluster_if/len(frame)**2
    j = len(cells); theta = float(aggregate.estimate.iloc[0]); se = math.sqrt(float(covariance[j,j]))
    result.update(estimate=theta,att=theta,status="pass",native_se=se,native_stat=theta/se if se else float("nan"),
                  native_p=math.erfc(abs(theta/se)/math.sqrt(2)) if se else float("nan"),
                  aggregation=aggregate,aggregation_weights=np.stack(weights),
                  influence_aggregate=influence,cluster_influence=cluster_if,covariance=covariance,
                  target_label=("comparison-period treated-mass" if weighting == "comparison" else "cohort-mass") + " weighted group-time ATT")
    result["metadata"] = dict(result["metadata"], aggregation_weighting=weighting,
                             deleted_sample_partial_support=bool(not valid.all()))
    return result
