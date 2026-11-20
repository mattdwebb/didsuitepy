from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

COHORT = "__didsuite_cohort__"
TREATMENT = "__didsuite_treatment__"


@dataclass(frozen=True)
class Prepared:
    data: pd.DataFrame
    y: str
    time: str
    unit: str | None
    assignment_unit: str
    cluster: str
    covariates: tuple
    weights: str | None
    mode: str
    units: tuple
    cohorts: tuple
    cluster_deletion: bool = False

    def reassigned(self, labels):
        labels = tuple(float(x) for x in labels)
        if len(labels) != len(self.units):
            raise ValueError("assignment length differs from assignment units")
        data = self.data.copy()
        data[COHORT] = data[self.assignment_unit].map(dict(zip(self.units, labels)))
        data[TREATMENT] = ((data[COHORT] > 0) & (data[self.time] >= data[COHORT])).astype(int)
        return replace(self, data=data, cohorts=labels)

    def without_cluster(self, label):
        data = self.data.loc[self.data[self.cluster] != label].copy()
        active = set(data[self.assignment_unit])
        keep = [(u, g) for u, g in zip(self.units, self.cohorts) if u in active]
        return replace(self, data=data, units=tuple(u for u, _ in keep),
                       cohorts=tuple(g for _, g in keep), cluster_deletion=True)

    def sharp_null_adjusted(self, null):
        data = self.data.copy()
        data[self.y] = data[self.y] - null * data[TREATMENT]
        return replace(self, data=data)


def prepare(data, *, y, time, unit=None, group=None, cluster=None,
            assignment_unit=None, treatment=None, gvar=None, covariates=(),
            weights=None, data_mode="auto"):
    if data_mode not in ("auto", "panel", "rcs"):
        raise ValueError("data_mode must be auto, panel, or rcs")
    if treatment is None and gvar is None:
        raise ValueError("provide treatment or gvar")
    assignment_unit = assignment_unit or group or cluster or unit
    cluster = cluster or assignment_unit
    if assignment_unit is None:
        raise ValueError("provide an assignment unit, group, cluster, or panel unit")
    if isinstance(covariates, str):
        covariates = (covariates,)
    covariates = tuple(covariates or ())
    if len(set(covariates)) != len(covariates):
        raise ValueError("duplicate covariates")
    frame = pd.DataFrame(data).copy()
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("require nonempty data with unique column names")
    if COHORT in frame or TREATMENT in frame:
        raise ValueError("reserved didsuite internal column present")
    columns = list(dict.fromkeys(c for c in
        (y, time, unit, assignment_unit, cluster, treatment, gvar, weights, *covariates)
        if c is not None))
    if frame[columns].isna().any().any():
        raise ValueError("missing values in requested variables; choose the sample explicitly")
    numeric = [y, time, *covariates] + ([weights] if weights else [])
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("outcome, time, covariates, and weights must be finite numeric values")
    frame[time] = frame[time].astype(float)
    if weights and (frame[weights] <= 0).any():
        raise ValueError("observation weights must be strictly positive")
    mode = data_mode
    if mode == "auto":
        mode = "panel" if unit and frame[unit].duplicated().any() else "rcs"
    if mode == "panel":
        if unit is None:
            raise ValueError("panel mode requires a unit identifier")
        if frame.duplicated([unit, time]).any():
            raise ValueError("panel unit-time observations must be unique")
        for column in (assignment_unit, cluster):
            if (frame.groupby(unit)[column].nunique() != 1).any():
                raise ValueError("panel units must remain nested in assignment units and clusters")
    try:
        units = tuple(sorted(frame[assignment_unit].unique()))
    except TypeError as exc:
        raise ValueError("assignment IDs must have mutually comparable types") from exc
    if len(set(map(str, units))) != len(units):
        raise ValueError("assignment IDs have ambiguous string representations")
    if treatment:
        if not frame[treatment].isin([0, 1]).all():
            raise ValueError("treatment must be binary")
        grouped = frame.groupby([assignment_unit, time])[treatment]
        if (grouped.nunique() != 1).any():
            raise ValueError("treatment varies within assignment-unit/time cells")
        cells = grouped.first().reset_index().sort_values(time)
        if (cells.groupby(assignment_unit)[treatment].diff().dropna() < 0).any():
            raise ValueError("treatment histories are not absorbing")
    if gvar:
        g = frame[gvar].to_numpy(dtype=float)
        if not np.isfinite(g).all() or (g < 0).any():
            raise ValueError("use finite positive cohort dates and 0 for never treated")
        if (frame.groupby(assignment_unit)[gvar].nunique() != 1).any():
            raise ValueError("cohort must be constant within assignment unit")
        mapping = frame.groupby(assignment_unit)[gvar].first().to_dict()
    else:
        mapping = {u: float(cells.loc[(cells[assignment_unit] == u) &
                    (cells[treatment] == 1), time].min()) for u in units}
        mapping = {u: 0.0 if np.isnan(g) else g for u, g in mapping.items()}
    frame[COHORT] = frame[assignment_unit].map(mapping).astype(float)
    frame[TREATMENT] = ((frame[COHORT] > 0) & (frame[time] >= frame[COHORT])).astype(int)
    if treatment and not np.array_equal(frame[treatment], frame[TREATMENT]):
        raise ValueError("treatment and cohort histories are inconsistent")
    times = np.unique(frame[time])
    for u, g in mapping.items():
        observed_times = frame.loc[frame[assignment_unit] == u, time]
        if g and (g not in times or not (observed_times < g).any() or not (observed_times >= g).any()):
            raise ValueError("each treated assignment unit needs observed pre/post support and a cohort date in the time grid")
    if not any(mapping.values()):
        raise ValueError("at least one treated assignment unit is required")
    return Prepared(frame, y, time, unit, assignment_unit, cluster, covariates,
                    weights, mode, units, tuple(float(mapping[u]) for u in units))
