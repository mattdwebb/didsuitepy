"""Fixed-count multiset ranks, sampled without replacement using Floyd's algorithm."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import random
import numpy as np
import pandas as pd


def space_size(counts):
    result = math.factorial(sum(counts))
    for count in counts:
        result //= math.factorial(count)
    return result


def rank_assignment(labels, values):
    counts = [labels.count(value) for value in values]
    rank = 0
    for label in labels:
        position = values.index(label)
        for j in range(position):
            if counts[j]:
                counts[j] -= 1
                rank += space_size(counts)
                counts[j] += 1
        counts[position] -= 1
    return rank


def unrank_assignment(rank, counts, values):
    counts = list(counts)
    result = []
    for _ in range(sum(counts)):
        for j, value in enumerate(values):
            if counts[j] == 0:
                continue
            counts[j] -= 1
            block = space_size(counts)
            if rank < block:
                result.append(value)
                break
            rank -= block
            counts[j] += 1
    return tuple(result)


@dataclass(frozen=True)
class AssignmentSet:
    units: tuple
    observed: tuple
    rows: tuple
    space: int

    @property
    def hash(self):
        # Versioned canonical text uses strings to avoid language-specific float JSON.
        payload = ["didsuite-assignments-v1", list(map(str, self.units)),
                   [format(v, ".17g") for v in self.observed],
                   [[format(v, ".17g") for v in row] for row in self.rows]]
        blob = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(blob).hexdigest()

    def to_frame(self):
        return pd.DataFrame(self.rows, columns=list(map(str, self.units)))

    def to_csv(self, path):
        self.to_frame().to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def assignment_set(observed, *, units=None, reps=999, seed=0, exact=False, supplied=None):
    observed = tuple(float(v) for v in observed)
    units = tuple(range(len(observed))) if units is None else tuple(units)
    if len(units) != len(observed) or len(set(units)) != len(units):
        raise ValueError("assignment units must be unique and match observed length")
    if not observed or not np.isfinite(observed).all() or min(observed) < 0:
        raise ValueError("cohorts must be finite and nonnegative")
    values = sorted(set(observed))
    counts = [observed.count(v) for v in values]
    space = space_size(counts)
    if space < 2:
        raise ValueError("assignment support contains no alternative")
    if supplied is not None:
        if isinstance(supplied, AssignmentSet):
            if supplied.units != units or supplied.observed != observed:
                raise ValueError("supplied assignment design differs")
            supplied = supplied.rows
        elif isinstance(supplied, pd.DataFrame):
            if list(map(str, supplied.columns)) != list(map(str, units)):
                raise ValueError("supplied assignment columns must match ordered assignment units")
            supplied = supplied.to_numpy()
        rows = tuple(tuple(float(v) for v in row) for row in supplied)
    else:
        if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
            raise ValueError("reps must be a positive integer")
        if exact and space > 100000:
            raise ValueError("exact enumeration exceeds 100000 assignments")
        n = space - 1 if exact else min(reps, space - 1)
        excluded = rank_assignment(observed, values)
        if n == space - 1:
            ranks = range(n)
        else:
            rng, seen, ranks = random.Random(seed), set(), []
            for j in range(space - 1 - n, space - 1):
                candidate = rng.randrange(j + 1)
                selected = j if candidate in seen else candidate
                seen.add(selected)
                ranks.append(selected)
        rows = tuple(unrank_assignment(rank + (rank >= excluded), counts, values) for rank in ranks)
    if not rows:
        raise ValueError("at least one alternative assignment is required")
    seen = set()
    for row in rows:
        if Counter(row) != Counter(observed):
            raise ValueError("assignment must preserve every cohort count and date")
        if row == observed or row in seen:
            raise ValueError("observed or duplicate assignment supplied")
        seen.add(row)
    return AssignmentSet(units, observed, rows, space)
