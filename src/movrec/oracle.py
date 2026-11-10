"""Generator-aware ceiling (DIAGNOSTIC ONLY, never a model).

Ranks services by an approximation of the true probability the generator used, so we can
see how much room is left above the models. Two versions:

- true segment:  knows each booking's hidden segment (reads data/_hidden). An approximate
                 upper bound; no real model can see this.
- profile only:  averages over segments with the generator's own segment probabilities
                 given the observable profile and month. Roughly the best any model could do
                 with the information it actually has.

Approximations (so neither is an exact Bayes-optimal bound):
- Per-booking noise is integrated with the probit trick E[sigmoid(l + e)] ~ sigmoid(l / sqrt(1 + pi s^2 / 8)),
  ignoring that the shared booking noise correlates items (a full cart hints at a big spender).
- A revealed family raises its co-occurrence partners by the full lift; random add/drop noise is ignored.
- Within a family, variants are ranked by their variant weight; a revealed variant leaves the others
  only the extra-variant probability.

It reuses the generator's own logit code (with noise switched off), so it cannot drift from it.
"""
import numpy as np
import pandas as pd

from movrec.catalog import TRUCK_UNIT, Catalog
from movrec.generator.profiles import booking_features
from movrec.generator.services import family_logits


class _NoNoise:
    """Assumptions view with the per-booking noise set to zero (the oracle integrates it analytically)."""

    def __init__(self, A):
        self.A = A

    def __getitem__(self, key):
        return 0.0 if key in ("booking_noise_sd", "item_noise_sd") else self.A[key]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def segment_probabilities(A, bookings: pd.DataFrame, features: dict) -> tuple[list[str], np.ndarray]:
    """P(segment | observable profile, month), mirroring generator.segments.assign_segments."""
    base = A["segment_base_logit"]
    commercial_p = A["commercial_segment_probs"]
    names = list(dict.fromkeys(list(base) + list(commercial_p)))
    n = len(bookings)
    logits = np.full((n, len(names)), -np.inf)
    logits[:, [names.index(s) for s in base]] = [base[s] for s in base]
    for rule in A["segment_profile_effects"]:
        logits[features[rule["feature"]], names.index(rule["segment"])] += rule["delta"]
    month = pd.DatetimeIndex(bookings["move_date"]).month.to_numpy()
    for seg, by_month in A["segment_month_effects"].items():
        for m, delta in by_month.items():
            logits[month == m, names.index(seg)] += delta
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    com = features["commercial"]
    p[com] = 0.0
    total = sum(commercial_p.values())
    for s, v in commercial_p.items():
        p[com, names.index(s)] = v / total
    return names, p


def _family_probs(A, catalog: Catalog, bookings, features, segment, revealed_units: np.ndarray) -> np.ndarray:
    units = catalog.units
    col = {u: i for i, u in enumerate(units)}
    logits = family_logits(np.random.default_rng(0), _NoNoise(A), units, features, segment)
    for rule in A["cooccurrence_lifts"]:
        logits[revealed_units[:, col[rule["if"]]], col[rule["then"]]] += rule["lift"]
    sd = np.hypot(A["booking_noise_sd"], A["item_noise_sd"])
    p = _sigmoid(logits / np.sqrt(1 + np.pi * sd ** 2 / 8))
    for first, second in A["exclusive_units"]:
        p[:, col[second]] *= 1 - p[:, col[first]]
    return p


def oracle_scores(A, catalog: Catalog, bookings: pd.DataFrame, revealed: np.ndarray,
                  items: list[str], segment: np.ndarray | None = None) -> np.ndarray:
    """Approximate true inclusion probability of each item in `items`.
    segment=None -> profile-only (segments averaged out); otherwise the true hidden segment."""
    bookings = bookings.reset_index(drop=True)
    features = booking_features(bookings, A)
    units = catalog.units
    item_col = {s: i for i, s in enumerate(items)}

    # which families already have a revealed member
    revealed_units = np.zeros((len(bookings), len(units)), dtype=bool)
    for j, u in enumerate(units):
        cols = [item_col[s] for s in catalog.members(u) if s in item_col]
        if cols:
            revealed_units[:, j] = revealed[:, cols].any(axis=1)

    if segment is not None:
        fam = _family_probs(A, catalog, bookings, features, np.asarray(segment, dtype=object), revealed_units)
    else:
        names, seg_p = segment_probabilities(A, bookings, features)
        fam = np.zeros((len(bookings), len(units)))
        for k, s in enumerate(names):
            fam += seg_p[:, [k]] * _family_probs(A, catalog, bookings, features,
                                                 np.full(len(bookings), s, dtype=object), revealed_units)

    size_class = bookings["home_size"].map(A["size_class"]).to_numpy()
    out = np.zeros((len(bookings), len(items)))
    for j, u in enumerate(units):
        members = catalog.members(u)
        if len(members) == 1:
            if members[0] in item_col:
                out[:, item_col[members[0]]] = fam[:, j]
            continue
        weights = A["variant_weights"][u]
        extra = A["extra_variant_prob"][u]
        for m in members:
            if m not in item_col:
                continue
            w = np.array([weights[sc].get(m, 0.0) for sc in size_class])
            out[:, item_col[m]] = np.where(revealed_units[:, j], extra * w, fam[:, j] * w)

    # truck: from the profile; a revealed truck rules out the others
    trucks = [s for s in catalog.members(TRUCK_UNIT) if s in item_col]
    if trucks:
        probs = A["truck_probs"]
        for t in trucks:
            p = np.array([probs[h].get(t, 0.0) for h in bookings["home_size"]])
            out[:, item_col[t]] = np.where(bookings["move_type"] == "labor_only", 0.0, p)
        any_truck = revealed[:, [item_col[t] for t in trucks]].any(axis=1)
        out[np.ix_(any_truck, [item_col[t] for t in trucks])] = 0.0
    return out
