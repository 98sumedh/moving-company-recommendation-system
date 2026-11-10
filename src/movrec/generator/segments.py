"""Hidden customer segments.

Segments drive add-on tastes but are NEVER written to the modeling files. They are
drawn as a soft function of the observable profile and the month, so features only
partly reveal them.
"""
import numpy as np
import pandas as pd


def _softmax_draw(rng, logits: np.ndarray) -> np.ndarray:
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    u = rng.random((len(p), 1))
    return (p.cumsum(axis=1) < u).sum(axis=1)


def assign_segments(rng, A, profiles: pd.DataFrame, features: dict) -> np.ndarray:
    n = len(profiles)
    base = A["segment_base_logit"]
    names = list(base)
    logits = np.tile(np.array([base[s] for s in names], dtype=float), (n, 1))

    for rule in A["segment_profile_effects"]:
        logits[features[rule["feature"]], names.index(rule["segment"])] += rule["delta"]

    month = pd.DatetimeIndex(profiles["move_date"]).month.to_numpy()
    for seg, by_month in A["segment_month_effects"].items():
        for m, delta in by_month.items():
            logits[month == m, names.index(seg)] += delta

    segment = np.array(names, dtype=object)[np.minimum(_softmax_draw(rng, logits), len(names) - 1)]

    commercial = features["commercial"]
    probs = A["commercial_segment_probs"]
    keys = list(probs)
    p = np.array([probs[k] for k in keys], dtype=float)
    segment[commercial] = rng.choice(np.array(keys, dtype=object), size=commercial.sum(), p=p / p.sum())
    return segment
