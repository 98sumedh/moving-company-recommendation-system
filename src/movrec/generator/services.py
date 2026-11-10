"""Add-on sampling.

For every booking and every product family ("unit"):
    logit = base + profile effects + hidden-segment effect + booking noise + item noise
    chosen ~ Bernoulli(sigmoid(logit))
Then: co-occurrence lifts (second draw), exclusive pairs, random add/drop noise,
variant choice within a family, and quantities. Trucks come from the profile.
"""
import numpy as np
import pandas as pd

from movrec.catalog import Catalog

STORAGE_PRICED = "storage_monthly"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def family_logits(rng, A, units: list[str], features: dict, segment: np.ndarray) -> np.ndarray:
    n = len(segment)
    base = A["unit_base_logit"]
    missing = [u for u in units if u not in base]
    if missing:
        raise KeyError(f"unit_base_logit has no entry for: {missing}")
    col = {u: i for i, u in enumerate(units)}

    logits = np.tile(np.array([base[u] for u in units], dtype=float), (n, 1))
    logits += rng.normal(0.0, A["booking_noise_sd"], size=(n, 1))
    logits += rng.normal(0.0, A["item_noise_sd"], size=(n, len(units)))

    for rule in A["profile_effects"]:
        logits[features[rule["feature"]], col[rule["unit"]]] += rule["delta"]
    for seg, effects in A["segment_effects"].items():
        rows = segment == seg
        for unit, delta in effects.items():
            logits[rows, col[unit]] += delta
    return logits


def _apply_exclusive(A, chosen: np.ndarray, col: dict) -> None:
    for first, second in A["exclusive_units"]:
        chosen[chosen[:, col[first]], col[second]] = False


def choose_families(rng, A, units: list[str], logits: np.ndarray) -> np.ndarray:
    n = len(logits)
    col = {u: i for i, u in enumerate(units)}
    chosen = rng.random(logits.shape) < _sigmoid(logits)

    # co-occurrence: given A in cart, B's probability becomes sigmoid(logit_B + lift)
    for rule in A["cooccurrence_lifts"]:
        a, b = col[rule["if"]], col[rule["then"]]
        p0 = _sigmoid(logits[:, b])
        p1 = _sigmoid(logits[:, b] + rule["lift"])
        extra = (p1 - p0) / np.maximum(1 - p0, 1e-9)
        rows = chosen[:, a] & ~chosen[:, b]
        chosen[rows, b] = rng.random(rows.sum()) < extra[rows]
    _apply_exclusive(A, chosen, col)

    # noise: one random extra family for a few bookings, random drops everywhere
    add_rows = np.flatnonzero(rng.random(n) < A["random_add_prob"])
    chosen[add_rows, rng.integers(0, len(units), size=len(add_rows))] = True
    chosen &= ~(rng.random(chosen.shape) < A["random_drop_prob"])
    _apply_exclusive(A, chosen, col)
    return chosen


def _pick_variants(rng, A, catalog: Catalog, unit: str, size_class: np.ndarray) -> list[list[str]]:
    members = catalog.members(unit)
    if len(members) == 1:
        return [members] * len(size_class)
    weights = A["variant_weights"][unit]
    extra_p = A["extra_variant_prob"][unit]
    picks: list[list[str]] = [None] * len(size_class)
    for sc in np.unique(size_class):
        idx = np.flatnonzero(size_class == sc)
        w = np.array([weights[sc].get(m, 0.0) for m in members])
        w = w / w.sum()
        first = rng.choice(len(members), size=len(idx), p=w)
        second_wanted = rng.random(len(idx)) < extra_p
        for j, i in enumerate(idx):
            chosen = [members[first[j]]]
            if second_wanted[j]:
                rest = w.copy()
                rest[first[j]] = 0
                if rest.sum() > 0:
                    chosen.append(members[rng.choice(len(members), p=rest / rest.sum())])
            picks[i] = chosen
    return picks


def _quantity(rng, rule: dict, size_class: np.ndarray) -> np.ndarray:
    median = rule["median"]
    med = np.array([median[sc] for sc in size_class] if isinstance(median, dict) else [median] * len(size_class), dtype=float)
    q = med * rng.lognormal(0.0, rule["sd"], size=len(med)) if rule["sd"] > 0 else med
    q = np.round(q / rule["step"]) * rule["step"]
    return np.maximum(q, rule["min"])


def _storage_months(rng, A, segment: np.ndarray) -> np.ndarray:
    cfg = A["storage_months"]
    long_p = np.array([cfg["long_term_prob"].get(s, cfg["long_term_prob"]["default"]) for s in segment])
    is_long = rng.random(len(segment)) < long_p
    median = np.where(is_long, cfg["long_median"], cfg["short_median"])
    return np.maximum(1, np.round(median * rng.lognormal(0.0, cfg["sd"], size=len(segment))))


def sample_carts(rng, A, catalog: Catalog, bookings: pd.DataFrame, features: dict, segment: np.ndarray) -> pd.DataFrame:
    """Return long-format cart lines: booking_id, service_id, quantity."""
    units = catalog.units
    logits = family_logits(rng, A, units, features, segment)
    chosen = choose_families(rng, A, units, logits)
    size_class = bookings["home_size"].map(A["size_class"]).to_numpy()
    booking_ids = bookings["booking_id"].to_numpy()
    rules = A["quantity_rules"]

    parts = []
    truck_rows = bookings["truck"] != "none"
    parts.append(pd.DataFrame({"booking_id": booking_ids[truck_rows], "service_id": bookings.loc[truck_rows, "truck"], "quantity": 1.0}))

    for j, unit in enumerate(units):
        rows = np.flatnonzero(chosen[:, j])
        if len(rows) == 0:
            continue
        variants = _pick_variants(rng, A, catalog, unit, size_class[rows])
        b_ids, s_ids, r_idx = [], [], []
        for r, vs in zip(rows, variants):
            for v in vs:
                b_ids.append(booking_ids[r]); s_ids.append(v); r_idx.append(r)
        r_idx = np.array(r_idx)
        if catalog.by_id[s_ids[0]].pricing == STORAGE_PRICED:
            qty = _storage_months(rng, A, segment[r_idx])
        else:
            qty = _quantity(rng, rules[unit], size_class[r_idx])
        parts.append(pd.DataFrame({"booking_id": b_ids, "service_id": s_ids, "quantity": qty}))

    carts = pd.concat(parts, ignore_index=True)
    order = {sid: i for i, sid in enumerate(catalog.ids)}
    carts["_o"] = carts["service_id"].map(order)
    return carts.sort_values(["booking_id", "_o"]).drop(columns="_o").reset_index(drop=True)
