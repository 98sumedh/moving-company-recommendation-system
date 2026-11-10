"""Pricing rules from the company knowledge base (numbers in assumptions.yaml)."""
import numpy as np
import pandas as pd

from movrec.catalog import Catalog

UNIT_PRICE_TABLES = [
    "box_prices", "wardrobe_box_prices", "plastic_wrap_prices",
    "other_supply_prices", "protection_equipment_prices",
]
STORAGE = "storage_monthly"
LOCAL_TYPES = {"local_residential", "commercial_office", "labor_only", "single_item"}


def unit_prices(A) -> dict[str, float]:
    prices = {}
    for key in UNIT_PRICE_TABLES:
        prices.update(A[key])
    return prices


def hourly_rate(A, crew, is_weekend, payment):
    """Crew rate (weekday/weekend) plus the payment-method surcharge. Works on scalars or arrays."""
    crew, is_weekend, payment = np.atleast_1d(crew), np.atleast_1d(is_weekend), np.atleast_1d(payment)
    rates = A["hourly_rates"]
    surcharge = A["payment_surcharge_per_hour"]
    base = np.array([rates[c]["weekend" if w else "weekday"] for c, w in zip(crew, is_weekend)], dtype=float)
    out = base + np.array([surcharge[p] for p in payment], dtype=float)
    return out if len(out) > 1 else float(out[0])


def travel_hours(A, drive_minutes):
    return A["travel_time_multiplier"] * np.asarray(drive_minutes, dtype=float) / 60.0


def local_base_price(A, rate, labor_hours, drive_minutes):
    """rate x (labor hours, at least the minimum, + travel time)."""
    billed = np.maximum(A["min_labor_hours"], np.asarray(labor_hours, dtype=float))
    return np.round(np.asarray(rate) * (billed + travel_hours(A, drive_minutes)), 2)


def specialty_fee(A, n_items):
    cfg = A["specialty_fee"]
    return cfg["per_group"] * np.ceil(np.asarray(n_items, dtype=float) / cfg["group_size"])


def storage_price(A, monthly_rate, months):
    """Monthly rent x months, minus one free month when longer than the promo threshold."""
    months = np.asarray(months, dtype=float)
    paid = np.where(months > A["storage_free_month_after"], months - 1, months)
    return np.round(np.asarray(monthly_rate, dtype=float) * paid, 2)


def long_distance_base_price(A, weight_lbs, miles):
    cfg = A["long_distance_rate_per_lb"]
    frac = np.clip((np.asarray(miles, dtype=float) - 100) / 400, 0, 1)
    rate = cfg["at_100_miles"] + frac * (cfg["at_500_miles"] - cfg["at_100_miles"])
    return np.round(np.maximum(cfg["min_charge"], np.asarray(weight_lbs) * rate), 2)


def sample_labor_hours(rng, A, bookings: pd.DataFrame, specialty_count: np.ndarray) -> np.ndarray:
    ranges = A["labor_hours_range"]
    lo = np.array([ranges[h][0] for h in bookings["home_size"]], dtype=float)
    hi = np.array([ranges[h][1] for h in bookings["home_size"]], dtype=float)
    hours = rng.uniform(lo, hi)
    hours = np.where(bookings["move_type"] == "labor_only", hours * A["labor_only_hours_factor"], hours)
    extra = A["extra_hours"]
    flights = bookings["origin_stairs_flights"] + bookings["dest_stairs_flights"]
    hours = hours + extra["per_stairs_flight"] * flights + extra["per_specialty_item"] * specialty_count
    hours = hours * rng.lognormal(0.0, extra["sd"], size=len(hours))
    return np.round(hours * 4) / 4  # quarter hours


def price_bookings(rng, A, catalog: Catalog, bookings: pd.DataFrame, carts: pd.DataFrame):
    """Add line_price to carts and labor/travel/price columns to bookings."""
    bookings = bookings.copy()
    carts = carts.copy()
    pricing = carts["service_id"].map(lambda s: catalog.by_id[s].pricing)

    specialty = carts[pricing == "specialty"].groupby("booking_id")["quantity"].sum()
    n_special = bookings["booking_id"].map(specialty).fillna(0).to_numpy()

    rate = hourly_rate(A, bookings["crew"].to_numpy(), bookings["is_weekend"].to_numpy(), bookings["payment_method"].to_numpy())
    bookings["hourly_rate"] = rate
    bookings["labor_hours"] = sample_labor_hours(rng, A, bookings, n_special)
    is_local = bookings["move_type"].isin(LOCAL_TYPES).to_numpy()
    bookings["travel_hours"] = np.where(is_local, np.round(travel_hours(A, bookings["drive_minutes"]), 2), 0.0)
    bookings["base_price"] = np.where(
        is_local,
        local_base_price(A, rate, bookings["labor_hours"], bookings["drive_minutes"]),
        long_distance_base_price(A, bookings["est_weight_lbs"], bookings["distance_miles"]),
    )

    # cart line prices
    rate_by_booking = pd.Series(rate, index=bookings["booking_id"])
    prices = unit_prices(A)
    monthly = A["storage_monthly_rates"]
    line = np.zeros(len(carts))
    p = pricing.to_numpy()
    q = carts["quantity"].to_numpy()
    sid = carts["service_id"].to_numpy()

    m = p == "unit"
    line[m] = np.array([prices[s] for s in sid[m]]) * q[m]
    m = p == "labor_hours"
    line[m] = carts.loc[m, "booking_id"].map(rate_by_booking).to_numpy() * q[m]
    m = p == STORAGE
    line[m] = storage_price(A, np.array([monthly[s] for s in sid[m]]), q[m])
    m = p == "specialty"
    if m.any():
        fee = pd.Series(specialty_fee(A, specialty.to_numpy()), index=specialty.index)
        share = q[m] / carts.loc[m, "booking_id"].map(specialty).to_numpy()
        line[m] = carts.loc[m, "booking_id"].map(fee).to_numpy() * share
    carts["line_price"] = np.round(line, 2)

    addon = carts.groupby("booking_id")["line_price"].sum()
    bookings["addon_total"] = bookings["booking_id"].map(addon).fillna(0.0).round(2)
    bookings["total_price"] = (bookings["base_price"] + bookings["addon_total"]).round(2)
    return bookings, carts

