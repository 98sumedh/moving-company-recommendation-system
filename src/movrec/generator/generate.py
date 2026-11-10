"""Generate the full SYNTHETIC dataset and write CSVs + MANIFEST.json."""
import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from movrec import DATA_DIR, PROJECT_ROOT
from movrec.assumptions import Assumptions, load_assumptions
from movrec.catalog import Catalog, load_catalog
from movrec.generator import pricing, profiles, segments, services, timeline

HIDDEN_DIR = "_hidden"
BOOKING_COLUMNS = [
    "booking_id", "customer_id", "booking_ts", "move_date", "lead_time_days", "is_weekend", "arrival_window",
    "move_type", "customer_type", "home_size", "sq_ft", "est_weight_lbs",
    "origin_region", "dest_region", "distance_miles", "drive_minutes",
    "origin_floor", "origin_elevator", "origin_stairs_flights", "dest_floor", "dest_elevator", "dest_stairs_flights",
    "truck", "crew", "estimate_method", "payment_method", "hourly_rate",
    "labor_hours", "travel_hours", "base_price", "addon_total", "total_price",
    "status", "deposit_forfeited",
]


@dataclass
class Dataset:
    services: pd.DataFrame
    customers: pd.DataFrame
    bookings: pd.DataFrame
    booking_services: pd.DataFrame
    segments: pd.DataFrame  # hidden: diagnostics only


def _assign_customers(rng, A, bookings: pd.DataFrame, segment: np.ndarray, had_storage: np.ndarray) -> np.ndarray:
    """Walk bookings in time order. Each is a new customer or, sometimes, a returning one
    with the same customer type and hidden segment (storage customers come back more)."""
    repeat_p = A["repeat_prob"]
    storage_pool_p = A["repeat_storage_pool_prob"]
    min_gap = pd.Timedelta(days=A["repeat_min_gap_days"])

    pools: dict[tuple, list[int]] = {}
    storage_pools: dict[tuple, list[int]] = {}
    last_move: list[pd.Timestamp] = []
    out = np.empty(len(bookings), dtype=int)

    ctypes = bookings["customer_type"].to_numpy()
    booked = pd.DatetimeIndex(bookings["booking_ts"])
    moved = pd.DatetimeIndex(bookings["move_date"])
    for i in range(len(bookings)):
        key = (ctypes[i], segment[i])
        cust = None
        if rng.random() < repeat_p[ctypes[i]]:
            pool = storage_pools.get(key) if rng.random() < storage_pool_p else None
            pool = pool or pools.get(key)
            if pool:
                for _ in range(5):
                    cand = pool[rng.integers(len(pool))]
                    if last_move[cand] + min_gap <= booked[i]:
                        cust = cand
                        break
        if cust is None:
            cust = len(last_move)
            last_move.append(moved[i])
            pools.setdefault(key, []).append(cust)
        else:
            last_move[cust] = max(last_move[cust], moved[i])
        if had_storage[i]:
            storage_pools.setdefault(key, []).append(cust)
        out[i] = cust
    return out


def generate(seed: int = 42, n_bookings: int | None = None,
             A: Assumptions | None = None, catalog: Catalog | None = None) -> Dataset:
    A = A or load_assumptions()
    catalog = catalog or load_catalog()
    rng = np.random.default_rng(seed)
    default_n = A["n_bookings"]  # real-scale default; tests pass a smaller n
    n = n_bookings or default_n

    # 1. when and what
    df = profiles.sample_profiles(rng, A, n)
    df["move_date"] = timeline.sample_move_dates(rng, A, n)
    df["is_weekend"] = pd.DatetimeIndex(df["move_date"]).day_of_week >= 5

    # 2. hidden segment (depends on profile + month), then booking time (depends on segment)
    df["_segment"] = segments.assign_segments(rng, A, df, profiles.booking_features(df, A))
    lead = timeline.sample_lead_days(rng, A, df["move_type"].to_numpy(), df["_segment"].to_numpy())
    times = timeline.sample_times(rng, A, pd.DatetimeIndex(df["move_date"]), lead)
    df = pd.concat([df.drop(columns=["move_date", "is_weekend"]), times], axis=1)

    # 3. order by booking time and assign ids
    df = df.sort_values("booking_ts", kind="stable").reset_index(drop=True)
    df["booking_id"] = [f"B{i + 1:06d}" for i in range(n)]
    segment = df.pop("_segment").to_numpy()

    # 4. carts and prices
    features = profiles.booking_features(df, A)
    carts = services.sample_carts(rng, A, catalog, df, features, segment)
    df, carts = pricing.price_bookings(rng, A, catalog, df, carts)

    # 5. status and customers
    status_p = A["status_probs"]
    df["status"] = rng.choice(list(status_p), size=n, p=np.array(list(status_p.values())) / sum(status_p.values()))
    df["deposit_forfeited"] = (df["status"] == "cancelled") & (rng.random(n) < A["late_cancel_share"])
    df["deposit"] = A["deposit"]

    storage_ids = set(catalog.ids_with_pricing("storage_monthly"))
    storage_bookings = set(carts.loc[carts["service_id"].isin(storage_ids), "booking_id"])
    had_storage = df["booking_id"].isin(storage_bookings).to_numpy()
    cust_idx = _assign_customers(rng, A, df, segment, had_storage)
    df["customer_id"] = [f"C{c + 1:06d}" for c in cust_idx]

    customers = (df.groupby("customer_id")
                   .agg(customer_type=("customer_type", "first"), first_booking_ts=("booking_ts", "min"),
                        n_bookings=("booking_id", "size"))
                   .reset_index())

    bookings = df[BOOKING_COLUMNS + ["deposit"]]
    return Dataset(
        services=catalog.to_frame(),
        customers=customers,
        bookings=bookings,
        booking_services=carts[["booking_id", "service_id", "quantity", "line_price"]],
        segments=pd.DataFrame({"booking_id": df["booking_id"], "segment": segment}),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "src", "config"))}


def write_dataset(data: Dataset, out_dir: Path, seed: int) -> dict:
    out_dir = Path(out_dir)
    (out_dir / HIDDEN_DIR).mkdir(parents=True, exist_ok=True)
    files = {
        "services.csv": data.services,
        "customers.csv": data.customers,
        "bookings.csv": data.bookings,
        "booking_services.csv": data.booking_services,
        f"{HIDDEN_DIR}/segments.csv": data.segments,
    }
    manifest = {"synthetic": True, "seed": seed, "generator": _git_state(), "files": {}}
    for name, frame in files.items():
        path = out_dir / name
        frame.to_csv(path, index=False)
        manifest["files"][name] = {"rows": len(frame), "sha256": _sha256(path)}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate the SYNTHETIC moving-booking dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bookings", type=int, default=None, help="default: n_bookings in assumptions.yaml")
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    data = generate(seed=args.seed, n_bookings=args.n_bookings)
    manifest = write_dataset(data, args.out, args.seed)
    for name, info in manifest["files"].items():
        print(f"{name:28s} {info['rows']:>8,d} rows")


if __name__ == "__main__":
    main()
