"""Time-based split on booking_ts (train -> val -> test, no shuffling)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: np.ndarray   # boolean row masks over ModelingData.bookings
    val: np.ndarray
    test: np.ndarray


def time_split(bookings: pd.DataFrame, val_start: str, test_start: str) -> Split:
    ts = bookings["booking_ts"]
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
    if not val_start < test_start:
        raise ValueError("val_start must be before test_start")
    train = (ts < val_start).to_numpy()
    test = (ts >= test_start).to_numpy()
    return Split(train=train, val=~train & ~test, test=test)


def describe(bookings: pd.DataFrame, split: Split) -> pd.DataFrame:
    rows = []
    for name in ("train", "val", "test"):
        part = bookings[getattr(split, name)]
        rows.append({
            "split": name,
            "bookings": len(part),
            "booked_from": f"{part['booking_ts'].min():%Y-%m-%d}",
            "booked_to": f"{part['booking_ts'].max():%Y-%m-%d}",
            "moves_from": f"{part['move_date'].min():%Y-%m-%d}",
            "moves_to": f"{part['move_date'].max():%Y-%m-%d}",
            "summer_moves": f"{100 * part['move_date'].dt.month.isin([6, 7, 8]).mean():.0f}%",
        })
    return pd.DataFrame(rows)
