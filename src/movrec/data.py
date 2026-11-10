"""Load the generated data for modeling. Never reads data/_hidden.

Before loading, the CSV hashes are checked against data/MANIFEST.json, so results can
only come from the frozen generator output.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from movrec import CONFIG_DIR, DATA_DIR

MODELING_FILES = ["services.csv", "customers.csv", "bookings.csv", "booking_services.csv"]
TRUCK_CATEGORY = "truck"


class ManifestMismatch(RuntimeError):
    pass


def load_experiment_config(path: Path = CONFIG_DIR / "experiment.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def verify_manifest(data_dir: Path = DATA_DIR) -> dict:
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "MANIFEST.json").read_text())
    for name in MODELING_FILES:
        actual = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        if actual != manifest["files"][name]["sha256"]:
            raise ManifestMismatch(f"{name} does not match MANIFEST.json; regenerate with `make data`")
    return manifest


@dataclass
class ModelingData:
    services: pd.DataFrame
    bookings: pd.DataFrame          # one row per booking, sorted by booking_ts
    items: list[str]                # column order of `matrix`
    matrix: np.ndarray              # bookings x items, bool: service in the booking's cart

    def truck_mask(self) -> np.ndarray:
        cat = self.services.set_index("id").loc[self.items, "category"]
        return (cat == TRUCK_CATEGORY).to_numpy()


def cart_matrix(bookings: pd.DataFrame, lines: pd.DataFrame, items: list[str]) -> np.ndarray:
    row = pd.Series(np.arange(len(bookings)), index=bookings["booking_id"])
    col = pd.Series(np.arange(len(items)), index=items)
    m = np.zeros((len(bookings), len(items)), dtype=bool)
    m[row[lines["booking_id"]].to_numpy(), col[lines["service_id"]].to_numpy()] = True
    return m


def load_modeling_data(data_dir: Path = DATA_DIR, verify: bool = True) -> ModelingData:
    data_dir = Path(data_dir)
    if verify:
        verify_manifest(data_dir)
    services = pd.read_csv(data_dir / "services.csv")
    bookings = (pd.read_csv(data_dir / "bookings.csv", parse_dates=["booking_ts", "move_date"])
                  .sort_values("booking_ts", kind="stable").reset_index(drop=True))
    lines = pd.read_csv(data_dir / "booking_services.csv")
    items = services["id"].tolist()
    return ModelingData(services, bookings, items, cart_matrix(bookings, lines, items))
