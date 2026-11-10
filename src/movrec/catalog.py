"""Load config/catalog.yaml: which services exist and how they group into families."""
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import yaml

from movrec import CONFIG_DIR

DEFAULT_PATH = CONFIG_DIR / "catalog.yaml"
PRICING_TYPES = {"included", "free", "unit", "labor_hours", "specialty", "storage_monthly"}
ORIGINS = {"company_catalog", "company_kb", "my_addition"}
TRUCK_UNIT = "truck"


@dataclass(frozen=True)
class Service:
    id: str
    category: str
    unit: str
    pricing: str
    origin: str


class Catalog:
    def __init__(self, services: list[Service], crews: list[str]):
        self.services = services
        self.crews = crews
        self.by_id = {s.id: s for s in services}

    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.services]

    @property
    def units(self) -> list[str]:
        """Decision units (product families) in catalog order, excluding the truck."""
        seen = []
        for s in self.services:
            if s.unit != TRUCK_UNIT and s.unit not in seen:
                seen.append(s.unit)
        return seen

    def members(self, unit: str) -> list[str]:
        return [s.id for s in self.services if s.unit == unit]

    def ids_with_pricing(self, pricing: str) -> list[str]:
        return [s.id for s in self.services if s.pricing == pricing]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in self.services])


def load_catalog(path: Path = DEFAULT_PATH) -> Catalog:
    with open(path) as f:
        raw = yaml.safe_load(f)
    services = [Service(**item) for item in raw["services"]]
    ids = [s.id for s in services]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate service ids in catalog")
    for s in services:
        if s.pricing not in PRICING_TYPES:
            raise ValueError(f"{s.id}: unknown pricing '{s.pricing}'")
        if s.origin not in ORIGINS:
            raise ValueError(f"{s.id}: unknown origin '{s.origin}'")
    return Catalog(services, raw["crews"])
