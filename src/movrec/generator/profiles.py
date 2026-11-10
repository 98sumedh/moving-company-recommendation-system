"""Observable move profile: type, size, weight, crew, truck, route, access, payment."""
import numpy as np
import pandas as pd


def _choice(rng, options: dict, size: int) -> np.ndarray:
    keys = list(options)
    p = np.array([options[k] for k in keys], dtype=float)
    return rng.choice(np.array(keys, dtype=object), size=size, p=p / p.sum())


def _choice_by_group(rng, groups: np.ndarray, table: dict) -> np.ndarray:
    """For each row, draw from table[group] (a dict of option -> probability)."""
    out = np.empty(len(groups), dtype=object)
    for g in np.unique(groups):
        mask = groups == g
        out[mask] = _choice(rng, table[g], mask.sum())
    return out


def _lognormal_by(rng, keys: np.ndarray, medians: dict, sigma: float) -> np.ndarray:
    median = np.array([medians[k] for k in keys], dtype=float)
    return median * rng.lognormal(0.0, sigma, size=len(keys))


def _access(rng, A, home_size: np.ndarray) -> pd.DataFrame:
    """Floor, elevator and stair flights for one address."""
    n = len(home_size)
    apt_p = np.array([A["apartment_prob"][h] for h in home_size])
    is_building = rng.random(n) < apt_p

    floor_probs = A["apartment_floor_probs"]
    floors = np.array(list(floor_probs), dtype=int)
    building_floor = rng.choice(floors, size=n, p=np.array(list(floor_probs.values())) / sum(floor_probs.values()))
    floor = np.where(is_building, building_floor, 1)

    elev_p = np.array([A["elevator_prob_by_floor"][f] for f in floor])
    elevator = is_building & (rng.random(n) < elev_p)

    two_story = ~is_building & (rng.random(n) < np.array([A["two_story_house_prob"][h] for h in home_size]))
    stairs = np.where(is_building, np.where(elevator, 0, floor - 1), two_story.astype(int))
    return pd.DataFrame({"floor": floor, "elevator": elevator, "stairs_flights": stairs})


def _route(rng, A, move_type: np.ndarray) -> pd.DataFrame:
    n = len(move_type)
    regions = A["local_regions"]
    region_w = {r: v["weight"] for r, v in regions.items()}
    county = {r: v["county"] for r, v in regions.items()}

    origin = _choice(rng, region_w, n)
    other = _choice(rng, region_w, n)
    same = rng.random(n) < A["local_same_region_prob"]
    dest = np.where(same, origin, other)

    drive = A["local_drive_minutes"]
    same_county = np.array([county[o] == county[d] for o, d in zip(origin, dest)])
    median = np.where(origin == dest, drive["same_region"],
                      np.where(same_county, drive["same_county"], drive["cross_county"]))
    minutes = median * rng.lognormal(0.0, drive["sigma"], size=n)
    miles = minutes * drive["miles_per_minute"]

    # long distance: one end local, the other a destination city
    ld = move_type == "long_distance"
    dests = A["long_distance_destinations"]
    city = _choice(rng, {c: v["weight"] for c, v in dests.items()}, n)
    ld_miles = np.array([dests[c]["miles"] for c in city]) * rng.lognormal(0.0, A["long_distance_miles_sigma"], size=n)
    ld_miles = np.minimum(ld_miles, A["service_area"]["long_distance_max_miles"])
    outbound = rng.random(n) < A["long_distance_outbound_prob"]
    origin = np.where(ld & ~outbound, city, origin)
    dest = np.where(ld & outbound, city, dest)
    miles = np.where(ld, ld_miles, miles)
    minutes = np.where(ld, ld_miles / A["long_distance_mph"] * 60, minutes)

    return pd.DataFrame({
        "origin_region": origin,
        "dest_region": dest,
        "distance_miles": np.round(miles, 1),
        "drive_minutes": np.round(minutes).astype(int),
    })


def _crew(rng, A, home_size: np.ndarray) -> np.ndarray:
    levels = np.array([2, 3, 4])
    typical = np.array([int(A["crew_by_home_size"][h].split("_")[1]) for h in home_size])
    shift = np.where(rng.random(len(home_size)) < A["crew_noise_prob"], rng.choice([-1, 1], size=len(home_size)), 0)
    crew = np.clip(typical + shift, levels.min(), levels.max())
    crew = np.where(home_size == "single_item", 2, crew)
    return np.array([f"crew_{c}" for c in crew], dtype=object)


def sample_profiles(rng, A, n: int) -> pd.DataFrame:
    move_type = _choice(rng, A["move_type_probs"], n)
    home_size = _choice_by_group(rng, move_type, A["home_size_probs"])

    sq_ft = np.round(_lognormal_by(rng, home_size, A["sq_ft_median"], A["sq_ft_sigma"]), -1).astype(int)
    weight = np.round(_lognormal_by(rng, home_size, A["weight_median_lbs"], A["weight_sigma"]), -1).astype(int)

    truck = _choice_by_group(rng, home_size, A["truck_probs"])
    truck[move_type == "labor_only"] = "none"

    origin_access = _access(rng, A, home_size).add_prefix("origin_")
    dest_access = _access(rng, A, home_size).add_prefix("dest_")

    payment_group = np.where(move_type == "long_distance", "long_distance", "default")
    payment = _choice_by_group(rng, payment_group, A["payment_method_probs"])

    in_person = (sq_ft > A["walkthrough_sq_ft"]) | (rng.random(n) < A["walkthrough_extra_prob"])

    df = pd.DataFrame({
        "move_type": move_type,
        "customer_type": np.where(move_type == "commercial_office", "commercial", "residential"),
        "home_size": home_size,
        "sq_ft": sq_ft,
        "est_weight_lbs": weight,
    })
    df = pd.concat([df, _route(rng, A, move_type), origin_access, dest_access], axis=1)
    df["crew"] = _crew(rng, A, home_size)
    df["truck"] = truck
    df["estimate_method"] = np.where(in_person, "in_person_walkthrough", "photos_video")
    df["payment_method"] = payment
    return df


def booking_features(df: pd.DataFrame, A) -> dict[str, np.ndarray]:
    """Boolean features used by segment and add-on rules (all observable)."""
    size_class = df["home_size"].map(A["size_class"])
    month = pd.DatetimeIndex(df["move_date"]).month
    return {
        "labor_only": (df["move_type"] == "labor_only").to_numpy(),
        "single_item": (df["move_type"] == "single_item").to_numpy(),
        "long_distance": (df["move_type"] == "long_distance").to_numpy(),
        "commercial": (df["move_type"] == "commercial_office").to_numpy(),
        "small_home": (size_class == "small").to_numpy(),
        "large_home": (size_class == "large").to_numpy(),
        "big_sq_ft": (df["sq_ft"] > A["walkthrough_sq_ft"]).to_numpy(),
        "in_person_estimate": (df["estimate_method"] == "in_person_walkthrough").to_numpy(),
        "has_stairs": ((df["origin_stairs_flights"] + df["dest_stairs_flights"]) > 0).to_numpy(),
        "has_elevator": (df["origin_elevator"] | df["dest_elevator"]).to_numpy(),
        "weekend": df["is_weekend"].to_numpy(),
        "summer": np.isin(month, A["season_months"]["summer"]),
        "rainy_season": np.isin(month, A["season_months"]["rainy_season"]),
    }
