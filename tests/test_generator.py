import time

import numpy as np
import pandas as pd

from movrec.generator.generate import generate


def test_same_seed_same_data():
    a, b = generate(seed=3, n_bookings=800), generate(seed=3, n_bookings=800)
    pd.testing.assert_frame_equal(a.bookings, b.bookings)
    pd.testing.assert_frame_equal(a.booking_services, b.booking_services)


def test_different_seed_different_data():
    a, b = generate(seed=3, n_bookings=800), generate(seed=4, n_bookings=800)
    assert not a.bookings["total_price"].equals(b.bookings["total_price"])


def test_smoke_speed():
    t = time.time()
    generate(seed=5, n_bookings=2000)
    assert time.time() - t < 5


def test_one_truck_except_labor_only(small_data):
    b, lines = small_data.bookings, small_data.booking_services
    trucks = lines[lines["service_id"].str.startswith("truck_")].groupby("booking_id").size()
    trucks = b["booking_id"].map(trucks).fillna(0)
    labor_only = b["move_type"] == "labor_only"
    assert (trucks[labor_only] == 0).all()
    assert (trucks[~labor_only] == 1).all()
    assert (b.loc[labor_only, "truck"] == "none").all()


def test_one_crew_per_booking(small_data):
    assert small_data.bookings["crew"].isin(["crew_2", "crew_3", "crew_4"]).all()


def test_at_most_one_storage_unit_and_packing_level(small_data):
    lines = small_data.booking_services
    storage = lines[lines["service_id"].str.startswith("storage_")].groupby("booking_id").size()
    assert storage.max() <= 1
    both = lines[lines["service_id"].isin(["full_packing", "day_of_packing"])].groupby("booking_id").size()
    assert both.max() <= 1


def test_no_duplicate_lines(small_data):
    assert not small_data.booking_services.duplicated(["booking_id", "service_id"]).any()


def test_booking_before_move(small_data):
    b = small_data.bookings
    start_hour = np.where(b["arrival_window"] == "afternoon", 13, 8)
    move_start = b["move_date"] + pd.to_timedelta(start_hour, unit="h")
    assert (b["booking_ts"] < move_start).all()
    assert (b["lead_time_days"] >= 0).all()


def test_service_area(small_data, A):
    b = small_data.bookings
    allowed = set(A["local_regions"]) | set(A["long_distance_destinations"])
    assert set(b["origin_region"]) | set(b["dest_region"]) <= allowed
    assert b["distance_miles"].max() <= A["service_area"]["long_distance_max_miles"]
    local = b[b["move_type"] != "long_distance"]
    assert set(local["origin_region"]) | set(local["dest_region"]) <= set(A["local_regions"])


def test_hidden_segment_not_in_modeling_files(small_data):
    for frame in (small_data.bookings, small_data.booking_services, small_data.customers, small_data.services):
        assert not any("segment" in c for c in frame.columns)


def test_cold_start_is_the_norm(small_data):
    assert (small_data.customers["n_bookings"] == 1).mean() >= 0.85


def test_summer_busier_than_winter(small_data):
    month = small_data.bookings["move_date"].dt.month
    assert month.isin([6, 7, 8]).sum() > 1.3 * month.isin([12, 1, 2]).sum()


def test_prices_positive_and_consistent(small_data):
    b, lines = small_data.bookings, small_data.booking_services
    assert (b["base_price"] > 0).all()
    assert (lines["line_price"] >= 0).all()
    addon = lines.groupby("booking_id")["line_price"].sum()
    assert np.allclose(b["booking_id"].map(addon).fillna(0), b["addon_total"], atol=0.05)
    assert np.allclose(b["base_price"] + b["addon_total"], b["total_price"], atol=0.01)


def test_free_items_cost_nothing(small_data):
    lines = small_data.booking_services
    assert (lines.loc[lines["service_id"].isin(["furniture_dolly", "hand_truck_dolly"]), "line_price"] == 0).all()


def test_walkthrough_for_large_homes(small_data, A):
    b = small_data.bookings
    assert (b.loc[b["sq_ft"] > A["walkthrough_sq_ft"], "estimate_method"] == "in_person_walkthrough").all()


def test_customer_ids_consistent(small_data):
    b, c = small_data.bookings, small_data.customers
    assert set(b["customer_id"]) == set(c["customer_id"])
    assert c["n_bookings"].sum() == len(b)
    # a returning customer keeps one customer type
    assert (b.groupby("customer_id")["customer_type"].nunique() == 1).all()
