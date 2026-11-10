"""Pricing must reproduce the worked examples in the company knowledge base."""
import numpy as np

from movrec.generator import pricing


def test_payment_method_rates(A):
    assert pricing.hourly_rate(A, "crew_2", False, "cash") == 119
    assert pricing.hourly_rate(A, "crew_2", False, "venmo") == 124
    assert pricing.hourly_rate(A, "crew_2", False, "card") == 129


def test_weekend_rates(A):
    assert pricing.hourly_rate(A, "crew_3", True, "cash") == 169
    assert pricing.hourly_rate(A, "crew_4", True, "card") == 219


def test_three_hour_minimum(A):
    assert pricing.local_base_price(A, 119, labor_hours=1.5, drive_minutes=0) == 357
    assert pricing.local_base_price(A, 129, labor_hours=3, drive_minutes=0) == 387


def test_travel_is_double_drive_time(A):
    assert pricing.travel_hours(A, 15) == 0.5
    assert pricing.local_base_price(A, 119, labor_hours=3, drive_minutes=15) == 119 * 3.5


def test_kb_range_example(A):
    # 2 bedrooms, 3 movers, 5-7 hours weekday cash -> $795-$1,113
    assert pricing.local_base_price(A, 159, 5, 0) == 795
    assert pricing.local_base_price(A, 159, 7, 0) == 1113


def test_specialty_fee_groups_of_three(A):
    assert list(pricing.specialty_fee(A, np.array([1, 3, 4, 6, 7, 9]))) == [175, 175, 350, 350, 525, 525]


def test_storage_first_month_free_over_six_months(A):
    assert pricing.storage_price(A, 100, 6) == 600
    assert pricing.storage_price(A, 100, 7) == 600
    assert pricing.storage_price(A, 100, 12) == 1100


def test_long_distance_flat_rate(A):
    cfg = A["long_distance_rate_per_lb"]
    assert pricing.long_distance_base_price(A, 10000, 100) == 10000 * cfg["at_100_miles"]
    assert pricing.long_distance_base_price(A, 10000, 500) == 10000 * cfg["at_500_miles"]
    assert pricing.long_distance_base_price(A, 100, 300) == cfg["min_charge"]
