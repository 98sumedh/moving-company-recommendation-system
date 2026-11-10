"""Move dates (seasonality), lead times and booking timestamps."""
import numpy as np
import pandas as pd

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def daily_weights(A) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Relative booking volume for each day in the window."""
    rng_cfg = A["date_range"]
    start = pd.Timestamp(rng_cfg["start"])
    end = start + pd.DateOffset(months=rng_cfg["months"]) - pd.Timedelta(days=1)
    days = pd.date_range(start, end, freq="D")

    month_mult = A["month_multiplier"]
    weekday_mult = A["weekday_multiplier"]
    w = np.array([month_mult[d.month] * weekday_mult[WEEKDAY_NAMES[d.day_of_week]] for d in days])

    lift = A["month_end_lift"]
    days_to_month_end = (days + pd.offsets.MonthEnd(0) - days).days.to_numpy()
    w[days_to_month_end < lift["last_days"]] *= lift["last_days_multiplier"]
    w[days.day == 1] *= lift["first_day_multiplier"]

    holidays = A["holiday_multiplier"]
    is_holiday = days.strftime("%m-%d").isin(holidays["dates"])
    w[is_holiday] *= holidays["multiplier"]

    elapsed_years = (days - start).days.to_numpy() / 365.25
    w *= 1 + A["annual_growth"] * elapsed_years
    return days, w / w.sum()


def sample_move_dates(rng, A, n: int) -> pd.DatetimeIndex:
    days, p = daily_weights(A)
    return pd.DatetimeIndex(np.sort(rng.choice(days.to_numpy(), size=n, p=p)))


def sample_lead_days(rng, A, move_type: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Days between booking and move: lognormal by move type, plus a same/next-day tail."""
    n = len(move_type)
    median = np.array([A["lead_time_median_days"][m] for m in move_type], dtype=float)
    seg_mult = A["segment_lead_time_multiplier"]
    median *= np.array([seg_mult.get(s, 1.0) for s in segment])
    lead = median * rng.lognormal(0.0, A["lead_time_sigma"], size=n)
    lead = np.clip(np.round(lead), 2, A["lead_time_max_days"]).astype(int)

    last_minute = (rng.random(n) < A["same_next_day_share"]) & (move_type != "long_distance")
    lead[last_minute] = rng.integers(0, 2, size=last_minute.sum())
    return lead


def sample_times(rng, A, move_dates: pd.DatetimeIndex, lead_days: np.ndarray) -> pd.DataFrame:
    """Arrival window and a booking timestamp inside office hours, before the crew arrives."""
    n = len(move_dates)
    windows = A["arrival_window_start_hour"]
    afternoon = rng.random(n) < A["afternoon_share"]
    afternoon |= lead_days == 0  # same-day jobs can only take the afternoon slot
    arrival_hour = np.where(afternoon, windows["afternoon"], windows["morning"])
    move_start = move_dates + pd.to_timedelta(arrival_hour, unit="h")

    booking_day = move_dates - pd.to_timedelta(lead_days, unit="D")
    hours = A["office_hours"]
    weekend = booking_day.day_of_week >= 5
    open_h = np.where(weekend, hours["weekend"][0], hours["weekday"][0])
    close_h = np.where(weekend, hours["weekend"][1], hours["weekday"][1])
    # same-day bookings must be confirmed before the afternoon window starts
    close_h = np.where(lead_days == 0, np.minimum(close_h, windows["afternoon"] - 1), close_h)
    minutes = rng.integers(open_h * 60, close_h * 60)
    booking_ts = booking_day + pd.to_timedelta(minutes, unit="m")

    return pd.DataFrame({
        "booking_ts": booking_ts,
        "move_date": move_dates,
        "move_start": move_start,
        "lead_time_days": lead_days,
        "is_weekend": move_dates.day_of_week >= 5,
        "arrival_window": np.where(afternoon, "afternoon", "morning"),
    })
