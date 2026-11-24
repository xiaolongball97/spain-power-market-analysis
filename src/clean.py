"""Load, validate and clean the Spain energy + weather dataset.

Every issue found is counted and returned in a data-quality report, so the
cleaning is visible rather than silent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Approximate city populations (millions), used to weight weather into one
# national index instead of treating Bilbao the same as Madrid.
CITY_WEIGHTS = {"Madrid": 3.3, "Barcelona": 1.6, "Valencia": 0.8,
                "Seville": 0.7, "Bilbao": 0.35}

# Physically plausible ranges for weather fields
PLAUSIBLE = {"pressure": (900, 1100), "wind_speed": (0, 50),
             "temp": (240, 330), "humidity": (0, 100)}

MAX_INTERPOLATE_HOURS = 2


def load_raw(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    energy = pd.read_csv(data_dir / "energy_dataset.csv")
    weather = pd.read_csv(data_dir / "weather_features.csv")
    return energy, weather


def clean_energy(energy: pd.DataFrame, dq: dict) -> pd.DataFrame:
    e = energy.copy()
    dq["energy_rows"] = len(e)
    dq["local_offsets"] = e["time"].str[-6:].value_counts().to_dict()

    # Timestamps carry CET (+01:00) and CEST (+02:00) offsets -> convert to UTC
    e["time"] = pd.to_datetime(e["time"], utc=True)
    full = pd.date_range(e["time"].min(), e["time"].max(), freq="h")
    dq["energy_missing_hours_utc"] = int(len(full.difference(e["time"])))
    dq["energy_duplicate_hours_utc"] = int(e["time"].duplicated().sum())

    # Columns that are entirely empty or always zero carry no information
    num_cols = e.columns.drop("time")
    empty = [c for c in num_cols if e[c].isna().all() or (e[c].fillna(0) == 0).all()]
    dq["empty_columns"] = empty
    e = e.drop(columns=empty)
    num_cols = e.columns.drop("time")

    # Implausible zeros: a large plant type reading 0 for a single hour
    suspect_zero_cols = ["generation nuclear", "generation fossil gas",
                         "generation wind onshore", "generation hydro water reservoir"]
    zero_mask = (e[suspect_zero_cols] == 0)
    dq["suspect_zero_values"] = {c: int(zero_mask[c].sum()) for c in suspect_zero_cols}
    for c in suspect_zero_cols:
        e.loc[e[c] == 0, c] = np.nan

    dq["energy_missing_values_before"] = {c: int(v) for c, v in e[num_cols].isna().sum().items() if v > 0}
    dq["energy_hours_with_gaps_before"] = int(e[num_cols].isna().any(axis=1).sum())

    # Fill only short gaps; flag every hour that needed filling or is still missing
    e = e.set_index("time").reindex(full).rename_axis("time")
    e["had_gap"] = e[num_cols].isna().any(axis=1)
    e[num_cols] = e[num_cols].interpolate(limit=MAX_INTERPOLATE_HOURS, limit_area="inside")
    e["still_missing"] = e[num_cols].isna().any(axis=1)
    dq["energy_hours_filled"] = int((e["had_gap"] & ~e["still_missing"]).sum())
    dq["energy_hours_still_missing"] = int(e["still_missing"].sum())
    return e


def clean_weather(weather: pd.DataFrame, dq: dict) -> pd.DataFrame:
    w = weather.copy()
    dq["weather_rows"] = len(w)
    w["city_name"] = w["city_name"].str.strip()  # " Barcelona" has a leading space
    w["time"] = pd.to_datetime(w["dt_iso"], utc=True)

    dupes = w.duplicated(["time", "city_name"])
    dq["weather_duplicate_city_hours"] = int(dupes.sum())
    dq["weather_duplicates_by_city"] = w.loc[dupes, "city_name"].value_counts().to_dict()
    w = w.loc[~dupes]

    outliers = {}
    for col, (lo, hi) in PLAUSIBLE.items():
        bad = (w[col] < lo) | (w[col] > hi)
        if bad.any():
            outliers[col] = {"count": int(bad.sum()),
                             "cities": w.loc[bad, "city_name"].value_counts().to_dict(),
                             "min": float(w.loc[bad, col].min()),
                             "max": float(w.loc[bad, col].max())}
            w.loc[bad, col] = np.nan
    dq["weather_outliers_removed"] = outliers

    w["temp_c"] = w["temp"] - 273.15  # Kelvin -> Celsius
    cols = ["temp_c", "pressure", "humidity", "wind_speed", "clouds_all"]
    wide = w.pivot(index="time", columns="city_name", values=cols)
    wide = wide.interpolate(limit=MAX_INTERPOLATE_HOURS, limit_area="inside")
    dq["weather_rows_per_city_after_dedup"] = w["city_name"].value_counts().to_dict()

    # Population-weighted national index
    weights = pd.Series(CITY_WEIGHTS)
    national = pd.DataFrame(index=wide.index)
    for col in cols:
        sub = wide[col][weights.index]
        national[f"{col}_national"] = (sub * weights).sum(axis=1, min_count=1) / \
            sub.notna().mul(weights).sum(axis=1)
    national["temp_c_simple_mean"] = wide["temp_c"].mean(axis=1)
    return national


def build_hourly(data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Return one clean hourly table (UTC index) and the data-quality report."""
    dq: dict = {}
    energy_raw, weather_raw = load_raw(data_dir)
    energy = clean_energy(energy_raw, dq)
    weather = clean_weather(weather_raw, dq)
    df = energy.join(weather, how="left")
    dq["hours_after_join"] = len(df)
    dq["hours_without_weather"] = int(df["temp_c_national"].isna().sum())

    # Useful derived fields
    local = df.index.tz_convert("Europe/Madrid")
    df["local_time"] = local.tz_localize(None)
    df["date"] = local.date
    df["year"] = local.year
    df["month"] = local.month
    df["hour"] = local.hour
    df["weekday"] = local.dayofweek
    wind_cols = [c for c in df.columns if c.startswith("generation wind")]
    df["wind"] = df[wind_cols].sum(axis=1, min_count=1)
    df["solar"] = df["generation solar"]
    df["vre_share"] = (df["wind"] + df["solar"]) / df["total load actual"]
    df["price_gap"] = df["price actual"] - df["price day ahead"]
    return df, dq
