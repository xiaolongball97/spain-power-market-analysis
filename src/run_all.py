"""Run the whole project end to end.

    python src/run_all.py --data data/raw

Writes results/ (tables + summary.json), charts/ (PNG) and
data/processed/ (clean hourly and daily tables for Power BI or Tableau).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis import (battery_arbitrage, capture_prices, price_gap, price_shape,
                      renewables_vs_price, tso_forecast_accuracy)
from clean import build_hourly
from forecast import run_forecast

ROOT = Path(__file__).resolve().parents[1]


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def main(data_dir: Path) -> dict:
    results_dir, chart_dir, processed_dir = (ROOT / "results", ROOT / "charts",
                                             ROOT / "data" / "processed")
    for p in (results_dir, chart_dir, processed_dir):
        p.mkdir(parents=True, exist_ok=True)

    df, dq = build_hourly(data_dir)

    sections = {
        "price_shape": price_shape(df, chart_dir),
        "renewables_vs_price": renewables_vs_price(df, chart_dir),
        "capture_prices": capture_prices(df, chart_dir),
        "battery": battery_arbitrage(df, chart_dir),
        "tso_forecast_accuracy": tso_forecast_accuracy(df),
        "price_gap": price_gap(df, chart_dir),
        "day_ahead_forecast": run_forecast(df, chart_dir),
    }

    # Save every table, then keep only headline numbers in the summary
    for section in sections.values():
        for name, table in section.pop("tables", {}).items():
            table.to_csv(results_dir / f"{name}.csv")
    summary = {"data_quality": dq, **sections}
    (results_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2))

    # Clean tables for BI tools (gitignored: regenerate them locally)
    bi_cols = ["local_time", "date", "year", "month", "hour", "weekday",
               "price day ahead", "price actual", "price_gap", "total load actual",
               "total load forecast", "wind", "solar", "vre_share",
               "generation nuclear", "generation fossil gas", "generation fossil hard coal",
               "generation hydro water reservoir", "temp_c_national", "had_gap"]
    df[bi_cols].to_csv(processed_dir / "hourly_clean.csv", index=False)
    daily = df.groupby("date").agg(
        price_day_ahead=("price day ahead", "mean"), price_actual=("price actual", "mean"),
        load_mwh=("total load actual", "sum"), wind_mwh=("wind", "sum"),
        solar_mwh=("solar", "sum"), temp_c=("temp_c_national", "mean"))
    daily.to_csv(processed_dir / "daily_summary.csv")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"),
                    help="Folder with energy_dataset.csv and weather_features.csv")
    args = ap.parse_args()
    out = main(Path(args.data))
    print(json.dumps(_json_ready(out), indent=2))
