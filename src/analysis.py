"""Market analyses: price shape, renewables vs price, capture prices,
battery arbitrage and TSO forecast accuracy.

Market questions use the DAY-AHEAD price (the wholesale auction price).
The dataset's "price actual" is the final price, which adds system costs on
top of the auction, so it is analysed separately as the price gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from charts import (bar_chart, line_chart, save, series_colors, styled_axes)

PRICE = "price day ahead"


# ---------------------------------------------------------------- Q1 price shape
def price_shape(df: pd.DataFrame, chart_dir) -> dict:
    by_year = df.groupby("year")[PRICE].mean()
    by_hour = df.groupby(["year", "hour"])[PRICE].mean().unstack(0)
    overall_hour = df.groupby("hour")[PRICE].mean()
    by_month = df.groupby("month")[PRICE].mean()

    ax, fig = styled_axes("Day-ahead price by hour of day",
                          "Hour of day (local time)", "EUR/MWh")
    line_chart(ax, by_hour, label_fmt="{}")
    save(fig, chart_dir / "01_price_by_hour.png")

    return {
        "avg_price_by_year": by_year.round(1).to_dict(),
        "cheapest_hour": int(overall_hour.idxmin()),
        "cheapest_hour_price": round(float(overall_hour.min()), 1),
        "dearest_hour": int(overall_hour.idxmax()),
        "dearest_hour_price": round(float(overall_hour.max()), 1),
        "cheapest_month": int(by_month.idxmin()),
        "dearest_month": int(by_month.idxmax()),
        "tables": {"price_by_year_hour": by_hour.round(2)},
    }


# ------------------------------------------------------ Q2 renewables vs price
def renewables_vs_price(df: pd.DataFrame, chart_dir) -> dict:
    d = df.dropna(subset=["vre_share", PRICE, "total load actual"]).copy()
    d["vre_decile"] = pd.qcut(d["vre_share"], 10, labels=range(1, 11))
    deciles = d.groupby("vre_decile", observed=True).agg(
        vre_share=("vre_share", "mean"), price=(PRICE, "mean"), hours=(PRICE, "size"))

    # Regression: price on wind+solar share, holding demand, hour, month and year fixed
    X = pd.get_dummies(d[["hour", "month", "year"]].astype(str), drop_first=True).astype(float)
    X["vre_share"] = d["vre_share"]
    X["load_gw"] = d["total load actual"] / 1000
    X.insert(0, "const", 1.0)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), d[PRICE].to_numpy(), rcond=None)
    coef = dict(zip(X.columns, beta))
    pred = X.to_numpy() @ beta
    r2 = 1 - ((d[PRICE] - pred) ** 2).sum() / ((d[PRICE] - d[PRICE].mean()) ** 2).sum()

    ax, fig = styled_axes("Day-ahead price falls as wind and solar cover more demand",
                          "Wind + solar share of demand (decile)", "Average EUR/MWh")
    labels = [f"{v:.0%}" for v in deciles["vre_share"]]
    bar_chart(ax, labels, deciles["price"].to_numpy(), value_fmt="{:.0f}")
    save(fig, chart_dir / "02_renewables_vs_price.png")

    return {
        "correlation_vre_share_price": round(float(d["vre_share"].corr(d[PRICE])), 2),
        "price_lowest_vre_decile": round(float(deciles["price"].iloc[0]), 1),
        "price_highest_vre_decile": round(float(deciles["price"].iloc[-1]), 1),
        "vre_share_lowest_decile": round(float(deciles["vre_share"].iloc[0]), 3),
        "vre_share_highest_decile": round(float(deciles["vre_share"].iloc[-1]), 3),
        "eur_per_mwh_per_10pp_vre_controlled": round(float(coef["vre_share"]) / 10, 2),
        "eur_per_mwh_per_gw_load_controlled": round(float(coef["load_gw"]), 2),
        "regression_r2": round(float(r2), 3),
        "tables": {"price_by_vre_decile": deciles.round(3)},
    }


def capture_prices(df: pd.DataFrame, chart_dir) -> dict:
    rows = []
    for year, g in df.dropna(subset=[PRICE]).groupby("year"):
        base = g[PRICE].mean()
        row = {"year": year, "baseload": base}
        for tech in ["solar", "wind"]:
            cap = (g[PRICE] * g[tech]).sum() / g[tech].sum()
            row[f"{tech}_capture"] = cap
            row[f"{tech}_capture_rate"] = cap / base
        rows.append(row)
    cap = pd.DataFrame(rows).set_index("year")

    premium = (cap[["solar_capture_rate", "wind_capture_rate"]] - 1) * 100
    ax, fig = styled_axes("Price earned by solar and wind vs the average price",
                          "Year", "% above / below average price")
    bar_chart(ax, cap.index.astype(str), premium.to_numpy(),
              series_names=["Solar", "Wind"], value_fmt="{:+.1f}%", ref_line=0,
              legend_loc="upper right")
    save(fig, chart_dir / "03_capture_rates.png")

    return {"capture_by_year": cap.round(3).to_dict(orient="index"),
            "tables": {"capture_prices": cap.round(3)}}


# ------------------------------------------------------- Q3 battery arbitrage
def battery_arbitrage(df: pd.DataFrame, chart_dir, power_mw=1.0, hours=2,
                      round_trip=0.88) -> dict:
    """One full cycle per day on day-ahead prices.

    All 24 day-ahead prices are published together the day before, so choosing
    the best charge/discharge hours within a day is realistic for this market.
    Charging must happen before discharging.
    """
    def best_day(prices: np.ndarray) -> float:
        best = 0.0
        for split in range(hours, len(prices) - hours + 1):
            buy = np.sort(prices[:split])[:hours].sum()
            sell = np.sort(prices[split:])[-hours:].sum()
            best = max(best, sell * round_trip - buy)
        return best * power_mw

    daily = []
    for date, g in df.dropna(subset=[PRICE]).groupby("date"):
        p = g.sort_index()[PRICE].to_numpy()
        if len(p) >= 23:
            daily.append({"date": pd.Timestamp(date), "revenue": best_day(p),
                          "spread": p.max() - p.min()})
    daily = pd.DataFrame(daily).set_index("date")
    annual = daily.groupby(daily.index.year)["revenue"].sum()
    monthly = daily.groupby(daily.index.month)["revenue"].mean()
    zero_days = float((daily["revenue"] <= 0).mean())

    ax, fig = styled_axes(f"Battery arbitrage revenue per MW ({power_mw:g} MW / "
                          f"{power_mw * hours:g} MWh, one cycle a day)", "Year", "EUR per year")
    bar_chart(ax, annual.index.astype(str), annual.to_numpy(), value_fmt="{:,.0f}")
    save(fig, chart_dir / "04_battery_revenue.png")

    return {
        "assumptions": {"power_mw": power_mw, "energy_mwh": power_mw * hours,
                        "round_trip_efficiency": round_trip, "cycles_per_day": 1},
        "revenue_per_mw_by_year": annual.round(0).to_dict(),
        "avg_daily_spread_eur": round(float(daily["spread"].mean()), 1),
        "best_month_avg": int(monthly.idxmax()),
        "worst_month_avg": int(monthly.idxmin()),
        "share_days_not_worth_cycling": round(zero_days, 3),
        "tables": {"battery_daily": daily.round(2),
                   "battery_annual": annual.round(0).rename("revenue_eur_per_mw").to_frame()},
    }


# -------------------------------------------------- Q4 TSO forecasts & price gap
def _mae_pct(actual: pd.Series, fc: pd.Series) -> float:
    m = actual.notna() & fc.notna()
    return float((actual[m] - fc[m]).abs().sum() / actual[m].abs().sum())


def tso_forecast_accuracy(df: pd.DataFrame) -> dict:
    """Weighted absolute percentage error (total |error| / total actual).
    Avoids MAPE blowing up when solar output is near zero."""
    out = {
        "load_wape": _mae_pct(df["total load actual"], df["total load forecast"]),
        "solar_wape": _mae_pct(df["solar"], df["forecast solar day ahead"]),
        "wind_wape": _mae_pct(df["generation wind onshore"], df["forecast wind onshore day ahead"]),
        "wind_forecast_correlation": float(
            df["generation wind onshore"].corr(df["forecast wind onshore day ahead"])),
    }
    return {k: round(v, 4) for k, v in out.items()}


def price_gap(df: pd.DataFrame, chart_dir) -> dict:
    by_year = df.groupby("year")["price_gap"].mean()
    ax, fig = styled_axes("Final price minus day-ahead price",
                          "Year", "Average gap, EUR/MWh")
    bar_chart(ax, by_year.index.astype(str), by_year.to_numpy(), value_fmt="{:.1f}")
    save(fig, chart_dir / "05_price_gap.png")
    return {"avg_gap_by_year": by_year.round(2).to_dict(),
            "share_hours_final_above_day_ahead": round(float((df["price_gap"] > 0).mean()), 3),
            "avg_gap_all": round(float(df["price_gap"].mean()), 2)}


__all__ = ["price_shape", "renewables_vs_price", "capture_prices",
           "battery_arbitrage", "tso_forecast_accuracy", "price_gap", "series_colors"]
