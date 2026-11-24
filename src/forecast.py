"""Day-ahead price forecast using only information known at gate closure.

The Spanish day-ahead auction closes at 12:00 local time on D-1. So for each
hour of day D the model may only use:
  * day-ahead prices already published for D-1 (and earlier),
  * actual system data from the morning of D-1 (hours 00-11),
  * the TSO's published forecasts for day D (load, wind, solar),
  * the calendar.
It never uses the price or generation of the hour just before the target
hour, which a real day-ahead forecaster would not have.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from charts import line_chart, save, styled_axes

TARGET = "price day ahead"
TSO_RENEWABLE_FC = ["forecast wind onshore day ahead", "forecast solar day ahead"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])

    # Same hour one day and one week earlier (already published day-ahead prices)
    d["da_lag24"] = d[TARGET].shift(24)
    d["da_lag168"] = d[TARGET].shift(168)

    # Daily summaries of D-1, mapped onto day D
    morning = d[d["hour"] < 12]
    daily = pd.DataFrame({
        "da_mean_d1": d.groupby("date")[TARGET].mean(),
        "da_max_d1": d.groupby("date")[TARGET].max(),
        "da_min_d1": d.groupby("date")[TARGET].min(),
        "hydro_reservoir_am_d1": morning.groupby("date")["generation hydro water reservoir"].mean(),
        "nuclear_am_d1": morning.groupby("date")["generation nuclear"].mean(),
        "gas_am_d1": morning.groupby("date")["generation fossil gas"].mean(),
        "temp_am_d1": morning.groupby("date")["temp_c_national"].mean(),
    })
    daily.index = daily.index + pd.Timedelta(days=1)
    d = d.join(daily, on="date")

    # TSO forecasts for the target hour, and the residual demand they imply
    d["residual_load_fc"] = (d["total load forecast"] - d["forecast wind onshore day ahead"]
                             - d["forecast solar day ahead"])
    d["is_weekend"] = (d["weekday"] >= 5).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * (d["month"] - 1) / 12)
    d["month_cos"] = np.cos(2 * np.pi * (d["month"] - 1) / 12)

    features = ["da_lag24", "da_lag168", "da_mean_d1", "da_max_d1", "da_min_d1",
                "hydro_reservoir_am_d1", "nuclear_am_d1", "gas_am_d1", "temp_am_d1",
                "total load forecast", *TSO_RENEWABLE_FC, "residual_load_fc",
                "hour", "weekday", "is_weekend", "hour_sin", "hour_cos",
                "month_sin", "month_cos"]
    d = d.dropna(subset=[TARGET, "da_lag168", "da_mean_d1"])
    return d, features


def _models():
    return {
        "ridge": Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler()),
                           ("model", Ridge(alpha=1.0))]),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42),
    }


# Two ways to frame the target: the price itself, or the price relative to
# yesterday's average (which copes better with shifts in the overall level).
TARGET_MODES = ("level", "relative_to_yesterday")


def _fit_predict(name, mode, train, test, features):
    model = _models()[name]
    y = train[TARGET] if mode == "level" else train[TARGET] - train["da_mean_d1"]
    model.fit(train[features], y)
    pred = model.predict(test[features])
    if mode != "level":
        pred = pred + test["da_mean_d1"].to_numpy()
    return model, pred


def walk_forward(d, year, name, mode, features):
    """Retrain at the start of every month on all data before it, then
    forecast that month. This is how a forecaster would run it in practice."""
    preds, model, last_month = [], None, None
    for month in range(1, 13):
        train = d[(d["year"] < year) | ((d["year"] == year) & (d["month"] < month))]
        test = d[(d["year"] == year) & (d["month"] == month)]
        model, p = _fit_predict(name, mode, train, test, features)
        preds.append(pd.Series(p, index=test.index))
        last_month = test
    return pd.concat(preds), model, last_month


def _scores(y, p, naive_mae=None) -> dict:
    err = y - p
    out = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
           "bias": float(err.mean())}
    if naive_mae:
        out["rmae_vs_naive_d1"] = out["mae"] / naive_mae
    return {k: round(v, 3) for k, v in out.items()}


def run_forecast(df: pd.DataFrame, chart_dir) -> dict:
    d, features = build_features(df)
    val, test = d[d["year"] == 2017], d[d["year"] == 2018]

    # 1. Choose model + target framing on 2017 only (walk-forward)
    val_rows = []
    for name in _models():
        for mode in TARGET_MODES:
            p, _, _ = walk_forward(d, 2017, name, mode, features)
            val_rows.append({"model": f"{name} | {mode}", "name": name, "mode": mode,
                             **_scores(val[TARGET], p.loc[val.index])})
    for label, col in [("naive_same_hour_yesterday", "da_lag24"),
                       ("naive_same_hour_last_week", "da_lag168")]:
        val_rows.append({"model": label, **_scores(val[TARGET], val[col])})
    val_table = pd.DataFrame(val_rows).set_index("model")
    best_row = val_table.dropna(subset=["name"]).sort_values("mae").iloc[0]
    best_name, best_mode = best_row["name"], best_row["mode"]
    best_label = f"{best_name} | {best_mode}"

    # 2. Test once on 2018 with the chosen set-up
    naive_mae = float((test[TARGET] - test["da_lag24"]).abs().mean())
    pred_s, final_model, last_month = walk_forward(d, 2018, best_name, best_mode, features)
    pred = pred_s.loc[test.index].to_numpy()

    # 3. Comparisons: trained once (no monthly retrain) and without TSO wind/solar forecasts
    _, pred_static = _fit_predict(best_name, best_mode, d[d["year"] <= 2017], test, features)
    reduced = [f for f in features if f not in TSO_RENEWABLE_FC + ["residual_load_fc"]]
    pred_reduced, _, _ = walk_forward(d, 2018, best_name, best_mode, reduced)

    test_table = pd.DataFrame([
        {"model": f"{best_label} (chosen, retrained monthly)", **_scores(test[TARGET], pred, naive_mae)},
        {"model": f"{best_label} trained once on 2015-2017",
         **_scores(test[TARGET], pred_static, naive_mae)},
        {"model": f"{best_label} without wind/solar forecasts",
         **_scores(test[TARGET], pred_reduced.loc[test.index].to_numpy(), naive_mae)},
        {"model": "naive_same_hour_yesterday", **_scores(test[TARGET], test["da_lag24"], naive_mae)},
        {"model": "naive_same_hour_last_week", **_scores(test[TARGET], test["da_lag168"], naive_mae)},
    ]).set_index("model")
    static_bias_by_month = (test[TARGET] - pred_static).groupby(test["month"]).mean().round(2)

    # 4. What drives the forecast: permutation importance, December model on December data
    y_dec = last_month[TARGET] if best_mode == "level" else last_month[TARGET] - last_month["da_mean_d1"]
    imp = permutation_importance(final_model, last_month[features], y_dec,
                                 scoring="neg_mean_absolute_error", n_repeats=5, random_state=1)
    importance = (pd.Series(imp.importances_mean, index=features)
                  .sort_values(ascending=False).round(3).rename("mae_increase_eur"))

    # 5. Chart: two weeks of the test year
    mask = ((test["local_time"] >= "2018-03-01") & (test["local_time"] < "2018-03-15")).to_numpy()
    plot = pd.DataFrame({"Actual day-ahead price": test[TARGET].to_numpy()[mask],
                         "Forecast": pred[mask],
                         "Same hour yesterday": test["da_lag24"].to_numpy()[mask]},
                        index=test["local_time"].to_numpy()[mask])
    ax, fig = styled_axes("Day-ahead price forecast vs actual, 1-14 March 2018 (test year)",
                          "", "EUR/MWh", size=(10, 4.5))
    line_chart(ax, plot)
    for label in ax.texts[:]:
        label.remove()  # long series names: keep the legend, drop end labels
    fig.autofmt_xdate()
    save(fig, chart_dir / "06_forecast_vs_actual.png")

    preds_out = pd.DataFrame({"local_time": test["local_time"].to_numpy(),
                              "price_day_ahead": test[TARGET].to_numpy(),
                              "forecast": pred.round(2),
                              "naive_same_hour_yesterday": test["da_lag24"].to_numpy()})
    chosen = test_table.iloc[0]
    return {
        "chosen_setup": best_label,
        "validation_period": "2017 (walk-forward, monthly retrain)",
        "test_period": "2018 (walk-forward, monthly retrain)",
        "test_rows": int(len(test)),
        "test_mae": float(chosen["mae"]),
        "test_rmse": float(chosen["rmse"]),
        "naive_d1_mae": round(naive_mae, 3),
        "improvement_vs_naive_d1": round(1 - float(chosen["mae"]) / naive_mae, 3),
        "trained_once_mae": float(test_table.iloc[1]["mae"]),
        "trained_once_bias": float(test_table.iloc[1]["bias"]),
        "without_wind_solar_fc_mae": float(test_table.iloc[2]["mae"]),
        "top_drivers": importance.head(5).to_dict(),
        "tables": {"forecast_validation_2017": val_table.drop(columns=["name", "mode"]),
                   "forecast_test_2018": test_table,
                   "forecast_trained_once_bias_by_month_2018": static_bias_by_month.rename("bias_eur").to_frame(),
                   "forecast_feature_importance": importance.to_frame(),
                   "forecast_predictions_2018": preds_out},
    }
