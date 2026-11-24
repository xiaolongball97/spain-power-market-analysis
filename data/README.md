# Data

The raw data is not stored in this repo.

1. Download the dataset from Kaggle: [Hourly energy demand generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).
2. Put `energy_dataset.csv` and `weather_features.csv` in `data/raw/`.
3. Run `python src/run_all.py`. Clean tables for BI tools are written to `data/processed/`:
   - `hourly_clean.csv`: one row per hour (UTC-based, with local time), prices, demand, wind, solar, key generation types, national temperature and a `had_gap` flag
   - `daily_summary.csv`: one row per day
