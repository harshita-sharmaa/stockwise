"""Generate a synthetic weekly demand dataset for 40 SKUs over 2 years.

The data is simulated (fixed seed) so the project runs anywhere with no
external download. Each SKU gets its own base volume, trend, seasonality
strength and noise level, so the SKUs behave differently from each other.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_SKUS = 40
N_WEEKS = 104
START = "2024-07-01"
CATEGORIES = ["Routers", "Antennas", "Cables", "Batteries", "SIM Kits"]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    rng = np.random.default_rng(SEED)
    weeks = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
    t = np.arange(N_WEEKS)

    skus, demand = [], []
    for i in range(N_SKUS):
        sku = f"SKU-{i + 1:03d}"
        # A few big sellers and a long tail of small ones
        base = rng.lognormal(mean=4.0, sigma=0.9)
        trend = rng.normal(0.0005, 0.002)
        season = rng.uniform(0.0, 0.35)
        peak_week = rng.integers(0, 52)
        noise = rng.choice([0.08, 0.2, 0.45], p=[0.4, 0.35, 0.25])

        level = base * (1 + trend * t)
        seasonal = 1 + season * np.cos(2 * np.pi * (t - peak_week) / 52)
        units = level * seasonal * rng.normal(1, noise, N_WEEKS)
        units = np.clip(units, 0, None).round().astype(int)

        lead_time = int(rng.integers(1, 9))
        skus.append(
            {
                "sku": sku,
                "category": CATEGORIES[i % len(CATEGORIES)],
                "unit_cost": round(float(rng.uniform(4, 180)), 2),
                "lead_time_weeks": lead_time,
                "lead_time_std_weeks": round(float(rng.uniform(0.1, 0.3)) * lead_time, 2),
                "order_cost": 75.0,
                "holding_cost_rate": 0.22,
                "on_hand_units": int(base * rng.uniform(2, 16)),
            }
        )
        demand.append(pd.DataFrame({"week": weeks, "sku": sku, "units": units}))

    DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame(skus).to_csv(DATA_DIR / "sku_master.csv", index=False)
    pd.concat(demand).to_csv(DATA_DIR / "weekly_demand.csv", index=False)
    print(f"Wrote {N_SKUS} SKUs x {N_WEEKS} weeks to {DATA_DIR}")


if __name__ == "__main__":
    main()
