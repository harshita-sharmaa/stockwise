"""Forecast weekly demand per SKU and turn it into an inventory policy.

Steps
1. Backtest five simple forecasting methods on the last 12 weeks.
2. Pick the most accurate method per SKU (lowest WAPE).
3. Classify SKUs with ABC (value) and XYZ (demand variability).
4. Compute safety stock, reorder point and EOQ, then flag what to reorder.
"""
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

HOLDOUT_WEEKS = 12
SERVICE_LEVEL = {"A": 0.98, "B": 0.95, "C": 0.90}

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "text.color": INK,
        "axes.labelcolor": "#52514e",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 10,
    }
)


# ---------- forecasting methods: each takes history, returns h forecasts ----------

def naive(y, h):
    return np.repeat(y[-1], h)


def moving_average(y, h, window=4):
    return np.repeat(y[-window:].mean(), h)


def exp_smoothing(y, h, alpha=0.3):
    level = y[0]
    for value in y[1:]:
        level = alpha * value + (1 - alpha) * level
    return np.repeat(level, h)


def seasonal_naive(y, h, season=52):
    return y[-season:][:h]


def trend_seasonal_regression(y, h, season=52):
    """Least-squares fit of a linear trend plus one annual sine/cosine pair."""

    def features(t):
        return np.column_stack(
            [np.ones_like(t), t, np.sin(2 * np.pi * t / season), np.cos(2 * np.pi * t / season)]
        )

    t = np.arange(len(y), dtype=float)
    coef, *_ = np.linalg.lstsq(features(t), y, rcond=None)
    future = np.arange(len(y), len(y) + h, dtype=float)
    return np.clip(features(future) @ coef, 0, None)


METHODS = {
    "Naive": naive,
    "Moving average (4w)": moving_average,
    "Exp. smoothing": exp_smoothing,
    "Seasonal naive": seasonal_naive,
    "Trend + seasonal regression": trend_seasonal_regression,
}


def wape(actual, forecast):
    return np.abs(actual - forecast).sum() / max(actual.sum(), 1)


def bias(actual, forecast):
    return (forecast - actual).sum() / max(actual.sum(), 1)


# ---------- pipeline ----------

def backtest(demand):
    rows = []
    for sku, y in demand.items():
        train, test = y.values[:-HOLDOUT_WEEKS], y.values[-HOLDOUT_WEEKS:]
        for name, method in METHODS.items():
            forecast = method(train.astype(float), HOLDOUT_WEEKS)
            rows.append(
                {
                    "sku": sku,
                    "method": name,
                    "wape": wape(test, forecast),
                    "bias": bias(test, forecast),
                    "rmse": float(np.sqrt(np.mean((test - forecast) ** 2))),
                }
            )
    return pd.DataFrame(rows)


def classify(demand, skus):
    annual_units = demand.tail(52).sum()
    value = (annual_units * skus["unit_cost"]).sort_values(ascending=False)
    share = (value.cumsum() / value.sum()).clip(upper=1.0)
    abc = pd.cut(share, [0, 0.8, 0.95, 1.0], labels=["A", "B", "C"]).astype(str)

    cv = demand.std() / demand.mean()
    xyz = pd.cut(cv, [0, 0.25, 0.5, np.inf], labels=["X", "Y", "Z"]).astype(str)
    return pd.DataFrame({"annual_value": value, "abc": abc, "demand_cv": cv, "xyz": xyz})


def inventory_policy(demand, skus, best, classes):
    rows = []
    for sku, y in demand.items():
        s = skus.loc[sku]
        method = best.loc[sku, "method"]
        weekly_forecast = METHODS[method](y.values.astype(float), HOLDOUT_WEEKS).mean()
        sigma_demand = best.loc[sku, "rmse"]  # forecast error, not raw demand spread
        lead_time, sigma_lt = s["lead_time_weeks"], s["lead_time_std_weeks"]

        abc = classes.loc[sku, "abc"]
        z = NormalDist().inv_cdf(SERVICE_LEVEL[abc])
        safety_stock = z * np.sqrt(lead_time * sigma_demand**2 + weekly_forecast**2 * sigma_lt**2)
        reorder_point = weekly_forecast * lead_time + safety_stock
        eoq = np.sqrt(2 * weekly_forecast * 52 * s["order_cost"] / (s["unit_cost"] * s["holding_cost_rate"]))

        on_hand = s["on_hand_units"]
        rows.append(
            {
                "sku": sku,
                "category": s["category"],
                "abc_xyz": abc + classes.loc[sku, "xyz"],
                "best_method": method,
                "wape": round(best.loc[sku, "wape"], 3),
                "weekly_forecast": round(weekly_forecast, 1),
                "lead_time_weeks": lead_time,
                "service_level": SERVICE_LEVEL[abc],
                "safety_stock": int(np.ceil(safety_stock)),
                "reorder_point": int(np.ceil(reorder_point)),
                "eoq": int(np.ceil(eoq)),
                "on_hand_units": on_hand,
                "weeks_of_cover": round(on_hand / max(weekly_forecast, 0.1), 1),
                "action": "REORDER" if on_hand <= reorder_point else "OK",
                "suggested_order_qty": int(np.ceil(max(eoq, reorder_point - on_hand))) if on_hand <= reorder_point else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["action", "abc_xyz"], ascending=[False, True])


# ---------- charts ----------

def chart_method_accuracy(results):
    summary = results.groupby("method")["wape"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(summary.index, summary.values * 100, color=BLUE, height=0.55)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median WAPE across SKUs (lower is better)")
    ax.set_title("Forecast error by method, 12-week holdout")
    ax.set_xlim(0, summary.max() * 115)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "method_accuracy.png", dpi=160)
    plt.close(fig)


def chart_forecast(demand, best, sku):
    y = demand[sku]
    method = best.loc[sku, "method"]
    forecast = METHODS[method](y.values[:-HOLDOUT_WEEKS].astype(float), HOLDOUT_WEEKS)
    test_idx = y.index[-HOLDOUT_WEEKS:]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(y.index, y.values, color=BLUE, linewidth=2, label="Actual demand")
    ax.plot(test_idx, forecast, color=ORANGE, linewidth=2, marker="o", markersize=4, label=f"Forecast ({method})")
    ax.axvline(test_idx[0], color=MUTED, linewidth=1, linestyle="--")
    ax.set_ylim(0)
    ax.set_ylabel("Units per week")
    ax.set_title(f"{sku}: actual vs. forecast on the 12-week holdout (WAPE {best.loc[sku, 'wape']:.1%})")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forecast_example.png", dpi=160)
    plt.close(fig)


def chart_abc_xyz(policy):
    grid = (
        policy.assign(abc=policy["abc_xyz"].str[0], xyz=policy["abc_xyz"].str[1])
        .pivot_table(index="abc", columns="xyz", values="sku", aggfunc="count", fill_value=0)
        .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.imshow(grid.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#0d366b"]))
    ax.set_xticks(range(3), ["X: steady", "Y: variable", "Z: erratic"])
    ax.set_yticks(range(3), ["A: top 80% value", "B: next 15%", "C: last 5%"])
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            v = grid.values[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if v > grid.values.max() / 2 else INK)
    ax.set_title("SKU count by ABC-XYZ class")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "abc_xyz_matrix.png", dpi=160)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    skus = pd.read_csv(DATA_DIR / "sku_master.csv", index_col="sku")
    demand = pd.read_csv(DATA_DIR / "weekly_demand.csv", parse_dates=["week"]).pivot(
        index="week", columns="sku", values="units"
    )

    results = backtest(demand)
    best = results.loc[results.groupby("sku")["wape"].idxmin()].set_index("sku")
    classes = classify(demand, skus)
    policy = inventory_policy(demand, skus, best, classes)

    results.round(4).to_csv(OUT_DIR / "forecast_backtest.csv", index=False)
    policy.to_csv(OUT_DIR / "inventory_policy.csv", index=False)

    chart_method_accuracy(results)
    # Example chart: the highest-value SKU among those won by the most common method
    top_method = best["method"].mode()[0]
    example = classes.loc[best.index[best["method"] == top_method], "annual_value"].idxmax()
    chart_forecast(demand, best, example)
    chart_abc_xyz(policy)

    reorder = policy[policy["action"] == "REORDER"]
    print(f"SKUs analysed:            {len(policy)}")
    print(f"Median WAPE, naive:       {results[results.method == 'Naive'].wape.median():.1%}")
    print(f"Median WAPE, best-per-SKU: {best.wape.median():.1%}")
    print("Best method wins:")
    print(best["method"].value_counts().to_string())
    print(f"SKUs to reorder now:      {len(reorder)} ({(reorder.abc_xyz.str[0] == 'A').sum()} are class A)")
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
