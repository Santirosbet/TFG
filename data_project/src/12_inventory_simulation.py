"""
Step 12: Inventory Simulation.

Simulates a (s, Q) reorder-point inventory policy over the 60-week test
period under three forecast scenarios:

  Scenario 0 — Perfect foresight  : orders driven by actual future demand
  Scenario 1 — XGBoost forecast   : orders driven by XGBoost Model B predictions
  Scenario 2 — SARIMA forecast    : orders driven by SARIMA predictions
  Scenario 3 — Naive (no forecast): reorder when stock drops below ROP, fixed Q

Policy logic (forecast-driven):
  At each week t, compute projected stock at t + lead_time using the
  available forecast. If projected stock < safety_stock, place an order
  for (target_stock - projected_stock) units.

Outputs:
  output/inventory_simulation.csv  — week-by-week state for all scenarios
  output/inventory_summary.csv     — KPI table (stockouts, service level, costs)
  output/inventory_plot.png        — static 4-panel comparison chart

Run: python src/12_inventory_simulation.py
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# ── Simulation parameters ─────────────────────────────────────────────────────
PARAMS = {
    "initial_stock":    200,     # units at simulation start
    "safety_stock":     50,      # desired minimum buffer
    "target_stock":     220,     # order-up-to level
    "reorder_point":    90,      # trigger threshold (forecast-driven)
    "lead_time":        4,       # weeks from order placement to receipt
    "holding_cost":     0.50,    # EUR per unit held per week
    "shortage_cost":    8.00,    # EUR per unit of unmet demand per week
    "order_cost":       25.00,   # EUR fixed cost per order placed
}

BLUE   = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"
PURPLE = "#7030a0"
GREY   = "#aaaaaa"
RED    = "#c00000"


def simulate(actual_demand, forecast, params, label):
    """
    Run one (s,Q) simulation.

    Parameters
    ----------
    actual_demand : array-like — true weekly demand (what actually happens)
    forecast      : array-like — what the system BELIEVES demand will be
    params        : dict       — simulation parameters
    label         : str        — scenario name

    Returns
    -------
    DataFrame with one row per week: stock, ordered, received, demand,
    shortage, holding_cost, shortage_cost, order_cost
    """
    n          = len(actual_demand)
    stock      = float(params["initial_stock"])
    lead_time  = params["lead_time"]
    safety     = params["safety_stock"]
    target     = params["target_stock"]
    rop        = params["reorder_point"]
    h_cost     = params["holding_cost"]
    s_cost     = params["shortage_cost"]
    o_cost     = params["order_cost"]

    # Order pipeline: pending[t] = units arriving at week t
    pending = np.zeros(n + lead_time + 1)

    records = []
    for t in range(n):
        # 1. Receive pending order
        received = pending[t]
        stock    += received

        # 2. Fulfil demand
        demand_t  = float(actual_demand[t])
        fulfilled = min(stock, demand_t)
        shortage  = max(0.0, demand_t - fulfilled)
        stock     = max(0.0, stock - demand_t)

        # 3. Decide whether to order
        order_placed = 0.0
        # Projected stock at t + lead_time using available forecasts
        horizon_demand = float(np.sum(forecast[t: t + lead_time])) if t + lead_time <= n else \
                         float(np.sum(forecast[t:]))
        projected_stock = stock - horizon_demand + sum(pending[t+1: t + lead_time + 1])

        if projected_stock < safety or stock < rop:
            order_qty = max(0.0, target - stock - sum(pending[t+1: t + lead_time + 1]))
            if order_qty > 0:
                arrive_at = min(t + lead_time, n + lead_time)
                pending[arrive_at] += order_qty
                order_placed = order_qty

        records.append({
            "week":         t,
            "label":        label,
            "actual_demand": demand_t,
            "forecast":     float(forecast[t]),
            "stock":        stock,
            "received":     received,
            "shortage":     shortage,
            "order_placed": order_placed,
            "holding_cost": stock * h_cost,
            "shortage_cost": shortage * s_cost,
            "order_cost":   o_cost if order_placed > 0 else 0.0,
        })

    return pd.DataFrame(records)


def kpis(sim_df):
    n = len(sim_df)
    stockout_weeks  = (sim_df["shortage"] > 0).sum()
    total_shortage  = sim_df["shortage"].sum()
    service_level   = (1 - stockout_weeks / n) * 100
    avg_stock       = sim_df["stock"].mean()
    n_orders        = (sim_df["order_placed"] > 0).sum()
    total_h_cost    = sim_df["holding_cost"].sum()
    total_s_cost    = sim_df["shortage_cost"].sum()
    total_o_cost    = sim_df["order_cost"].sum()
    total_cost      = total_h_cost + total_s_cost + total_o_cost
    return {
        "stockout_weeks":  int(stockout_weeks),
        "total_shortage":  round(total_shortage, 1),
        "service_level":   round(service_level, 2),
        "avg_stock":       round(avg_stock, 1),
        "n_orders":        int(n_orders),
        "holding_cost":    round(total_h_cost, 2),
        "shortage_cost":   round(total_s_cost, 2),
        "order_cost":      round(total_o_cost, 2),
        "total_cost":      round(total_cost, 2),
    }


def plot_simulation(all_sims, dates, out_path):
    scenarios = list(all_sims.keys())
    colors    = [GREY, ORANGE, GREEN, PURPLE]
    n_rows    = 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 10))
    fig.suptitle("Inventory Simulation — Forecast Policy Comparison\n"
                 "(s,Q) reorder-point policy | 60-week test period",
                 fontsize=13, fontweight="bold", color=BLUE, y=0.99)

    safety = PARAMS["safety_stock"]
    rop    = PARAMS["reorder_point"]

    for idx, (name, df_s) in enumerate(all_sims.items()):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        color = colors[idx]

        # Stock level
        ax.fill_between(dates, df_s["stock"], 0,
                        where=(df_s["stock"] > rop),
                        alpha=0.15, color=GREEN, label="_nolegend_")
        ax.fill_between(dates, df_s["stock"], 0,
                        where=(df_s["stock"] <= rop) & (df_s["stock"] > 0),
                        alpha=0.25, color=ORANGE, label="_nolegend_")
        ax.fill_between(dates, df_s["stock"], 0,
                        where=(df_s["shortage"] > 0),
                        alpha=0.35, color=RED, label="_nolegend_")

        ax.plot(dates, df_s["stock"], color=color, lw=2, label="Stock level")
        ax.axhline(safety, color=GREEN,  linestyle=":", lw=1.2, label=f"Safety stock ({safety})")
        ax.axhline(rop,    color=ORANGE, linestyle=":", lw=1.2, label=f"Reorder point ({rop})")

        # Order markers
        order_dates = dates[df_s["order_placed"] > 0]
        order_qty   = df_s.loc[df_s["order_placed"] > 0, "stock"]
        ax.scatter(order_dates, order_qty, marker="^", s=60, color=BLUE,
                   zorder=5, label="Order placed")

        # Stockout markers
        stockout_dates = dates[df_s["shortage"] > 0]
        if len(stockout_dates):
            ax.scatter(stockout_dates,
                       np.zeros(len(stockout_dates)),
                       marker="x", s=80, color=RED, zorder=6, label="Stockout")

        k = kpis(df_s)
        ax.set_title(
            f"{name}\nService level: {k['service_level']:.1f}%  |  "
            f"Stockouts: {k['stockout_weeks']}w  |  Total cost: EUR {k['total_cost']:.0f}",
            fontsize=9, color=BLUE
        )
        ax.set_ylabel("Units in stock", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def main():
    print("=" * 65)
    print("STEP 12: INVENTORY SIMULATION")
    print("=" * 65)

    # ── Load predictions ──────────────────────────────────────────────────────
    xgb_df = pd.read_csv(os.path.join(OUT_DIR, "test_predictions.csv"),
                          parse_dates=["week_date"])
    actual   = xgb_df["actual_R03"].values
    xgb_pred = xgb_df["predicted_R03"].values
    dates    = pd.to_datetime(xgb_df["week_date"])
    n        = len(actual)

    sarima_path = os.path.join(OUT_DIR, "sarima_predictions.csv")
    if os.path.exists(sarima_path):
        sarima_df   = pd.read_csv(sarima_path, parse_dates=["week_date"])
        sarima_pred = sarima_df["sarima_pred"].values
        has_sarima  = True
        print("[OK] SARIMA predictions loaded")
    else:
        sarima_pred = np.full(n, np.mean(actual))
        has_sarima  = False
        print("[WARN] SARIMA predictions not found — using mean as placeholder")

    print(f"[OK] Simulation period: {dates.iloc[0].date()} -> {dates.iloc[-1].date()}"
          f"  ({n} weeks)")
    print(f"     Actual demand — mean={actual.mean():.1f}  "
          f"std={actual.std():.1f}  max={actual.max():.0f}")

    # ── Run four scenarios ────────────────────────────────────────────────────
    scenarios = {
        "Naive (sin forecast)":    simulate(actual, np.full(n, actual.mean()), PARAMS, "Naive"),
        "XGBoost forecast":        simulate(actual, xgb_pred,  PARAMS, "XGBoost"),
        "SARIMA forecast":         simulate(actual, sarima_pred, PARAMS, "SARIMA"),
        "Prevision perfecta":      simulate(actual, actual,     PARAMS, "Perfect"),
    }

    # ── Compute KPIs ──────────────────────────────────────────────────────────
    summary_rows = []
    print("\n" + "=" * 65)
    print("KPI COMPARISON")
    print("=" * 65)
    for name, df_s in scenarios.items():
        k = kpis(df_s)
        k["scenario"] = name
        summary_rows.append(k)
        print(f"\n  {name}")
        print(f"    Service level  : {k['service_level']:.1f}%  "
              f"(stockouts: {k['stockout_weeks']} weeks)")
        print(f"    Avg stock      : {k['avg_stock']:.1f} units")
        print(f"    Orders placed  : {k['n_orders']}")
        print(f"    Holding cost   : EUR {k['holding_cost']:.0f}")
        print(f"    Shortage cost  : EUR {k['shortage_cost']:.0f}")
        print(f"    Total cost     : EUR {k['total_cost']:.0f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    cols_order = ["scenario", "service_level", "stockout_weeks", "total_shortage",
                  "avg_stock", "n_orders", "holding_cost", "shortage_cost",
                  "order_cost", "total_cost"]
    summary_df[cols_order].to_csv(os.path.join(OUT_DIR, "inventory_summary.csv"), index=False)
    print(f"\n[OK] Saved: output/inventory_summary.csv")

    # Combined per-week simulation CSV
    all_dfs = []
    for name, df_s in scenarios.items():
        df_s = df_s.copy()
        df_s["week_date"] = dates.values
        all_dfs.append(df_s)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "inventory_simulation.csv"), index=False)
    print(f"[OK] Saved: output/inventory_simulation.csv")

    # Save params for dashboard
    inv_meta = {
        "params": PARAMS,
        "n_weeks": int(n),
        "test_start": str(dates.iloc[0].date()),
        "test_end":   str(dates.iloc[-1].date()),
        "scenarios":  list(scenarios.keys()),
        "kpis":       {k["scenario"]: k for k in summary_rows},
    }
    with open(os.path.join(OUT_DIR, "inventory_meta.json"), "w") as f:
        json.dump(inv_meta, f, indent=2)
    print(f"[OK] Saved: output/inventory_meta.json")

    # ── Static plot ───────────────────────────────────────────────────────────
    plot_simulation(scenarios, dates, os.path.join(OUT_DIR, "inventory_plot.png"))

    # ── Winner analysis ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    xgb_k   = next(k for k in summary_rows if "XGBoost" in k["scenario"])
    naive_k  = next(k for k in summary_rows if "Naive"   in k["scenario"])
    saving   = naive_k["total_cost"] - xgb_k["total_cost"]
    svc_delta = xgb_k["service_level"] - naive_k["service_level"]
    print(f"  XGBoost vs Naive:")
    print(f"    Cost saving      : EUR {saving:.0f} over {n} weeks")
    print(f"    Service level +  : {svc_delta:+.1f} pp")
    print(f"    Stockout weeks - : {naive_k['stockout_weeks'] - xgb_k['stockout_weeks']}")
    print("\n[DONE] Inventory simulation complete.")


if __name__ == "__main__":
    main()
