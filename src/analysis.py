"""Checks, prices, sensitivity analysis, and charts.

This module never builds Pyomo models itself; it calls the run_* wrappers in model.py and works on
the resulting pandas tables. That keeps the verification logic independent of the solver.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import model
from .data import BASE_GAS_PRICE, RESERVE_FRACTION, Inputs, breakeven_gas_price, marginal_costs

TOL = 1e-4  # MW tolerance for feasibility and merit-order checks


# ======================================================================================
# Verification (independent of the solver: works only on the output tables)
# ======================================================================================
def starts_and_stops(commitment: pd.DataFrame, inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive start-up (v) and shut-down (w) indicators from the on/off schedule."""
    v = pd.DataFrame(0, index=commitment.index, columns=commitment.columns)
    w = v.copy()
    for g in commitment.columns:
        prev = int(inputs.thermal.loc[g, "init_on"])
        for t in commitment.index:
            cur = int(commitment.loc[t, g])
            v.loc[t, g] = int(cur == 1 and prev == 0)
            w.loc[t, g] = int(cur == 0 and prev == 1)
            prev = cur
    return v, w


def verify_uc_solution(dispatch: pd.DataFrame, commitment: pd.DataFrame, inputs: Inputs,
                       reserve_frac: float = RESERVE_FRACTION, demand: list[float] | None = None) -> pd.DataFrame:
    """Re-check every constraint on the solved schedule. Returns one row per check:
    passed (bool) and the worst violation in MW (or hours for min up/down)."""
    D = np.array(demand if demand is not None else inputs.load, dtype=float)
    th = inputs.thermal
    units = inputs.units
    p = dispatch[units]
    u = commitment[units]
    out = {}

    gen = p.sum(axis=1) + dispatch["WIND"] + inputs.nuclear_mw
    out["demand balance"] = float(np.abs(gen.to_numpy() - D).max())

    lo = (p.to_numpy() - u.to_numpy() * th["pmin"].to_numpy())  # should be >= 0
    hi = (p.to_numpy() - u.to_numpy() * th["pmax"].to_numpy())  # should be <= 0
    out["output limits (Pmin*u <= p <= Pmax*u)"] = float(max((-lo).max(), hi.max(), 0.0))

    v, w = starts_and_stops(commitment, inputs)
    up_viol = down_viol = 0
    T = len(D)
    for g in units:
        UT, DT = int(th.loc[g, "min_up"]), int(th.loc[g, "min_down"])
        for t in range(T):
            if v.loc[t, g] == 1:  # after a start, stay on for UT hours (within the horizon)
                up_viol += int((commitment.loc[t:min(T - 1, t + UT - 1), g] == 0).sum())
            if w.loc[t, g] == 1:  # after a stop, stay off for DT hours (within the horizon)
                down_viol += int((commitment.loc[t:min(T - 1, t + DT - 1), g] == 1).sum())
    out["min up time"] = float(up_viol)
    out["min down time"] = float(down_viol)

    ramp_viol = 0.0
    for g in units:
        prev_p = float(th.loc[g, "init_p"])
        for t in range(T):
            d = p.loc[t, g] - prev_p
            pm = th.loc[g, "pmax"]
            r = th.loc[g, "ramp"]
            if d > 0:
                ramp_viol = max(ramp_viol, d - r - pm * v.loc[t, g])
            else:
                ramp_viol = max(ramp_viol, -d - r - pm * w.loc[t, g])
            prev_p = p.loc[t, g]
    out["ramp limits"] = float(max(ramp_viol, 0.0))

    headroom = (u.to_numpy() * th["pmax"].to_numpy() - p.to_numpy()).sum(axis=1)
    out["reserve (headroom >= r*demand)"] = float(max((reserve_frac * D - headroom).max(), 0.0))

    df = pd.DataFrame({"worst_violation": out})
    df["passed"] = df["worst_violation"] <= TOL
    return df


def merit_order_violations(dispatch: pd.DataFrame, inputs: Inputs, gas_price: float) -> list[tuple]:
    """Economic-dispatch test: no unit is running while a cheaper unit still has headroom.

    Returns a list of (hour, expensive_unit, cheaper_unit); an empty list means the test passes.
    """
    mc = marginal_costs(inputs, gas_price)
    pmax = inputs.thermal["pmax"].to_dict()
    bad = []
    for t in dispatch.index:
        for a in inputs.units:
            if dispatch.loc[t, a] <= TOL:
                continue
            for b in inputs.units:
                if mc[a] > mc[b] + 1e-9 and dispatch.loc[t, b] < pmax[b] - TOL:
                    bad.append((int(t), a, b))
        if dispatch.loc[t, "WIND_CURTAILED"] > TOL and any(dispatch.loc[t, g] > TOL for g in inputs.units):
            bad.append((int(t), "thermal", "WIND"))  # wind costs $0, so it should never be curtailed
    return bad


def marginal_unit_ed(dispatch: pd.DataFrame, inputs: Inputs, gas_price: float) -> list[str]:
    """Per hour, the unit that would supply one more MW.

    If a unit is partly loaded it is the marginal unit. If every running unit is at a limit
    (a breakpoint in the merit order), the next MW comes from the cheapest unit with headroom.
    If wind is being curtailed, an extra MW is free (price 0).
    """
    mc = marginal_costs(inputs, gas_price)
    pmax = inputs.thermal["pmax"].to_dict()
    out = []
    for t in dispatch.index:
        partial = [g for g in inputs.units if TOL < dispatch.loc[t, g] < pmax[g] - TOL]
        headroom = [g for g in inputs.units if dispatch.loc[t, g] < pmax[g] - TOL]
        if dispatch.loc[t, "WIND_CURTAILED"] > TOL:
            out.append("WIND(curtailed)")
        elif partial:
            out.append(max(partial, key=mc.get))
        elif headroom:
            out.append(min(headroom, key=mc.get))
        else:
            out.append("none (no headroom)")
    return out


def count_starts(commitment: pd.DataFrame, inputs: Inputs) -> int:
    v, _ = starts_and_stops(commitment, inputs)
    return int(v.to_numpy().sum())


def dispatch_shares(dispatch: pd.DataFrame, inputs: Inputs) -> pd.Series:
    """Share of total energy delivered by each unit (nuclear and wind included)."""
    cols = inputs.units + ["NUC1", "WIND"]
    mwh = dispatch[cols].sum()
    return mwh / mwh.sum()


# ======================================================================================
# Marginal prices
# ======================================================================================
def ed_marginal_prices(inputs: Inputs, gas_price: float = BASE_GAS_PRICE, wind_scale: float = 1.0) -> pd.Series:
    """Hourly system marginal price for economic dispatch: re-solve with demand +1 MW in hour t and
    take the change in total cost."""
    base = model.run_ed(inputs, gas_price, wind_scale=wind_scale)
    prices = {}
    for t in inputs.hours:
        d = list(inputs.load)
        d[t] += 1.0
        bumped = model.run_ed(inputs, gas_price, demand=d, wind_scale=wind_scale)
        prices[t] = (bumped.info.objective - base.info.objective) if bumped.info.optimal else np.nan
    return pd.Series(prices, name="price_$per_MWh")


def uc_marginal_prices(inputs: Inputs, commitment: pd.DataFrame, gas_price: float = BASE_GAS_PRICE,
                       reserve_frac: float = RESERVE_FRACTION, wind_scale: float = 1.0) -> pd.Series:
    """Energy price with the commitment held fixed (an LP): +1 MW of demand in hour t.

    This excludes start-up and no-load effects by construction (they are sunk once the schedule
    is fixed). NaN means one more MW could not be served by the committed units.
    """
    base = model.run_uc_fixed(inputs, commitment, gas_price, reserve_frac=reserve_frac, wind_scale=wind_scale)
    prices = {}
    for t in inputs.hours:
        d = list(inputs.load)
        d[t] += 1.0
        bumped = model.run_uc_fixed(inputs, commitment, gas_price, demand=d, reserve_frac=reserve_frac,
                                    wind_scale=wind_scale)
        ok = base.info.optimal and bumped.info.optimal
        prices[t] = (bumped.info.objective - base.info.objective) if ok else np.nan
    return pd.Series(prices, name="price_$per_MWh")


def load_weighted_average(prices: pd.Series, inputs: Inputs) -> float:
    d = np.array(inputs.load)
    p = prices.to_numpy(dtype=float)
    ok = ~np.isnan(p)
    return float((p[ok] * d[ok]).sum() / d[ok].sum()) if ok.any() else float("nan")


# ======================================================================================
# Phase 3: sensitivity analysis
# ======================================================================================
def gas_sweep(inputs: Inputs, gas_prices=range(2, 11), reserve_frac: float = RESERVE_FRACTION) -> pd.DataFrame:
    rows = []
    for g in gas_prices:
        uc = model.run_uc(inputs, float(g), reserve_frac=reserve_frac)
        if not uc.info.optimal:
            rows.append({"gas_price": g, "total_cost": np.nan})
            continue
        prices = uc_marginal_prices(inputs, uc.commitment, float(g), reserve_frac)
        shares = dispatch_shares(uc.dispatch, inputs)
        row = {
            "gas_price": g,
            "total_cost": uc.info.objective,
            "avg_marginal_price": load_weighted_average(prices, inputs),
            "starts": count_starts(uc.commitment, inputs),
            "curtailed_mwh": float(uc.dispatch["WIND_CURTAILED"].sum()),
            "solve_seconds": uc.info.solve_seconds,
        }
        row.update({f"share_{u}": float(shares[u]) for u in shares.index})
        rows.append(row)
    return pd.DataFrame(rows).set_index("gas_price")


def breakeven_check(inputs: Inputs, unit_a: str = "ST1", unit_b: str = "GT1", span: float = 1.0,
                    step: float = 0.2) -> dict:
    """Compute by hand the gas price where two units swap merit order, then confirm the economic
    dispatch flips around it.

    By hand: hr_a*g + vom_a = hr_b*g + vom_b  ->  g* = (vom_b - vom_a) / (hr_a - hr_b).
    Model: run ED on a grid around g* and watch which unit's energy rises and which falls.
    """
    a, b = inputs.thermal.loc[unit_a], inputs.thermal.loc[unit_b]
    g_star = breakeven_gas_price(a, b)
    if g_star is None:
        raise ValueError("units have equal heat rates; no crossing")
    grid = np.round(np.arange(g_star - span, g_star + span + 1e-9, step), 3)
    rows = []
    for g in grid:
        ed = model.run_ed(inputs, float(g))
        mc = marginal_costs(inputs, float(g))
        rows.append({
            "gas_price": g,
            f"mc_{unit_a}": mc[unit_a],
            f"mc_{unit_b}": mc[unit_b],
            "cheaper_unit": unit_a if mc[unit_a] < mc[unit_b] else unit_b,
            f"mwh_{unit_a}": float(ed.dispatch[unit_a].sum()),
            f"mwh_{unit_b}": float(ed.dispatch[unit_b].sum()),
        })
    table = pd.DataFrame(rows).set_index("gas_price")
    lo, hi = table.iloc[0], table.iloc[-1]
    flipped = bool(hi[f"mwh_{unit_a}"] > lo[f"mwh_{unit_a}"] and hi[f"mwh_{unit_b}"] < lo[f"mwh_{unit_b}"])
    # first grid price where the cheaper unit changes
    switch = table.index[(table["cheaper_unit"] != table["cheaper_unit"].iloc[0]).to_numpy()]
    return {"breakeven_by_hand": float(g_star), "table": table, "flipped": flipped,
            "first_price_after_flip": float(switch[0]) if len(switch) else None}


def wind_scenario(inputs: Inputs, base: model.Result, gas_price: float = BASE_GAS_PRICE,
                  wind_scale: float = 2.0) -> dict:
    alt = model.run_uc(inputs, gas_price, wind_scale=wind_scale)
    if not alt.info.optimal:
        return {"feasible": False}
    return {
        "feasible": True,
        "base_cost": base.info.objective, "alt_cost": alt.info.objective,
        "base_curtailed_mwh": float(base.dispatch["WIND_CURTAILED"].sum()),
        "alt_curtailed_mwh": float(alt.dispatch["WIND_CURTAILED"].sum()),
        "base_starts": count_starts(base.commitment, inputs),
        "alt_starts": count_starts(alt.commitment, inputs),
    }


def reserve_scenario(inputs: Inputs, base: model.Result, gas_price: float = BASE_GAS_PRICE,
                     reserve_frac: float = 0.15) -> dict:
    alt = model.run_uc(inputs, gas_price, reserve_frac=reserve_frac)
    if not alt.info.optimal:
        return {"feasible": False, "termination": alt.info.termination}
    return {
        "feasible": True,
        "base_cost": base.info.objective, "alt_cost": alt.info.objective,
        "cost_increase": alt.info.objective - base.info.objective,
        "cost_increase_pct": 100 * (alt.info.objective / base.info.objective - 1),
        "base_starts": count_starts(base.commitment, inputs),
        "alt_starts": count_starts(alt.commitment, inputs),
    }


# ======================================================================================
# Charts
# ======================================================================================
_COLORS = {
    "NUC1": "#7b6fb0", "WIND": "#5aa9a3", "CCGT1": "#4f81bd", "CCGT2": "#7ba7d9", "ST1": "#9a9a9a",
    "GT1": "#e0a030", "GT2": "#d97b2a", "OIL1": "#a5382d",
}


def plot_dispatch_stack(dispatch: pd.DataFrame, inputs: Inputs, path, price: pd.Series | None = None,
                        title: str = "Hourly dispatch by unit") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["NUC1", "WIND"] + inputs.units
    hours = list(dispatch.index)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.stackplot(hours, [dispatch[c] for c in order], labels=order, colors=[_COLORS[c] for c in order], alpha=0.9)
    ax.plot(hours, inputs.load, color="black", lw=2, label="Load")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MW")
    ax.set_title(title)
    ax.set_xlim(hours[0], hours[-1])
    ax.grid(alpha=0.25)
    if price is not None:
        ax2 = ax.twinx()
        ax2.step(hours, price.reindex(hours), where="mid", color="crimson", lw=1.5, ls="--", label="Marginal price")
        ax2.set_ylabel("Marginal price, $/MWh")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, ncol=2, fontsize=8, loc="upper left")
    else:
        ax.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_commitment_heatmap(commitment: pd.DataFrame, path, title: str = "Commitment schedule (1 = on)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.imshow(commitment.T.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(commitment.columns)))
    ax.set_yticklabels(commitment.columns)
    ax.set_xticks(range(0, len(commitment.index), 2))
    ax.set_xticklabels(list(commitment.index)[::2])
    ax.set_xlabel("Hour of day")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gas_sweep(sweep: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(sweep.index, sweep["total_cost"] / 1000, marker="o")
    axes[0].set_title("Total daily cost")
    axes[0].set_ylabel("$ thousands")
    axes[1].plot(sweep.index, sweep["avg_marginal_price"], marker="o", color="crimson")
    axes[1].set_title("Load-weighted marginal price")
    axes[1].set_ylabel("$/MWh")
    share_cols = [c for c in sweep.columns if c.startswith("share_")]
    axes[2].stackplot(sweep.index, [sweep[c] * 100 for c in share_cols],
                      labels=[c.replace("share_", "") for c in share_cols],
                      colors=[_COLORS[c.replace("share_", "")] for c in share_cols])
    axes[2].set_title("Energy share by unit")
    axes[2].set_ylabel("% of daily energy")
    axes[2].legend(fontsize=7, ncol=2, loc="lower left")
    for ax in axes:
        ax.set_xlabel("Gas price, $/MMBtu")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
