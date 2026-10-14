"""Reproduce every result in the README with one command:  python run_all.py

Writes charts and tables to docs/ and prints a summary you can copy into the README.
"""
import json
from pathlib import Path

import pandas as pd

from src import analysis as an
from src import model
from src.data import BASE_GAS_PRICE, RESERVE_FRACTION, load_inputs, marginal_costs, merit_order

DOCS = Path(__file__).parent / "docs"
DOCS.mkdir(exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 8 + f" {text} " + "=" * 8)


def main() -> dict:
    inputs = load_inputs()
    gas = BASE_GAS_PRICE
    results: dict = {"gas_price": gas, "reserve_fraction": RESERVE_FRACTION}

    banner("Marginal costs and merit order at base gas price")
    mc = marginal_costs(inputs, gas)
    print(pd.Series(mc).round(2).sort_values().to_string())
    print("merit order:", " < ".join(merit_order(inputs, gas)))

    # ---------------- Phase 1: economic dispatch ----------------
    banner("Phase 1: economic dispatch (LP)")
    ed = model.run_ed(inputs, gas)
    assert ed.info.optimal, f"ED did not solve: {ed.info.termination}"
    bad = an.merit_order_violations(ed.dispatch, inputs, gas)
    print(f"ED total cost: ${ed.info.objective:,.0f}   merit-order violations: {len(bad)}")
    assert not bad, f"merit-order test failed: {bad[:5]}"
    ed_prices = an.ed_marginal_prices(inputs, gas)
    marg = an.marginal_unit_ed(ed.dispatch, inputs, gas)
    ed_table = pd.DataFrame({"marginal_price": ed_prices.round(2), "marginal_unit": marg})
    ed_table["unit_mc"] = [mc.get(u, 0.0) for u in marg]
    print(ed_table.to_string())
    results["ed_cost"] = ed.info.objective
    results["ed_peak_hour_price"] = float(ed_prices.iloc[int(pd.Series(inputs.load).idxmax())])
    an.plot_dispatch_stack(ed.dispatch, inputs, DOCS / "ed_dispatch.png", price=ed_prices,
                           title="Economic dispatch: hourly output and marginal price")
    ed.dispatch.to_csv(DOCS / "ed_dispatch.csv")
    ed_table.to_csv(DOCS / "ed_prices.csv")

    # ---------------- Phase 2: unit commitment ----------------
    banner("Phase 2: unit commitment (MILP)")
    uc = model.run_uc(inputs, gas)
    assert uc.info.optimal, f"UC did not solve: {uc.info.termination}"
    i = uc.info
    print(f"UC total cost: ${i.objective:,.0f}   solve time: {i.solve_seconds:.2f}s   "
          f"MIP gap: {i.mip_gap}   variables: {i.n_variables}   constraints: {i.n_constraints}")
    checks = an.verify_uc_solution(uc.dispatch, uc.commitment, inputs)
    print(checks.to_string())
    assert checks["passed"].all(), "a constraint check failed"
    assert uc.info.objective >= ed.info.objective - 1e-6, "UC cost must be >= ED cost"
    print(f"UC - ED cost gap: ${uc.info.objective - ed.info.objective:,.0f}")
    print("starts:", an.count_starts(uc.commitment, inputs))
    results.update(uc_cost=i.objective, uc_solve_seconds=i.solve_seconds, uc_mip_gap=i.mip_gap,
                   uc_variables=i.n_variables, uc_constraints=i.n_constraints,
                   uc_starts=an.count_starts(uc.commitment, inputs),
                   uc_curtailed_mwh=float(uc.dispatch["WIND_CURTAILED"].sum()))
    uc_prices = an.uc_marginal_prices(inputs, uc.commitment, gas)
    print("UC (fixed-commitment) prices:", uc_prices.round(2).tolist())
    results["uc_avg_marginal_price"] = an.load_weighted_average(uc_prices, inputs)
    an.plot_dispatch_stack(uc.dispatch, inputs, DOCS / "uc_dispatch.png", price=uc_prices,
                           title="Unit commitment: hourly output and marginal price")
    an.plot_commitment_heatmap(uc.commitment, DOCS / "commitment_heatmap.png")
    uc.dispatch.to_csv(DOCS / "uc_dispatch.csv")
    uc.commitment.to_csv(DOCS / "uc_commitment.csv")

    # ---------------- Phase 3: analysis ----------------
    banner("Phase 3: gas price sweep")
    sweep = an.gas_sweep(inputs)
    print(sweep.round(3).to_string())
    sweep.to_csv(DOCS / "gas_sweep.csv")
    an.plot_gas_sweep(sweep, DOCS / "gas_sweep.png")
    results["sweep_avg_price_at_2"] = float(sweep.loc[2, "avg_marginal_price"])
    results["sweep_avg_price_at_10"] = float(sweep.loc[10, "avg_marginal_price"])

    banner("Break-even check (ST1 vs GT1)")
    be = an.breakeven_check(inputs, "ST1", "GT1")
    print(f"by hand: g* = (vom_GT1 - vom_ST1) / (hr_ST1 - hr_GT1) = {be['breakeven_by_hand']:.2f} $/MMBtu")
    print(be["table"].round(2).to_string())
    print("model flips around g*:", be["flipped"], "| first grid price after flip:", be["first_price_after_flip"])
    be["table"].to_csv(DOCS / "breakeven.csv")
    results["breakeven_by_hand"] = be["breakeven_by_hand"]
    results["breakeven_model_flips"] = be["flipped"]

    banner("Wind scenario: 2x wind capacity")
    ws = an.wind_scenario(inputs, uc, gas, wind_scale=2.0)
    print(ws)
    results["wind_2x"] = ws

    banner("Reserve scenario: 15%")
    rs = an.reserve_scenario(inputs, uc, gas, reserve_frac=0.15)
    print(rs)
    results["reserve_15pct"] = rs

    banner("Forecast miss: commit on a 3% wrong demand forecast, then serve the actual demand")
    fm = an.forecast_miss_scenario(inputs, uc, gas, miss=0.03)
    for label, row in fm.items():
        print(label, row)
    results["forecast_miss_3pct"] = fm

    (DOCS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved charts, tables and results.json to {DOCS}")
    return results


if __name__ == "__main__":
    main()
