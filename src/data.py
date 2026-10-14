"""Inputs for the dispatch / unit-commitment model.

Everything here is an ILLUSTRATIVE assumption (a stylized fleet, not real plants).
The CSVs in data/ are the single source of truth; this module only reads them and
turns them into plain Python dicts that model.py can use.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Base-case gas price, $/MMBtu. Taken from the weekly dashboard's Henry Hub spot for the
# week ending 2026-10-02 (EIA). Change it here or pass gas_price= to the model builders.
BASE_GAS_PRICE = 2.90
RESERVE_FRACTION = 0.10  # spinning-reserve requirement as a share of hourly demand


@dataclass
class Inputs:
    """Plain-Python view of the CSVs.

    thermal      : dispatchable units (no nuclear, no wind), one row per unit
    nuclear_mw   : fixed must-run nuclear output
    nuclear_mc   : nuclear marginal cost, $/MWh
    load         : list of 24 hourly demands, MW
    wind_avail   : list of 24 hourly wind availabilities, MW
    """

    thermal: pd.DataFrame
    nuclear_mw: float
    nuclear_mc: float
    load: list[float]
    wind_avail: list[float]

    @property
    def units(self) -> list[str]:
        return list(self.thermal.index)

    @property
    def hours(self) -> list[int]:
        return list(range(len(self.load)))


def load_inputs(data_dir: Path | str = DATA_DIR) -> Inputs:
    data_dir = Path(data_dir)
    fleet = pd.read_csv(data_dir / "fleet.csv").set_index("unit")
    load = pd.read_csv(data_dir / "load_24h.csv")["load_mw"].astype(float).tolist()
    wind = pd.read_csv(data_dir / "wind_24h.csv")["wind_avail_mw"].astype(float).tolist()
    if len(load) != 24 or len(wind) != 24:
        raise ValueError("load_24h.csv and wind_24h.csv must each have 24 rows")

    nuc = fleet.loc["NUC1"]
    thermal = fleet[~fleet["type"].isin(["nuclear", "wind"])].copy()
    return Inputs(
        thermal=thermal,
        nuclear_mw=float(nuc["pmax"]),
        nuclear_mc=float(nuc["fixed_mc"]),
        load=load,
        wind_avail=wind,
    )


def marginal_cost(row: pd.Series, gas_price: float) -> float:
    """$/MWh. Gas units: heat rate x gas price + VOM. Oil peaker: a fixed $/MWh."""
    if pd.notna(row["fixed_mc"]):
        return float(row["fixed_mc"])
    return float(row["heat_rate"] * gas_price + row["vom"])


def marginal_costs(inputs: Inputs, gas_price: float) -> dict[str, float]:
    return {g: marginal_cost(inputs.thermal.loc[g], gas_price) for g in inputs.units}


def merit_order(inputs: Inputs, gas_price: float) -> list[str]:
    """Dispatchable units sorted from cheapest to most expensive marginal cost."""
    mc = marginal_costs(inputs, gas_price)
    return sorted(mc, key=mc.get)


def breakeven_gas_price(row_a: pd.Series, row_b: pd.Series) -> float | None:
    """Gas price where two gas-fired units have equal marginal cost.

    hr_a*g + vom_a = hr_b*g + vom_b  =>  g = (vom_b - vom_a) / (hr_a - hr_b)
    Returns None if the lines are parallel.
    """
    d_hr = float(row_a["heat_rate"] - row_b["heat_rate"])
    if d_hr == 0:
        return None
    return float((row_b["vom"] - row_a["vom"]) / d_hr)
