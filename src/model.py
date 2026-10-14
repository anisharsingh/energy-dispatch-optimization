"""Pyomo models: economic dispatch (LP) and unit commitment (MILP).

Index conventions
    g in units   dispatchable units (CCGT1, CCGT2, ST1, GT1, GT2, OIL1)
    t in hours   0..23

Nuclear is a constant (500 MW at $8/MWh) outside the decision variables.
Wind is a variable p_wind[t] in [0, availability[t]] at zero marginal cost (so it can be curtailed).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import pyomo.environ as pyo

from .data import BASE_GAS_PRICE, RESERVE_FRACTION, Inputs, marginal_costs

SOLVER_NAME = "appsi_highs"


@dataclass
class SolveInfo:
    optimal: bool
    termination: str
    objective: float | None
    solve_seconds: float
    mip_gap: float | None
    n_variables: int
    n_constraints: int


# --------------------------------------------------------------------------------------
# Phase 1: economic dispatch (LP)
# --------------------------------------------------------------------------------------
def build_ed(inputs: Inputs, gas_price: float = BASE_GAS_PRICE, demand: list[float] | None = None,
             wind_scale: float = 1.0) -> pyo.ConcreteModel:
    """Minimize sum(mc[g] * p[g,t]) s.t. generation = demand each hour.

    No Pmin, no commitment, no ramps: every unit can run anywhere in [0, Pmax].
    """
    D = list(demand) if demand is not None else list(inputs.load)
    mc = marginal_costs(inputs, gas_price)
    pmax = inputs.thermal["pmax"].to_dict()
    units, hours = inputs.units, inputs.hours

    m = pyo.ConcreteModel("economic_dispatch")
    m.G = pyo.Set(initialize=units, ordered=True)
    m.T = pyo.Set(initialize=hours, ordered=True)

    m.p = pyo.Var(m.G, m.T, bounds=lambda m, g, t: (0, pmax[g]))
    m.p_wind = pyo.Var(m.T, bounds=lambda m, t: (0, inputs.wind_avail[t] * wind_scale))

    def balance_rule(m, t):
        return sum(m.p[g, t] for g in m.G) + m.p_wind[t] + inputs.nuclear_mw == D[t]

    m.balance = pyo.Constraint(m.T, rule=balance_rule)

    m.cost = pyo.Objective(
        expr=sum(mc[g] * m.p[g, t] for g in m.G for t in m.T) + inputs.nuclear_mc * inputs.nuclear_mw * len(hours),
        sense=pyo.minimize,
    )
    return m


# --------------------------------------------------------------------------------------
# Phase 2: unit commitment (MILP)
# --------------------------------------------------------------------------------------
def build_uc(inputs: Inputs, gas_price: float = BASE_GAS_PRICE, demand: list[float] | None = None,
             reserve_frac: float = RESERVE_FRACTION, wind_scale: float = 1.0) -> pyo.ConcreteModel:
    """Unit commitment with u (on), v (start-up), w (shut-down) binaries per unit-hour.

    Simplifications (stated in the README): single bus (no transmission), no start-up/shut-down
    ramp trajectories, perfect foresight, stylized fleet, and a starting state that already
    satisfies min up/down times so there is no carry-over from before hour 0.
    """
    D = list(demand) if demand is not None else list(inputs.load)
    mc = marginal_costs(inputs, gas_price)
    th = inputs.thermal
    pmax, pmin = th["pmax"].to_dict(), th["pmin"].to_dict()
    noload, startup = th["noload"].to_dict(), th["startup"].to_dict()
    ut, dt = th["min_up"].astype(int).to_dict(), th["min_down"].astype(int).to_dict()
    ramp = th["ramp"].to_dict()
    u0, p0 = th["init_on"].astype(int).to_dict(), th["init_p"].to_dict()
    units, hours = inputs.units, inputs.hours
    first = hours[0]

    m = pyo.ConcreteModel("unit_commitment")
    m.G = pyo.Set(initialize=units, ordered=True)
    m.T = pyo.Set(initialize=hours, ordered=True)

    m.p = pyo.Var(m.G, m.T, bounds=lambda m, g, t: (0, pmax[g]))
    m.u = pyo.Var(m.G, m.T, within=pyo.Binary)  # 1 if unit is on
    m.v = pyo.Var(m.G, m.T, within=pyo.Binary)  # 1 if unit starts up in hour t
    m.w = pyo.Var(m.G, m.T, within=pyo.Binary)  # 1 if unit shuts down in hour t
    m.p_wind = pyo.Var(m.T, bounds=lambda m, t: (0, inputs.wind_avail[t] * wind_scale))

    # (1) demand balance
    m.balance = pyo.Constraint(
        m.T, rule=lambda m, t: sum(m.p[g, t] for g in m.G) + m.p_wind[t] + inputs.nuclear_mw == D[t]
    )

    # (2) output limits tied to commitment: Pmin*u <= p <= Pmax*u
    m.p_min = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.p[g, t] >= pmin[g] * m.u[g, t])
    m.p_max = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.p[g, t] <= pmax[g] * m.u[g, t])

    # (3) state logic: u[t] - u[t-1] = v[t] - w[t]   (u[-1] is the initial status)
    def state_rule(m, g, t):
        prev = u0[g] if t == first else m.u[g, t - 1]
        return m.u[g, t] - prev == m.v[g, t] - m.w[g, t]

    m.state = pyo.Constraint(m.G, m.T, rule=state_rule)

    # (4) min up time: starts in the last UT hours (including t) <= u[t]
    def min_up_rule(m, g, t):
        window = range(max(first, t - ut[g] + 1), t + 1)
        return sum(m.v[g, tau] for tau in window) <= m.u[g, t]

    m.min_up = pyo.Constraint(m.G, m.T, rule=min_up_rule)

    # (5) min down time: shut-downs in the last DT hours (including t) <= 1 - u[t]
    def min_down_rule(m, g, t):
        window = range(max(first, t - dt[g] + 1), t + 1)
        return sum(m.w[g, tau] for tau in window) <= 1 - m.u[g, t]

    m.min_down = pyo.Constraint(m.G, m.T, rule=min_down_rule)

    # (6) ramping; the Pmax*v / Pmax*w terms relax the limit in start-up / shut-down hours
    def ramp_up_rule(m, g, t):
        prev_p = p0[g] if t == first else m.p[g, t - 1]
        return m.p[g, t] - prev_p <= ramp[g] + pmax[g] * m.v[g, t]

    def ramp_down_rule(m, g, t):
        prev_p = p0[g] if t == first else m.p[g, t - 1]
        return prev_p - m.p[g, t] <= ramp[g] + pmax[g] * m.w[g, t]

    m.ramp_up = pyo.Constraint(m.G, m.T, rule=ramp_up_rule)
    m.ramp_down = pyo.Constraint(m.G, m.T, rule=ramp_down_rule)

    # (7) reserve: headroom of committed dispatchable units >= reserve_frac * demand
    m.reserve = pyo.Constraint(
        m.T,
        rule=lambda m, t: sum(pmax[g] * m.u[g, t] - m.p[g, t] for g in m.G) >= reserve_frac * D[t],
    )

    # objective: fuel/VOM + no-load + start-up (+ constant nuclear energy cost)
    m.cost = pyo.Objective(
        expr=sum(
            mc[g] * m.p[g, t] + noload[g] * m.u[g, t] + startup[g] * m.v[g, t]
            for g in m.G for t in m.T
        ) + inputs.nuclear_mc * inputs.nuclear_mw * len(hours),
        sense=pyo.minimize,
    )
    return m


# --------------------------------------------------------------------------------------
# Solving and extracting results
# --------------------------------------------------------------------------------------
def solve(m: pyo.ConcreteModel, tee: bool = False, time_limit: float | None = 60,
          mip_rel_gap: float | None = 1e-6) -> SolveInfo:
    """Solve with HiGHS through Pyomo's appsi interface. Never raises on infeasibility;
    inspect SolveInfo.optimal / .termination instead.

    Solver options are passed first; if this Pyomo version rejects them, the solve is retried
    once with HiGHS defaults so a version quirk cannot silently break every run.
    """
    options = {}
    if time_limit is not None:
        options["time_limit"] = time_limit
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = mip_rel_gap

    t0 = time.perf_counter()
    res, last_error = None, None
    for opts in (options, {}):
        opt = pyo.SolverFactory(SOLVER_NAME)
        for key, val in opts.items():
            opt.options[key] = val
        try:
            res = opt.solve(m, tee=tee)
            break
        except Exception as exc:  # appsi raises on some infeasible / no-solution cases
            last_error = exc
    elapsed = time.perf_counter() - t0

    if res is None:
        return SolveInfo(False, f"error: {type(last_error).__name__}: {last_error}", None, elapsed,
                         None, m.nvariables(), m.nconstraints())

    term = _termination(res)
    optimal = ("optimal" in term.lower()) or ("convergence" in term.lower())
    obj = float(pyo.value(m.cost)) if optimal else None
    gap = _mip_gap(res) if optimal else None
    return SolveInfo(optimal, term, obj, elapsed, gap, m.nvariables(), m.nconstraints())


def _termination(res) -> str:
    """Termination condition as a string; Pyomo's legacy and native appsi results differ."""
    try:
        return str(res.solver.termination_condition)
    except AttributeError:
        return str(res.termination_condition)


def _mip_gap(res) -> float | None:
    """Relative MIP gap if the solver reported bounds, else None (always None for LPs)."""
    try:
        ub, lb = res.problem.upper_bound, res.problem.lower_bound
    except AttributeError:
        ub = getattr(res, "best_feasible_objective", None)
        lb = getattr(res, "best_objective_bound", None)
    try:
        if ub is None or lb is None or abs(float(ub)) < 1e-9:
            return None
        return abs(float(ub) - float(lb)) / abs(float(ub))
    except (TypeError, ValueError):
        return None


def dispatch_table(m: pyo.ConcreteModel, inputs: Inputs) -> pd.DataFrame:
    """Hour-by-unit table of MW, plus nuclear, wind used, and wind curtailed."""
    rows = []
    for t in inputs.hours:
        row = {g: pyo.value(m.p[g, t]) for g in inputs.units}
        row["NUC1"] = inputs.nuclear_mw
        row["WIND"] = pyo.value(m.p_wind[t])
        row["WIND_CURTAILED"] = m.p_wind[t].ub - row["WIND"]
        rows.append(row)
    df = pd.DataFrame(rows, index=pd.Index(inputs.hours, name="hour"))
    return df.clip(lower=0).round(6)


def commitment_table(m: pyo.ConcreteModel, inputs: Inputs) -> pd.DataFrame:
    """Hour-by-unit table of u (1 = on)."""
    data = {g: [int(round(pyo.value(m.u[g, t]))) for t in inputs.hours] for g in inputs.units}
    return pd.DataFrame(data, index=pd.Index(inputs.hours, name="hour"))


def starts_table(m: pyo.ConcreteModel, inputs: Inputs) -> pd.DataFrame:
    data = {g: [int(round(pyo.value(m.v[g, t]))) for t in inputs.hours] for g in inputs.units}
    return pd.DataFrame(data, index=pd.Index(inputs.hours, name="hour"))


def fix_commitment(m: pyo.ConcreteModel, commitment: pd.DataFrame, inputs: Inputs) -> None:
    """Fix u (and hence v, w through the state constraints) so the MILP becomes an LP.

    Used to price energy with the commitment held fixed: re-solve with demand + 1 MW.
    """
    prev = {g: int(inputs.thermal.loc[g, "init_on"]) for g in inputs.units}
    for t in inputs.hours:
        for g in inputs.units:
            cur = int(commitment.loc[t, g])
            m.u[g, t].fix(cur)
            m.v[g, t].fix(1 if (cur == 1 and prev[g] == 0) else 0)
            m.w[g, t].fix(1 if (cur == 0 and prev[g] == 1) else 0)
            prev[g] = cur


# --------------------------------------------------------------------------------------
# High-level wrappers (analysis.py only calls these)
# --------------------------------------------------------------------------------------
@dataclass
class Result:
    dispatch: pd.DataFrame | None
    commitment: pd.DataFrame | None
    starts: pd.DataFrame | None
    info: SolveInfo


def run_ed(inputs: Inputs, gas_price: float = BASE_GAS_PRICE, demand: list[float] | None = None,
           wind_scale: float = 1.0) -> Result:
    m = build_ed(inputs, gas_price, demand, wind_scale)
    info = solve(m)
    disp = dispatch_table(m, inputs) if info.optimal else None
    return Result(disp, None, None, info)


def run_uc(inputs: Inputs, gas_price: float = BASE_GAS_PRICE, demand: list[float] | None = None,
           reserve_frac: float = RESERVE_FRACTION, wind_scale: float = 1.0) -> Result:
    m = build_uc(inputs, gas_price, demand, reserve_frac, wind_scale)
    info = solve(m)
    if not info.optimal:
        return Result(None, None, None, info)
    return Result(dispatch_table(m, inputs), commitment_table(m, inputs), starts_table(m, inputs), info)


def run_uc_fixed(inputs: Inputs, commitment: pd.DataFrame, gas_price: float = BASE_GAS_PRICE,
                 demand: list[float] | None = None, reserve_frac: float = RESERVE_FRACTION,
                 wind_scale: float = 1.0) -> Result:
    """UC with the commitment held fixed (an LP). Used to price energy at a given schedule."""
    m = build_uc(inputs, gas_price, demand, reserve_frac, wind_scale)
    fix_commitment(m, commitment, inputs)
    info = solve(m)
    disp = dispatch_table(m, inputs) if info.optimal else None
    return Result(disp, commitment, None, info)
