"""Tests for the dispatch / unit-commitment model.  Run:  pytest

The solver-based tests solve the real model with HiGHS. Two tests (verifier catches a violation,
break-even formula) need no solver.
"""
import numpy as np
import pytest

from src import analysis as an
from src import model
from src.data import BASE_GAS_PRICE, RESERVE_FRACTION, breakeven_gas_price, load_inputs, marginal_costs


@pytest.fixture(scope="module")
def inputs():
    return load_inputs()


@pytest.fixture(scope="module")
def ed(inputs):
    res = model.run_ed(inputs, BASE_GAS_PRICE)
    assert res.info.optimal, res.info.termination
    return res


@pytest.fixture(scope="module")
def uc(inputs):
    res = model.run_uc(inputs, BASE_GAS_PRICE)
    assert res.info.optimal, res.info.termination
    return res


# ---------------- economic dispatch ----------------
def test_ed_balance(inputs, ed):
    gen = ed.dispatch[inputs.units].sum(axis=1) + ed.dispatch["WIND"] + inputs.nuclear_mw
    assert np.allclose(gen.to_numpy(), inputs.load, atol=1e-4)


def test_ed_bounds(inputs, ed):
    for g in inputs.units:
        assert (ed.dispatch[g] >= -1e-6).all()
        assert (ed.dispatch[g] <= inputs.thermal.loc[g, "pmax"] + 1e-6).all()
    assert (ed.dispatch["WIND"] <= np.array(inputs.wind_avail) + 1e-6).all()


def test_ed_merit_order(inputs, ed):
    assert an.merit_order_violations(ed.dispatch, inputs, BASE_GAS_PRICE) == []


def test_ed_price_equals_marginal_unit_cost(inputs, ed):
    prices = an.ed_marginal_prices(inputs, BASE_GAS_PRICE)
    mc = marginal_costs(inputs, BASE_GAS_PRICE)
    units = an.marginal_unit_ed(ed.dispatch, inputs, BASE_GAS_PRICE)
    for t, u in zip(inputs.hours, units):
        assert prices[t] == pytest.approx(mc.get(u, 0.0), abs=1e-3)


# ---------------- unit commitment ----------------
def test_uc_all_constraints_hold(inputs, uc):
    checks = an.verify_uc_solution(uc.dispatch, uc.commitment, inputs, RESERVE_FRACTION)
    assert checks["passed"].all(), checks


def test_uc_cost_at_least_ed_cost(ed, uc):
    # UC adds Pmin, ramps, min up/down, reserve, no-load and start-up costs to ED's problem
    assert uc.info.objective >= ed.info.objective - 1e-6


def test_higher_reserve_never_cheaper(inputs, uc):
    alt = model.run_uc(inputs, BASE_GAS_PRICE, reserve_frac=0.15)
    if alt.info.optimal:
        assert alt.info.objective >= uc.info.objective - 1e-6


# ---------------- infeasibility ----------------
def test_infeasible_demand_is_detected(inputs):
    too_much = [3000.0] * 24  # more than total capacity (about 2,130 MW)
    assert not model.run_ed(inputs, demand=too_much).info.optimal
    assert not model.run_uc(inputs, demand=too_much).info.optimal


# ---------------- solver-free tests ----------------
def test_verifier_catches_a_balance_violation(inputs, uc):
    broken = uc.dispatch.copy()
    broken.loc[5, "CCGT1"] += 50.0  # generates 50 MW more than demand in hour 5
    checks = an.verify_uc_solution(broken, uc.commitment, inputs)
    assert not checks.loc["demand balance", "passed"]


def test_breakeven_formula_for_steam_vs_peaker(inputs):
    g_star = breakeven_gas_price(inputs.thermal.loc["ST1"], inputs.thermal.loc["GT1"])
    assert g_star == pytest.approx(4.4)
    # at the break-even price the two marginal costs are equal
    mc = marginal_costs(inputs, g_star)
    assert mc["ST1"] == pytest.approx(mc["GT1"])


# ---------------- forecast miss ----------------
def test_zero_forecast_miss_costs_nothing(inputs, uc):
    res = an.forecast_miss_scenario(inputs, uc, BASE_GAS_PRICE, miss=0.0)
    for case in res.values():
        assert case["feasible"]
        assert case["extra_cost"] == pytest.approx(0.0, abs=1.0)


def test_forecast_miss_never_beats_perfect_foresight(inputs, uc):
    # the schedule built on a wrong forecast is still a feasible schedule for the real demand,
    # so it cannot cost less than the optimum that knew the real demand
    res = an.forecast_miss_scenario(inputs, uc, BASE_GAS_PRICE, miss=0.03)
    for case in res.values():
        if case.get("feasible"):
            assert case["extra_cost"] >= -1e-6
