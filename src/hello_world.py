"""Setup check: 2 generators, 1 hour, 100 MW of demand. The cheaper unit should dispatch first.

Run:  python -m src.hello_world
"""
import pyomo.environ as pyo

m = pyo.ConcreteModel()
m.G = pyo.Set(initialize=["cheap", "expensive"])
cost = {"cheap": 20.0, "expensive": 50.0}   # $/MWh
pmax = {"cheap": 80.0, "expensive": 80.0}   # MW each, so one unit alone cannot cover 100 MW

m.p = pyo.Var(m.G, bounds=lambda m, g: (0, pmax[g]))
m.balance = pyo.Constraint(expr=sum(m.p[g] for g in m.G) == 100.0)
m.cost = pyo.Objective(expr=sum(cost[g] * m.p[g] for g in m.G), sense=pyo.minimize)

pyo.SolverFactory("appsi_highs").solve(m)
for g in m.G:
    print(f"{g:10s} {pyo.value(m.p[g]):6.1f} MW")
print(f"total cost ${pyo.value(m.cost):,.0f}")  # expect cheap=80, expensive=20, cost=2,600
