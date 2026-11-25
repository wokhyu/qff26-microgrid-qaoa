"""Interactive demo: dispatch a rural microgrid with QAOA and compare with classical methods.

Run from the repository root:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microgrid import (  # noqa: E402
    brute_force,
    build_qubo,
    dispatch_report,
    greedy_merit_order,
    make_microgrid,
    prepare,
    probability_of_optimal,
    sample_best,
    simulated_annealing,
    solve_layerwise,
)
from microgrid.qaoa import top_bitstrings  # noqa: E402

KIND_COLORS = {"solar": "#E8A400", "wind": "#2F8FCE", "battery": "#7A5CF0", "diesel": "#5E5E5E"}

st.set_page_config(page_title="Quantum Microgrid Dispatch", layout="wide")


@st.cache_data(show_spinner=False)
def solve(n: int, density: float, seed: int, demand_ratio: float, max_p: int, objective: str, shots: int):
    grid = make_microgrid(n, density, seed, demand_ratio=demand_ratio)
    qubo = build_qubo(grid)
    problem = prepare(qubo)
    exact, land = brute_force(qubo)
    greedy = greedy_merit_order(grid, qubo)
    sa = simulated_annealing(qubo, seed=seed)
    t0 = time.perf_counter()
    results = solve_layerwise(problem, max_p, objective=objective, maxiter=200)
    qaoa_time = time.perf_counter() - t0
    final = results[-1]
    x_q, e_q = sample_best(problem, final.probs, shots, seed=seed)
    return {
        "grid": grid,
        "qubo_couplings": qubo.num_couplings(),
        "energies": problem.energies,
        "land": land,
        "solutions": {
            "QAOA (best of shots)": (x_q, e_q, qaoa_time),
            "Exact (brute force)": (exact.x, exact.energy, exact.runtime_s),
            "Greedy merit-order": (greedy.x, greedy.energy, greedy.runtime_s),
            "Simulated annealing": (sa.x, sa.energy, sa.runtime_s),
        },
        "results": results,
    }


def draw_grid(grid, x, title: str):
    supply = grid.share_matrix() @ np.asarray(x, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    pos = grid.positions
    for i, j in grid.edges:
        ax.plot(*zip(pos[i], pos[j]), color="#B9C2CC", lw=1.6, zorder=1)
    for j in range(grid.n):
        color = KIND_COLORS[grid.kinds[j]]
        on = bool(x[j])
        ax.scatter(*pos[j], s=8 * grid.capacity_kw[j] + 60, color=color if on else "white",
                   edgecolors=color, linewidths=2.2, zorder=3)
        served = min(1.0, supply[j] / grid.demand_kw[j])
        ax.annotate(f"{j}\n{served:.0%}", pos[j], xytext=(0, -26), textcoords="offset points",
                    ha="center", fontsize=8, color="#222")
    for kind, color in KIND_COLORS.items():
        ax.scatter([], [], color=color, s=60, label=kind)
    ax.scatter([], [], color="white", edgecolors="#555", s=60, label="idle (hollow)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.12, 1.08)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


st.title("Quantum-assisted microgrid dispatch")
st.caption(
    "Which solar panels, wind turbines, batteries and diesel generators should a village microgrid switch on "
    "this hour, so every household gets power, little clean energy is wasted, and diesel stays off? "
    "Formulated as a QUBO and solved with QAOA in Qiskit. Synthetic data."
)

with st.sidebar:
    st.header("Microgrid")
    n = st.slider("Nodes (= qubits)", 6, 12, 8)
    density = st.slider("Feeder density", 0.0, 1.0, 0.3, 0.05)
    seed = st.number_input("Instance seed", 0, 999, 5)
    demand_ratio = st.slider("Demand / total capacity", 0.3, 0.9, 0.6, 0.05)
    st.header("QAOA")
    max_p = st.slider("Layers p", 1, 3, 2)
    objective = st.radio("Objective", ["expectation", "cvar"], horizontal=True)
    shots = st.select_slider("Shots", [128, 256, 512, 1024, 4096], value=1024)

with st.spinner("Optimising QAOA on the Qiskit statevector simulator..."):
    data = solve(n, density, int(seed), demand_ratio, max_p, objective, shots)

grid, land, solutions = data["grid"], data["land"], data["solutions"]
exact_e = land.e_min

left, right = st.columns([1.1, 1])
with left:
    choice = st.radio("Show dispatch from", list(solutions), horizontal=True)
    x, e, _ = solutions[choice]
    st.pyplot(draw_grid(grid, x, f"{choice}: filled = switched on, % = demand served"))

with right:
    st.subheader("Compare methods")
    table = []
    for name, (xs, es, rt) in solutions.items():
        rep = dispatch_report(grid, xs)
        table.append({
            "method": name,
            "optimal": "yes" if np.isclose(es, exact_e) else "no",
            "demand met %": round(rep["pct_demand_met"], 1),
            "surplus kW": round(rep["surplus_kw"], 1),
            "diesel kW": round(rep["diesel_kw"], 1),
            "cost USD/h": round(rep["cost_usd_per_h"], 2),
            "runtime s": round(rt, 3),
        })
    st.dataframe(table, hide_index=True, width="stretch")

    final = data["results"][-1]
    p_opt = probability_of_optimal(final.probs, land.optimal_indices)
    p_rand = len(land.optimal_indices) / 2**n
    c1, c2, c3 = st.columns(3)
    c1.metric("P(optimal) per shot", f"{p_opt:.3f}")
    c2.metric("vs random guess", f"{p_opt / p_rand:.0f}x")
    c3.metric("QUBO couplings", data["qubo_couplings"])

    st.subheader("QAOA output distribution (top 12)")
    top = top_bitstrings(final.probs, n, 12)
    energies = data["energies"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    colors = ["#2E9E5B" if np.isclose(energies[int(b, 2)], exact_e) else "#9AA5B1" for b, _ in top]
    ax.bar(range(len(top)), [p for _, p in top], color=colors)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([b for b, _ in top], rotation=60, fontsize=7, family="monospace")
    ax.set_ylabel("probability")
    ax.set_title("green = optimal dispatch (rightmost bit = node 0)", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

with st.expander("How it works"):
    st.markdown(
        r"""
Each node $j$ has a resource with capacity $p_j$ and cost $c_j$; $x_j = 1$ switches it on. A dispatched source
shares its output with itself and its direct feeder neighbours, in proportion to their demand ($A_{ij}$).

$$E(x) = \sum_j c_j p_j x_j + \lambda \sum_i \Big(\sum_j A_{ij} x_j - (1+r)\,d_i\Big)^2$$

The first term is generation cost; the second penalises blackouts and wasted energy at every node, with a
15% reserve margin $r$. Expanding gives a QUBO, which maps to an Ising Hamiltonian with $x_j = (1 - Z_j)/2$.
QAOA alternates $e^{-i\gamma H}$ and $e^{+i\beta \sum X}$, warm-starting each extra layer from the previous
optimum (INTERP). Classical baselines: brute force (exact), greedy merit-order dispatch, simulated annealing.
"""
    )
    st.write("QAOA layer results:", [
        {"p": r.reps, "approx. ratio of <E>": round(land.approximation_ratio(r.expectation), 3),
         "P(optimal)": round(probability_of_optimal(r.probs, land.optimal_indices), 4), "evaluations": r.nfev}
        for r in data["results"]
    ])
