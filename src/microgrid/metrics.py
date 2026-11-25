"""Deployment metrics (what a community cares about) and circuit metrics."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .instance import Microgrid


def dispatch_report(grid: Microgrid, x: np.ndarray) -> dict:
    """Physical outcome of a dispatch decision for one hour."""
    x = np.asarray(x, dtype=float)
    supply = grid.share_matrix() @ x
    demand = grid.demand_kw
    served = np.minimum(supply, demand)
    total_demand = float(demand.sum())
    return {
        "dispatched_nodes": [int(j) for j in np.flatnonzero(x)],
        "cost_usd_per_h": float(np.sum(grid.cost_per_kwh * grid.capacity_kw * x)),
        "demand_kw": total_demand,
        "served_kw": float(served.sum()),
        "pct_demand_met": 100.0 * float(served.sum()) / total_demand,
        "unmet_kw": float(np.maximum(demand - supply, 0.0).sum()),
        "surplus_kw": float(np.maximum(supply - demand, 0.0).sum()),
        "nodes_fully_served": int(np.sum(supply >= demand - 1e-9)),
        "diesel_kw": float(sum(grid.capacity_kw[j] * x[j] for j in range(grid.n) if grid.kinds[j] == "diesel")),
    }


def distribution_from_counts(counts: dict[str, int], n: int) -> np.ndarray:
    """Counts keyed by Qiskit bitstring -> probability vector over 2^n basis states."""
    probs = np.zeros(2**n)
    total = sum(counts.values())
    for bitstring, c in counts.items():
        probs[int(bitstring, 2)] += c / total
    return probs


def expectation(probs: np.ndarray, energies: np.ndarray) -> float:
    return float(probs @ energies)


def cvar(probs: np.ndarray, energies: np.ndarray, alpha: float) -> float:
    """Mean energy of the best alpha-fraction of the distribution (Barkoutsos et al. 2020)."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    order = np.argsort(energies, kind="stable")
    p = probs[order]
    cum = np.cumsum(p)
    # Weight of each state inside the alpha tail.
    take = np.clip(alpha - (cum - p), 0.0, p)
    return float(take @ energies[order] / alpha)


def probability_of_optimal(probs: np.ndarray, optimal_indices: np.ndarray) -> float:
    return float(probs[optimal_indices].sum())


def circuit_stats(circuit: QuantumCircuit) -> dict:
    """Depth and two-qubit cost of a (usually transpiled) circuit."""
    def is_gate(inst) -> bool:
        return inst.operation.name not in ("barrier", "measure", "delay")

    def is_two_qubit(inst) -> bool:
        return is_gate(inst) and inst.operation.num_qubits == 2

    return {
        "depth": int(circuit.depth(lambda inst: inst.operation.name != "barrier")),
        "two_qubit_gates": sum(1 for inst in circuit.data if is_two_qubit(inst)),
        "two_qubit_depth": int(circuit.depth(is_two_qubit)),
        "num_qubits_used": len({circuit.find_bit(q).index for inst in circuit.data if is_gate(inst) for q in inst.qubits}),
    }
