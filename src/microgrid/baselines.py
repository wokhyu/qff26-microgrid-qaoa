"""Classical baselines for the dispatch QUBO."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .instance import Microgrid
from .qubo import QUBO, all_bitstrings


@dataclass(frozen=True)
class Solution:
    method: str
    x: np.ndarray
    energy: float
    runtime_s: float


@dataclass(frozen=True)
class Landscape:
    """Every bitstring's energy. Feasible for n <= ~22 and used as ground truth."""

    energies: np.ndarray
    e_min: float
    e_max: float
    optimal_indices: np.ndarray

    def approximation_ratio(self, energy: float) -> float:
        """1 = optimal, 0 = worst bitstring."""
        span = self.e_max - self.e_min
        return 1.0 if span == 0 else float((self.e_max - energy) / span)


def brute_force(qubo: QUBO, tol: float = 1e-9) -> tuple[Solution, Landscape]:
    start = time.perf_counter()
    bits = all_bitstrings(qubo.n)
    energies = qubo.energies(bits)
    k = int(np.argmin(energies))
    elapsed = time.perf_counter() - start
    e_min, e_max = float(energies.min()), float(energies.max())
    optimal = np.flatnonzero(energies <= e_min + tol * max(1.0, abs(e_min)))
    return (
        Solution("exact (brute force)", bits[k].copy(), e_min, elapsed),
        Landscape(energies, e_min, e_max, optimal),
    )


def greedy_merit_order(grid: Microgrid, qubo: QUBO) -> Solution:
    """What a grid operator does by hand: dispatch the cheapest sources first.

    Walk the sources from lowest to highest cost per kWh and switch each one on
    only if that lowers the objective.
    """
    start = time.perf_counter()
    x = np.zeros(qubo.n, dtype=np.int8)
    best = qubo.energy(x)
    for j in np.argsort(grid.cost_per_kwh, kind="stable"):
        x[j] = 1
        e = qubo.energy(x)
        if e < best:
            best = e
        else:
            x[j] = 0
    return Solution("greedy merit-order", x, best, time.perf_counter() - start)


def simulated_annealing(
    qubo: QUBO,
    sweeps: int = 500,
    restarts: int = 8,
    t_start: float | None = None,
    t_end: float | None = None,
    seed: int = 0,
) -> Solution:
    """Single-flip Metropolis annealing with a geometric temperature schedule."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = qubo.n
    h = qubo.linear
    J = qubo.symmetric()

    # Temperatures scaled to the typical size of a single-flip energy change.
    scale = float(np.mean(np.abs(h)) + np.mean(np.abs(J).sum(axis=1))) or 1.0
    t_start = 2.0 * scale if t_start is None else t_start
    t_end = 1e-3 * scale if t_end is None else t_end
    temps = np.geomspace(t_start, t_end, sweeps)

    best_x, best_e = None, np.inf
    for _ in range(restarts):
        x = rng.integers(0, 2, size=n).astype(float)
        e = qubo.energy(x)
        for t in temps:
            for j in rng.permutation(n):
                delta = (1.0 - 2.0 * x[j]) * (h[j] + J[j] @ x)
                if delta <= 0.0 or rng.random() < np.exp(-delta / t):
                    x[j] = 1.0 - x[j]
                    e += delta
        if e < best_e:
            best_e, best_x = e, x.astype(np.int8)
    return Solution("simulated annealing", best_x, qubo.energy(best_x), time.perf_counter() - start)
