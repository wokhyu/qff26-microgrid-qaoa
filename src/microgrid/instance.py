"""Seeded synthetic microgrid instances.

A microgrid has ``n`` nodes connected by a feeder graph. Every node is a small
community (a cluster of households, a clinic, a water pump) with a local load
``demand_kw`` and one dispatchable resource: rooftop solar, a small wind
turbine, a battery bank or a diesel generator. Dispatching node ``j`` makes
``capacity_kw[j]`` available for the hour at ``cost_per_kwh[j]``.

All numbers are synthetic and illustrative. The ranges below are modelling
assumptions, not measured data; replace them with sourced values before using
the model for any real planning decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Resource type -> (capacity range kW, cost range USD/kWh, sampling weight).
# Illustrative assumptions: renewables are cheap to dispatch, batteries carry a
# degradation cost, diesel is expensive on remote sites.
RESOURCE_TYPES: dict[str, tuple[tuple[float, float], tuple[float, float], float]] = {
    "solar": ((20.0, 60.0), (0.03, 0.06), 0.35),
    "wind": ((15.0, 50.0), (0.04, 0.07), 0.20),
    "battery": ((10.0, 40.0), (0.08, 0.12), 0.30),
    "diesel": ((30.0, 80.0), (0.30, 0.45), 0.15),
}


@dataclass(frozen=True)
class Microgrid:
    """One microgrid instance. Arrays are indexed by node (= qubit) number."""

    kinds: tuple[str, ...]
    capacity_kw: np.ndarray
    cost_per_kwh: np.ndarray
    demand_kw: np.ndarray
    edges: tuple[tuple[int, int], ...]
    positions: np.ndarray
    density: float
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.kinds)

    def adjacency(self) -> np.ndarray:
        adj = np.zeros((self.n, self.n), dtype=bool)
        for i, j in self.edges:
            adj[i, j] = adj[j, i] = True
        return adj

    def closed_neighbourhoods(self) -> list[np.ndarray]:
        """N[i] = {i} plus the feeder neighbours of i."""
        adj = self.adjacency()
        return [np.flatnonzero(adj[i] | (np.arange(self.n) == i)) for i in range(self.n)]

    def share_matrix(self) -> np.ndarray:
        """A[i, j] = kW that node j delivers to node i when j is dispatched.

        A dispatched source shares its output across its closed neighbourhood
        (itself and its direct feeder neighbours) in proportion to their
        demand. Power is only delivered one hop, which models the short reach
        of low-voltage rural feeders. Column sums equal ``capacity_kw``, so
        total supply is conserved.
        """
        hoods = self.closed_neighbourhoods()
        share = np.zeros((self.n, self.n))
        for j, hood in enumerate(hoods):
            weights = self.demand_kw[hood]
            share[hood, j] = self.capacity_kw[j] * weights / weights.sum()
        return share


def _minimum_spanning_tree(dist: np.ndarray) -> list[tuple[int, int]]:
    """Prim's algorithm on a dense distance matrix."""
    n = dist.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = dist[0].copy()
    parent = np.zeros(n, dtype=int)
    edges: list[tuple[int, int]] = []
    for _ in range(n - 1):
        candidates = np.where(in_tree, np.inf, best)
        k = int(np.argmin(candidates))
        edges.append((int(min(parent[k], k)), int(max(parent[k], k))))
        in_tree[k] = True
        closer = dist[k] < best
        best = np.where(closer, dist[k], best)
        parent = np.where(closer, k, parent)
    return edges


def make_microgrid(
    n: int,
    density: float = 0.3,
    seed: int = 0,
    demand_ratio: float = 0.6,
) -> Microgrid:
    """Generate a connected microgrid.

    Args:
        n: number of nodes (= qubits).
        density: target feeder edge density in [0, 1], as a fraction of all
            n(n-1)/2 node pairs. The graph is always connected, so the real
            density is at least 2/n.
        seed: RNG seed; the same arguments always give the same instance.
        demand_ratio: total demand as a fraction of total dispatchable capacity.
    """
    if n < 2:
        raise ValueError("need at least 2 nodes")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")

    rng = np.random.default_rng(seed)
    names = list(RESOURCE_TYPES)
    weights = np.array([RESOURCE_TYPES[k][2] for k in names])
    kinds = tuple(rng.choice(names, size=n, p=weights / weights.sum()))

    capacity = np.array([rng.uniform(*RESOURCE_TYPES[k][0]) for k in kinds])
    cost = np.array([rng.uniform(*RESOURCE_TYPES[k][1]) for k in kinds])

    raw_demand = rng.uniform(5.0, 30.0, size=n)
    demand = raw_demand * demand_ratio * capacity.sum() / raw_demand.sum()

    positions = rng.uniform(0.0, 1.0, size=(n, 2))
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)

    # Feeders follow geography: start from the minimum spanning tree, then add
    # the shortest remaining links until the target density is reached.
    edges = set(_minimum_spanning_tree(dist))
    target = max(n - 1, int(round(density * n * (n - 1) / 2)))
    pairs = sorted(
        ((dist[i, j], i, j) for i in range(n) for j in range(i + 1, n)),
        key=lambda t: t[0],
    )
    for _, i, j in pairs:
        if len(edges) >= target:
            break
        edges.add((i, j))

    return Microgrid(
        kinds=kinds,
        capacity_kw=np.round(capacity, 2),
        cost_per_kwh=np.round(cost, 4),
        demand_kw=np.round(demand, 2),
        edges=tuple(sorted(edges)),
        positions=positions,
        density=density,
        seed=seed,
        meta={"demand_ratio": demand_ratio},
    )
