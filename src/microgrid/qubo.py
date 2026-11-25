"""Microgrid dispatch as a QUBO, and the QUBO as an Ising Hamiltonian.

Decision variable: x_j = 1 if node j's resource is dispatched for the hour.

    E(x) = sum_j c_j p_j x_j  +  lam * sum_i ( sum_j A_ij x_j - (1 + r) d_i )^2

The first term is the generation cost in USD/hour. The second term penalises
both unmet demand (blackout risk) and oversupply (wasted clean energy) at every
node i, where A_ij is the power node j delivers to node i (see
``Microgrid.share_matrix``) and r is an operating reserve margin. A quadratic
penalty is symmetric, so the reserve margin is what tilts the optimum towards
keeping the lights on rather than towards curtailment. Two sources only
interact when they can reach a common node, so a sparse feeder graph gives a
sparse QUBO.

Bit convention (same as Qiskit): qubit j holds x_j, and in a Qiskit bitstring
such as "0110" the right-most character is qubit 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .instance import Microgrid


@dataclass(frozen=True)
class QUBO:
    """E(x) = constant + linear @ x + x @ quadratic @ x, quadratic strictly upper."""

    linear: np.ndarray
    quadratic: np.ndarray
    constant: float

    @property
    def n(self) -> int:
        return len(self.linear)

    def energies(self, bits: np.ndarray) -> np.ndarray:
        """Energy of each row of a (m, n) 0/1 matrix."""
        bits = np.atleast_2d(bits).astype(float)
        return self.constant + bits @ self.linear + np.einsum("mi,ij,mj->m", bits, self.quadratic, bits)

    def energy(self, x: np.ndarray) -> float:
        return float(self.energies(np.asarray(x))[0])

    def symmetric(self) -> np.ndarray:
        return self.quadratic + self.quadratic.T

    def num_couplings(self, tol: float = 1e-12) -> int:
        return int(np.count_nonzero(np.abs(self.quadratic) > tol))


DEFAULT_VOLL = 2.0        # value of lost load, USD/kWh (illustrative assumption)
DEFAULT_RESERVE = 0.15    # operating reserve margin (illustrative assumption)


def default_balance_weight(grid: Microgrid, voll_per_kwh: float = DEFAULT_VOLL) -> float:
    """Penalty weight lam in USD/kW^2.

    Chosen so that the marginal penalty for a mismatch of half the mean node
    demand equals the value of lost load:  2 * lam * (mean_d / 2) = VOLL.
    """
    return voll_per_kwh / float(np.mean(grid.demand_kw))


def build_qubo(
    grid: Microgrid,
    voll_per_kwh: float = DEFAULT_VOLL,
    reserve_margin: float = DEFAULT_RESERVE,
    balance_weight: float | None = None,
) -> QUBO:
    lam = default_balance_weight(grid, voll_per_kwh) if balance_weight is None else balance_weight
    share = grid.share_matrix()                     # A, shape (nodes, sources)
    target = (1.0 + reserve_margin) * grid.demand_kw  # (1 + r) d
    gram = share.T @ share                          # M = A^T A

    linear = grid.cost_per_kwh * grid.capacity_kw + lam * (np.diag(gram) - 2.0 * share.T @ target)
    quadratic = 2.0 * lam * np.triu(gram, k=1)
    constant = lam * float(target @ target)
    return QUBO(linear=linear, quadratic=quadratic, constant=constant)


def objective_direct(
    grid: Microgrid,
    x: np.ndarray,
    voll_per_kwh: float = DEFAULT_VOLL,
    reserve_margin: float = DEFAULT_RESERVE,
) -> float:
    """The objective written out term by term; used to test ``build_qubo``."""
    lam = default_balance_weight(grid, voll_per_kwh)
    x = np.asarray(x, dtype=float)
    supply = grid.share_matrix() @ x
    cost = float(np.sum(grid.cost_per_kwh * grid.capacity_kw * x))
    return cost + lam * float(np.sum((supply - (1.0 + reserve_margin) * grid.demand_kw) ** 2))


def to_ising(qubo: QUBO, tol: float = 1e-12) -> tuple[SparsePauliOp, float]:
    """Map x_j = (1 - Z_j) / 2. Returns (H, offset) with E(x) = <x|H|x> + offset."""
    n = qubo.n
    h, J = qubo.linear, qubo.quadratic
    offset = qubo.constant + h.sum() / 2.0 + J.sum() / 4.0
    z_coeff = -h / 2.0 - (J.sum(axis=1) + J.sum(axis=0)) / 4.0

    terms: list[tuple[str, list[int], float]] = []
    for j in range(n):
        if abs(z_coeff[j]) > tol:
            terms.append(("Z", [j], float(z_coeff[j])))
    for j in range(n):
        for k in range(j + 1, n):
            if abs(J[j, k]) > tol:
                terms.append(("ZZ", [j, k], float(J[j, k] / 4.0)))
    if not terms:
        terms.append(("", [], 0.0))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify(atol=tol), float(offset)


def all_bitstrings(n: int) -> np.ndarray:
    """(2^n, n) matrix; row k is the integer k in little-endian bits (column j = qubit j)."""
    k = np.arange(2**n, dtype=np.int64)[:, None]
    return ((k >> np.arange(n)) & 1).astype(np.int8)


def bitstring_to_array(bitstring: str) -> np.ndarray:
    """Qiskit bitstring (right-most char = qubit 0) -> x array indexed by qubit."""
    return np.array([int(c) for c in reversed(bitstring)], dtype=np.int8)


def array_to_bitstring(x: np.ndarray) -> str:
    return "".join(str(int(b)) for b in reversed(np.asarray(x)))
