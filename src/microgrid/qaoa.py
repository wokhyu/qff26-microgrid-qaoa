"""QAOA for the dispatch QUBO.

Design choices:

* The cost layer is exp(-i * gamma * H / s), with H the Ising Hamiltonian and s
  its largest |coefficient|. Normalising makes good gamma values comparable
  across instances, which is what makes parameter transfer work.
* The mixer is exp(+i * beta * sum_j X_j), i.e. evolution under -sum X, whose
  ground state is the |+>^n start state. With this sign a linear ramp (gamma
  rising, beta falling) is a Trotterised adiabatic path, a good starting point.
* The objective is evaluated classically from the output distribution: the
  mean energy, or the CVaR (mean of the best alpha fraction of samples). The
  same code path works for exact statevectors and for shot-based samplers,
  including real IBM hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from .metrics import cvar, distribution_from_counts, expectation
from .qubo import QUBO, all_bitstrings, to_ising


@dataclass(frozen=True)
class QAOAProblem:
    qubo: QUBO
    hamiltonian: SparsePauliOp
    offset: float
    scale: float
    energies: np.ndarray

    @property
    def n(self) -> int:
        return self.qubo.n

    @property
    def normalized_hamiltonian(self) -> SparsePauliOp:
        return (self.hamiltonian / self.scale).simplify()


def prepare(qubo: QUBO) -> QAOAProblem:
    hamiltonian, offset = to_ising(qubo)
    non_identity = [abs(c) for p, c in zip(hamiltonian.paulis, hamiltonian.coeffs) if p.z.any() or p.x.any()]
    scale = float(max(non_identity)) if non_identity else 1.0
    energies = qubo.energies(all_bitstrings(qubo.n))
    return QAOAProblem(qubo, hamiltonian, offset, scale, energies)


@dataclass(frozen=True)
class Ansatz:
    circuit: QuantumCircuit
    gammas: ParameterVector
    betas: ParameterVector

    @property
    def reps(self) -> int:
        return len(self.gammas)

    def values(self, theta: np.ndarray, circuit: QuantumCircuit | None = None) -> np.ndarray:
        """theta = [gamma_1..gamma_p, beta_1..beta_p] -> values in circuit.parameters order."""
        theta = np.asarray(theta, dtype=float)
        lookup = {g: theta[i] for i, g in enumerate(self.gammas)}
        lookup.update({b: theta[self.reps + i] for i, b in enumerate(self.betas)})
        circuit = self.circuit if circuit is None else circuit
        return np.array([lookup[p] for p in circuit.parameters])

    def bind(self, theta: np.ndarray, circuit: QuantumCircuit | None = None) -> QuantumCircuit:
        circuit = self.circuit if circuit is None else circuit
        return circuit.assign_parameters(self.values(theta, circuit))


def build_ansatz(cost: SparsePauliOp, reps: int, measure: bool = False) -> Ansatz:
    """Explicit QAOA circuit: H^n, then p x (cost layer, mixer layer)."""
    n = cost.num_qubits
    gammas = ParameterVector("gamma", reps)
    betas = ParameterVector("beta", reps)
    qc = QuantumCircuit(n, name=f"QAOA(p={reps})")
    qc.h(range(n))

    terms = []
    for pauli, coeff in zip(cost.paulis, cost.coeffs):
        support = [int(q) for q in np.flatnonzero(pauli.z)]
        if pauli.x.any():
            raise ValueError("cost Hamiltonian must be diagonal (Z terms only)")
        if support:
            terms.append((support, float(np.real(coeff))))

    for layer in range(reps):
        for support, coeff in terms:
            angle = 2.0 * coeff * gammas[layer]
            if len(support) == 1:
                qc.rz(angle, support[0])
            elif len(support) == 2:
                qc.rzz(angle, support[0], support[1])
            else:
                raise ValueError("only 1- and 2-local terms are supported")
        for q in range(n):
            qc.rx(-2.0 * betas[layer], q)
    if measure:
        qc.measure_all()
    return Ansatz(qc, gammas, betas)


# ---------------------------------------------------------------- evaluators


class Evaluator(Protocol):
    def probabilities(self, theta: np.ndarray) -> np.ndarray: ...


class ExactEvaluator:
    """Noise-free output distribution from the Qiskit statevector."""

    def __init__(self, ansatz: Ansatz):
        self.ansatz = ansatz

    def probabilities(self, theta: np.ndarray) -> np.ndarray:
        return Statevector(self.ansatz.bind(theta)).probabilities()


class SamplerEvaluator:
    """Shot-based distribution from any Qiskit SamplerV2.

    Works with ``qiskit.primitives.StatevectorSampler``, with
    ``qiskit_ibm_runtime.SamplerV2`` on a fake backend (local noisy simulation)
    and with ``qiskit_ibm_runtime.SamplerV2`` on real IBM hardware. The circuit
    must already be transpiled for the sampler's target (an ISA circuit) and
    must end with ``measure_all``.
    """

    def __init__(self, sampler, ansatz: Ansatz, isa_circuit: QuantumCircuit, shots: int = 4096,
                 on_job: Callable | None = None):
        self.sampler = sampler
        self.ansatz = ansatz
        self.isa_circuit = isa_circuit
        self.shots = shots
        self.on_job = on_job
        self.last_counts: dict[str, int] | None = None

    def counts(self, theta: np.ndarray) -> dict[str, int]:
        values = self.ansatz.values(theta, self.isa_circuit)
        job = self.sampler.run([(self.isa_circuit, values)], shots=self.shots)
        if self.on_job is not None:
            self.on_job(job)
        result = job.result()[0]
        self.last_counts = result.data.meas.get_counts()
        return self.last_counts

    def probabilities(self, theta: np.ndarray) -> np.ndarray:
        return distribution_from_counts(self.counts(theta), self.ansatz.circuit.num_qubits)


# -------------------------------------------------------------- optimisation


@dataclass
class QAOAResult:
    reps: int
    theta: np.ndarray
    objective: str
    objective_value: float
    expectation: float
    probs: np.ndarray
    nfev: int
    runtime_s: float
    history: list[float] = field(default_factory=list)


def linear_ramp(reps: int, dt: float = 0.8) -> np.ndarray:
    s = (np.arange(reps) + 0.5) / reps
    return np.concatenate([dt * s, dt * (1.0 - s)])


def interp(theta: np.ndarray) -> np.ndarray:
    """INTERP heuristic (Zhou et al., PRX 2020): p-layer optimum -> p+1 start."""
    p = len(theta) // 2
    out = []
    for half in (theta[:p], theta[p:]):
        padded = np.concatenate([[0.0], half, [0.0]])
        i = np.arange(1, p + 2)
        out.append((i - 1) / p * padded[i - 1] + (p - i + 1) / p * padded[i])
    return np.concatenate(out)


def make_objective(problem: QAOAProblem, kind: str = "expectation", alpha: float = 0.25):
    if kind == "expectation":
        return lambda probs: expectation(probs, problem.energies)
    if kind == "cvar":
        return lambda probs: cvar(probs, problem.energies, alpha)
    raise ValueError(f"unknown objective {kind!r}")


def optimize(
    problem: QAOAProblem,
    evaluator: Evaluator,
    reps: int,
    starts: list[np.ndarray],
    objective: str = "expectation",
    alpha: float = 0.25,
    maxiter: int = 300,
) -> QAOAResult:
    """COBYLA from the best of several candidate starting points."""
    t0 = time.perf_counter()
    score = make_objective(problem, objective, alpha)
    history: list[float] = []

    def f(theta: np.ndarray) -> float:
        value = score(evaluator.probabilities(theta))
        history.append(value)
        return value

    start_values = [f(np.asarray(s, dtype=float)) for s in starts]
    x0 = np.asarray(starts[int(np.argmin(start_values))], dtype=float)
    res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
    probs = evaluator.probabilities(res.x)
    return QAOAResult(
        reps=reps,
        theta=np.asarray(res.x),
        objective=objective,
        objective_value=score(probs),
        expectation=expectation(probs, problem.energies),
        probs=probs,
        nfev=len(history) + 1,
        runtime_s=time.perf_counter() - t0,
        history=history,
    )


def p1_grid(problem: QAOAProblem, evaluator: Evaluator, objective: str = "expectation",
            alpha: float = 0.25, size: int = 8) -> np.ndarray:
    """Coarse grid over (gamma, beta) for p = 1; returns the best grid point."""
    score = make_objective(problem, objective, alpha)
    best, best_theta = np.inf, None
    for g in np.linspace(-1.5, 1.5, 2 * size):
        for b in np.linspace(-0.75, 0.75, size):
            theta = np.array([g, b])
            value = score(evaluator.probabilities(theta))
            if value < best:
                best, best_theta = value, theta
    return best_theta


def solve_layerwise(
    problem: QAOAProblem,
    max_reps: int,
    evaluator_factory: Callable[[Ansatz], Evaluator] = ExactEvaluator,
    objective: str = "expectation",
    alpha: float = 0.25,
    maxiter: int = 300,
    transfer: dict[int, np.ndarray] | None = None,
    use_grid: bool = True,
) -> list[QAOAResult]:
    """Optimise p = 1..max_reps, warm-starting each p from INTERP of p-1.

    ``transfer`` maps p -> parameters optimised on another (usually smaller)
    instance; they are tried as an extra starting point.
    """
    results: list[QAOAResult] = []
    cost = problem.normalized_hamiltonian
    for reps in range(1, max_reps + 1):
        ansatz = build_ansatz(cost, reps)
        evaluator = evaluator_factory(ansatz)
        starts = [linear_ramp(reps)]
        if results:
            starts.append(interp(results[-1].theta))
        elif use_grid:
            starts.append(p1_grid(problem, evaluator, objective, alpha))
        if transfer and reps in transfer:
            starts.append(np.asarray(transfer[reps]))
        results.append(optimize(problem, evaluator, reps, starts, objective, alpha, maxiter))
    return results


def top_bitstrings(probs: np.ndarray, n: int, k: int = 10) -> list[tuple[str, float]]:
    order = np.argsort(probs)[::-1][:k]
    return [(format(int(i), f"0{n}b"), float(probs[i])) for i in order if probs[i] > 0]


def sample_best(problem: QAOAProblem, probs: np.ndarray, shots: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """Draw ``shots`` samples and return the lowest-energy bitstring seen.

    This is how QAOA is used in practice: the answer is the best sample, not
    the mean.
    """
    rng = np.random.default_rng(seed)
    p = np.clip(probs, 0.0, None)
    idx = rng.choice(len(p), size=shots, p=p / p.sum())
    best = int(idx[np.argmin(problem.energies[idx])])
    x = (best >> np.arange(problem.n)) & 1
    return x.astype(np.int8), float(problem.energies[best])
