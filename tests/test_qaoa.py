import numpy as np
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from microgrid import (
    ExactEvaluator,
    SamplerEvaluator,
    brute_force,
    build_ansatz,
    build_qubo,
    make_microgrid,
    prepare,
    probability_of_optimal,
    solve_layerwise,
)
from microgrid.metrics import cvar
from microgrid.qaoa import interp


def _problem(n=6, density=0.3, seed=0):
    return prepare(build_qubo(make_microgrid(n, density, seed)))


def test_estimator_agrees_with_distribution_energy():
    """<H> from the Qiskit Estimator primitive + offset == classical energy averaged over |amplitude|^2."""
    problem = _problem()
    ansatz = build_ansatz(problem.normalized_hamiltonian, reps=2)
    theta = np.array([0.4, 0.7, 0.3, 0.1])
    evs = StatevectorEstimator().run([(ansatz.circuit, problem.hamiltonian, ansatz.values(theta))]).result()[0].data.evs
    exact = ExactEvaluator(ansatz).probabilities(theta) @ problem.energies
    assert np.isclose(float(evs) + problem.offset, exact)


def test_sampler_evaluator_matches_exact_distribution():
    problem = _problem()
    ansatz = build_ansatz(problem.normalized_hamiltonian, reps=1, measure=True)
    theta = np.array([0.5, 0.3])
    sampled = SamplerEvaluator(StatevectorSampler(seed=1), ansatz, ansatz.circuit, shots=20000).probabilities(theta)
    exact = ExactEvaluator(build_ansatz(problem.normalized_hamiltonian, reps=1)).probabilities(theta)
    assert np.abs(sampled - exact).sum() < 0.08  # total variation distance from shot noise


def test_qaoa_beats_random_guessing_and_improves_with_depth():
    problem = _problem(n=8, density=0.3, seed=1)
    _, land = brute_force(problem.qubo)
    results = solve_layerwise(problem, max_reps=3, maxiter=150)
    p_random = len(land.optimal_indices) / 2**problem.n
    p_opt = [probability_of_optimal(r.probs, land.optimal_indices) for r in results]
    assert p_opt[0] > 2 * p_random
    assert results[-1].expectation < results[0].expectation
    assert results[-1].expectation < problem.energies.mean()


def test_cvar_limits():
    energies = np.array([3.0, 1.0, 2.0, 0.0])
    probs = np.full(4, 0.25)
    assert np.isclose(cvar(probs, energies, 1.0), energies.mean())
    assert np.isclose(cvar(probs, energies, 0.25), 0.0)
    assert np.isclose(cvar(probs, energies, 0.5), 0.5)


def test_interp_grows_one_layer():
    theta = np.array([0.1, 0.2, 0.9, 0.8])
    grown = interp(theta)
    assert grown.shape == (6,)
    assert np.isclose(grown[0], theta[0]) and np.isclose(grown[3], theta[2])
