import numpy as np

from microgrid import brute_force, build_qubo, dispatch_report, greedy_merit_order, make_microgrid, simulated_annealing
from microgrid.qubo import all_bitstrings


def test_brute_force_finds_global_minimum():
    qubo = build_qubo(make_microgrid(8, 0.4, 0))
    exact, land = brute_force(qubo)
    assert np.isclose(exact.energy, qubo.energies(all_bitstrings(8)).min())
    assert np.isclose(qubo.energy(exact.x), exact.energy)
    assert land.approximation_ratio(exact.energy) == 1.0
    assert land.approximation_ratio(land.e_max) == 0.0


def test_simulated_annealing_reaches_optimum_on_small_instances():
    for seed in range(5):
        qubo = build_qubo(make_microgrid(10, 0.3, seed))
        exact, _ = brute_force(qubo)
        sa = simulated_annealing(qubo, seed=seed)
        assert np.isclose(sa.energy, exact.energy)


def test_greedy_is_valid_and_never_beats_exact():
    for seed in range(5):
        grid = make_microgrid(8, 0.4, seed)
        qubo = build_qubo(grid)
        exact, _ = brute_force(qubo)
        greedy = greedy_merit_order(grid, qubo)
        assert set(np.unique(greedy.x)) <= {0, 1}
        assert greedy.energy >= exact.energy - 1e-9


def test_dispatch_report_accounting():
    grid = make_microgrid(8, 0.4, 1)
    report = dispatch_report(grid, np.ones(8))
    assert np.isclose(report["served_kw"] + report["unmet_kw"], report["demand_kw"])
    assert 0.0 <= report["pct_demand_met"] <= 100.0 + 1e-9
    assert dispatch_report(grid, np.zeros(8))["pct_demand_met"] == 0.0
