import numpy as np
import pytest

from microgrid import build_qubo, make_microgrid, to_ising
from microgrid.qubo import all_bitstrings, array_to_bitstring, bitstring_to_array, objective_direct


@pytest.mark.parametrize("n,density,seed", [(6, 0.2, 0), (6, 0.6, 1), (8, 0.3, 2), (8, 0.8, 3)])
def test_qubo_matches_direct_objective(n, density, seed):
    grid = make_microgrid(n, density, seed)
    qubo = build_qubo(grid)
    bits = all_bitstrings(n)
    direct = np.array([objective_direct(grid, x) for x in bits])
    assert np.allclose(qubo.energies(bits), direct)


@pytest.mark.parametrize("n,density,seed", [(6, 0.3, 0), (8, 0.5, 1)])
def test_ising_diagonal_matches_qubo(n, density, seed):
    qubo = build_qubo(make_microgrid(n, density, seed))
    hamiltonian, offset = to_ising(qubo)
    diagonal = np.real(hamiltonian.to_matrix(sparse=True).diagonal()) + offset
    # Row k of all_bitstrings is basis state |k>, the same ordering as the matrix diagonal.
    assert np.allclose(diagonal, qubo.energies(all_bitstrings(n)))


def test_share_matrix_conserves_supply():
    grid = make_microgrid(10, 0.4, 5)
    assert np.allclose(grid.share_matrix().sum(axis=0), grid.capacity_kw)


def test_sparse_grid_gives_sparse_qubo():
    sparse = build_qubo(make_microgrid(12, 0.0, 0))
    dense = build_qubo(make_microgrid(12, 1.0, 0))
    assert sparse.num_couplings() < dense.num_couplings() == 12 * 11 // 2


def test_instances_are_reproducible_and_connected():
    a, b = make_microgrid(9, 0.3, 42), make_microgrid(9, 0.3, 42)
    assert a.edges == b.edges and np.array_equal(a.capacity_kw, b.capacity_kw)
    # Connected: a breadth-first search from node 0 reaches every node.
    adj, seen, frontier = a.adjacency(), {0}, [0]
    while frontier:
        i = frontier.pop()
        for j in np.flatnonzero(adj[i]):
            if j not in seen:
                seen.add(int(j))
                frontier.append(int(j))
    assert len(seen) == a.n


def test_bitstring_convention_matches_qiskit():
    x = np.array([1, 0, 0, 1, 1])  # qubits 0, 3, 4 set
    assert array_to_bitstring(x) == "11001"
    assert np.array_equal(bitstring_to_array("11001"), x)
    assert int("11001", 2) == sum(2**j for j in np.flatnonzero(x))
