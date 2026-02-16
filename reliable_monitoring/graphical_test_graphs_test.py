"""Tests for graphical_test_graphs.py — graph construction invariants."""

import numpy as np
import pytest

from reliable_monitoring.graphical_test_graphs import (
    chain_graph,
    lattice_graph,
    row_chain_graph,
    uniform_lattice_graph,
)

# ---------------------------------------------------------------------------
# Invariants that ALL graph factories must satisfy
# ---------------------------------------------------------------------------

LATTICE_FACTORIES = [lattice_graph, uniform_lattice_graph, row_chain_graph]
DIMS = [(2, 3), (3, 4), (5, 5), (1, 1), (1, 5), (4, 1)]


def _build(factory, dims):
    """Call factory with appropriate args."""
    if factory is chain_graph:
        return factory(dims[0] * dims[1])
    return factory(*dims)


@pytest.fixture(params=LATTICE_FACTORIES + [chain_graph], ids=lambda f: f.__name__)
def graph_fn(request):
    return request.param


@pytest.fixture(params=DIMS, ids=lambda d: f"{d[0]}x{d[1]}")
def dims(request):
    return request.param


class TestGraphInvariants:
    def test_weight_sum_at_most_one(self, graph_fn, dims):
        w, _ = _build(graph_fn, dims)
        assert w.sum() <= 1.0 + 1e-10

    def test_nonneg_weights(self, graph_fn, dims):
        w, _ = _build(graph_fn, dims)
        assert np.all(w >= -1e-10)

    def test_diagonal_zero(self, graph_fn, dims):
        _, g = _build(graph_fn, dims)
        assert np.allclose(np.diag(g), 0.0)

    def test_row_sums_at_most_one(self, graph_fn, dims):
        _, g = _build(graph_fn, dims)
        assert np.all(g.sum(axis=1) <= 1.0 + 1e-10)

    def test_nonneg_transitions(self, graph_fn, dims):
        _, g = _build(graph_fn, dims)
        assert np.all(g >= -1e-10)


# ---------------------------------------------------------------------------
# Chain-specific tests
# ---------------------------------------------------------------------------


class TestChainGraph:
    def test_all_weight_at_first_node(self):
        w, _ = chain_graph(5)
        assert w[0] == 1.0
        assert np.all(w[1:] == 0.0)

    def test_sequential_transitions(self):
        _, g = chain_graph(4)
        for i in range(3):
            assert g[i, i + 1] == 1.0
        assert g[3].sum() == 0.0

    def test_single_node(self):
        w, g = chain_graph(1)
        assert w[0] == 1.0
        assert g.shape == (1, 1)
        assert g[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Lattice-specific tests
# ---------------------------------------------------------------------------


class TestLatticeGraph:
    def test_weight_at_origin(self):
        w, _ = lattice_graph(3, 4)
        assert w[0] == 1.0
        assert np.isclose(w.sum(), 1.0)

    def test_interior_node_two_neighbors(self):
        """Interior node (1,1) in a 3x4 grid has edges down and right."""
        _, g = lattice_graph(3, 4)
        idx = 1 * 4 + 1  # (1,1)
        down = 2 * 4 + 1  # (2,1)
        right = 1 * 4 + 2  # (1,2)
        assert g[idx, down] == pytest.approx(0.5)
        assert g[idx, right] == pytest.approx(0.5)
        assert g[idx].sum() == pytest.approx(1.0)

    def test_boundary_node_one_neighbor(self):
        """Bottom-left (2,0) in 3x4: only right neighbor."""
        _, g = lattice_graph(3, 4)
        idx = 2 * 4 + 0  # (2,0)
        right = 2 * 4 + 1  # (2,1)
        assert g[idx, right] == pytest.approx(1.0)
        assert g[idx].sum() == pytest.approx(1.0)

    def test_terminal_node_no_neighbors(self):
        """Bottom-right corner (2,3) in 3x4: no outgoing edges."""
        _, g = lattice_graph(3, 4)
        idx = 2 * 4 + 3  # (2,3)
        assert g[idx].sum() == pytest.approx(0.0)

    def test_top_right_boundary(self):
        """Top-right (0,3) in 3x4: only down neighbor."""
        _, g = lattice_graph(3, 4)
        idx = 0 * 4 + 3  # (0,3)
        down = 1 * 4 + 3  # (1,3)
        assert g[idx, down] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Uniform lattice
# ---------------------------------------------------------------------------


class TestUniformLatticeGraph:
    def test_uniform_weights(self):
        w, _ = uniform_lattice_graph(3, 4)
        m = 3 * 4
        assert np.allclose(w, 1.0 / m)

    def test_same_transitions_as_lattice(self):
        _, g_lattice = lattice_graph(3, 4)
        _, g_uniform = uniform_lattice_graph(3, 4)
        assert np.allclose(g_lattice, g_uniform)


# ---------------------------------------------------------------------------
# Row-chain graph
# ---------------------------------------------------------------------------


class TestRowChainGraph:
    def test_equal_weight_per_row(self):
        n_r, n_c = 3, 4
        w, _ = row_chain_graph(n_r, n_c)
        for r in range(n_r):
            assert w[r * n_c] == pytest.approx(1.0 / n_r)
            assert np.all(w[r * n_c + 1 : (r + 1) * n_c] == 0.0)

    def test_chain_within_row(self):
        n_r, n_c = 2, 4
        _, g = row_chain_graph(n_r, n_c)
        for r in range(n_r):
            for c in range(n_c - 1):
                assert g[r * n_c + c, r * n_c + c + 1] == pytest.approx(1.0)

    def test_surplus_transfer_between_rows(self):
        n_r, n_c = 3, 4
        _, g = row_chain_graph(n_r, n_c)
        for r in range(n_r - 1):
            end = r * n_c + n_c - 1
            start_next = (r + 1) * n_c
            assert g[end, start_next] == pytest.approx(1.0)

    def test_last_row_no_surplus(self):
        n_r, n_c = 3, 4
        _, g = row_chain_graph(n_r, n_c)
        last_node = (n_r - 1) * n_c + n_c - 1
        assert g[last_node].sum() == pytest.approx(0.0)

    def test_no_cross_row_edges_except_surplus(self):
        """No edges between rows except the end→start surplus links."""
        n_r, n_c = 3, 4
        _, g = row_chain_graph(n_r, n_c)
        for r in range(n_r):
            for c in range(n_c):
                idx = r * n_c + c
                for r2 in range(n_r):
                    if r2 == r:
                        continue
                    for c2 in range(n_c):
                        idx2 = r2 * n_c + c2
                        # Only allowed cross-row edge: end of r → start of r+1
                        if c == n_c - 1 and r2 == r + 1 and c2 == 0:
                            continue
                        assert g[idx, idx2] == 0.0, f"unexpected edge ({r},{c})->({r2},{c2})"

    def test_single_column_is_chain(self):
        """With 1 column, row_chain_graph degenerates to chain_graph."""
        n_r = 5
        w_rc, g_rc = row_chain_graph(n_r, 1)
        w_ch, g_ch = chain_graph(n_r)
        # Weights differ (1/n_r vs 1 at node 0), but the surplus transfer
        # makes the effective behavior similar.  Check structure only.
        for r in range(n_r - 1):
            assert g_rc[r, r + 1] == pytest.approx(1.0)
        assert np.allclose(w_rc, 1.0 / n_r)
