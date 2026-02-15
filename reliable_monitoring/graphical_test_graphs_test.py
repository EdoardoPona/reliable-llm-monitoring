"""Tests for graphical_test_graphs.py — graph construction invariants."""

import numpy as np
import pytest

from reliable_monitoring.graphical_test_graphs import (
    chain_graph,
    lattice_graph,
    uniform_lattice_graph,
)

# ---------------------------------------------------------------------------
# Invariants that ALL graph factories must satisfy
# ---------------------------------------------------------------------------

LATTICE_FACTORIES = [lattice_graph, uniform_lattice_graph]
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
