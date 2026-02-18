"""Tests for learn_then_test.py — graphical testing and FST equivalence."""

import numpy as np
import pytest

from reliable_monitoring.graphical_test_graphs import chain_graph, lattice_graph
from reliable_monitoring.learn_then_test import (
    Hypothesis,
    compute_p_values,
    fixed_sequence_testing,
    graphical_testing,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def easy_then_hard_pvalues():
    """First 3 hypotheses easy, last 2 hard."""
    return np.array([0.001, 0.002, 0.003, 0.8, 0.9])


# ---------------------------------------------------------------------------
# FST equivalence: graphical_testing + chain_graph == fixed_sequence_testing
# ---------------------------------------------------------------------------


class TestFSTEquivalence:
    """graphical_testing with a chain graph must match fixed_sequence_testing."""

    def test_partial_rejection(self, easy_then_hard_pvalues):
        pv = easy_then_hard_pvalues
        w, g = chain_graph(len(pv))
        gt = graphical_testing(pv, w, g, delta=0.1)
        fst = fixed_sequence_testing(pv, delta=0.1)
        assert gt.rejected == fst

    def test_all_rejected(self):
        pv = np.array([0.001, 0.002, 0.003])
        w, g = chain_graph(len(pv))
        gt = graphical_testing(pv, w, g, delta=0.1)
        fst = fixed_sequence_testing(pv, delta=0.1)
        assert gt.rejected == fst == [0, 1, 2]

    def test_none_rejected(self):
        pv = np.array([0.5, 0.6, 0.7])
        w, g = chain_graph(len(pv))
        gt = graphical_testing(pv, w, g, delta=0.1)
        fst = fixed_sequence_testing(pv, delta=0.1)
        assert gt.rejected == fst == []

    def test_single_hypothesis(self):
        for p in [0.01, 0.5]:
            pv = np.array([p])
            w, g = chain_graph(1)
            gt = graphical_testing(pv, w, g, delta=0.1)
            fst = fixed_sequence_testing(pv, delta=0.1)
            assert gt.rejected == fst


# ---------------------------------------------------------------------------
# Weight redistribution
# ---------------------------------------------------------------------------


class TestWeightRedistribution:
    def test_weights_do_not_exceed_budget(self, easy_then_hard_pvalues):
        """Active weights should never exceed the initial budget (1.0)."""
        pv = easy_then_hard_pvalues
        w, g = chain_graph(len(pv))
        result = graphical_testing(pv, w, g, delta=0.1)
        assert result.final_weights.sum() <= 1.0 + 1e-12

    def test_rejected_weights_are_zero(self, easy_then_hard_pvalues):
        pv = easy_then_hard_pvalues
        w, g = chain_graph(len(pv))
        result = graphical_testing(pv, w, g, delta=0.1)
        for idx in result.rejected:
            assert result.final_weights[idx] == 0.0

    def test_lattice_weight_preserved(self):
        """On a lattice, the weight budget should be preserved after rejections."""
        pv = np.full(6, 0.0001)  # all easy -> all rejected
        w, g = lattice_graph(2, 3)
        result = graphical_testing(pv, w, g, delta=0.1)
        assert len(result.rejected) == 6
        assert np.isclose(result.final_weights.sum(), 0.0)


# ---------------------------------------------------------------------------
# Bretz et al. example (4 hypotheses)
# ---------------------------------------------------------------------------


class TestBretzExample:
    """Reproduce the 4-hypothesis example from Bretz et al. 2009.

    Setup:
      H1, H2 primary (w1=w2=0.5); H3, H4 secondary (w3=w4=0).
      Transitions: H1->H3, H3->H2, H2->H4, H4->H1.
    """

    @pytest.fixture
    def bretz_graph(self):
        w = np.array([0.5, 0.5, 0.0, 0.0])
        g = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=float,
        )
        return w, g

    def test_both_primary_rejected(self, bretz_graph):
        """p1=0.01, p2=0.005, p3=0.1, p4=0.5 with alpha=0.025."""
        w, g = bretz_graph
        pv = np.array([0.01, 0.005, 0.1, 0.5])
        result = graphical_testing(pv, w, g, delta=0.025)
        # Both primaries should be rejected (0.01 < 0.5*0.025=0.0125,
        # 0.005 < 0.5*0.025=0.0125).
        assert 0 in result.rejected
        assert 1 in result.rejected

    def test_secondary_not_rejected_with_large_p(self, bretz_graph):
        w, g = bretz_graph
        pv = np.array([0.01, 0.005, 0.1, 0.5])
        result = graphical_testing(pv, w, g, delta=0.025)
        # After both primaries rejected, H3 gets weight from H1 (via
        # transition propagation).  But p3=0.1 is still too large.
        assert 2 not in result.rejected
        assert 3 not in result.rejected


# ---------------------------------------------------------------------------
# Lattice testing
# ---------------------------------------------------------------------------


class TestLatticeGraphicalTesting:
    def test_easy_lattice_rejects_all(self):
        """Tiny p-values everywhere -> everything rejected."""
        pv = np.full(12, 0.0001)
        w, g = lattice_graph(3, 4)
        result = graphical_testing(pv, w, g, delta=0.1)
        assert len(result.rejected) == 12

    def test_hard_lattice_rejects_none(self):
        pv = np.full(12, 0.99)
        w, g = lattice_graph(3, 4)
        result = graphical_testing(pv, w, g, delta=0.1)
        assert result.rejected == []

    def test_monotone_rejects_origin(self):
        """P-values increasing away from origin -> at least (0,0) rejected."""
        n_r, n_c = 3, 3
        pv = np.zeros(n_r * n_c)
        for r in range(n_r):
            for c in range(n_c):
                pv[r * n_c + c] = 0.001 * (1 + r + c)  # increases with r+c
        w, g = lattice_graph(n_r, n_c)
        result = graphical_testing(pv, w, g, delta=0.1)
        assert 0 in result.rejected  # origin

    def test_rejection_respects_graph_structure(self):
        """Only reachable nodes (via weight flow) can be rejected."""
        # Make p-values so that only origin and its right neighbor are easy
        pv = np.array(
            [
                0.001,
                0.001,
                0.99,
                0.99,
                0.99,
                0.99,
            ]
        )
        w, g = lattice_graph(2, 3)
        result = graphical_testing(pv, w, g, delta=0.1)
        # Origin (0,0) should be rejected; its neighbors get weight
        assert 0 in result.rejected


# ---------------------------------------------------------------------------
# Hypothesis wrapper and compute_p_values
# ---------------------------------------------------------------------------


class TestHypothesis:
    def test_p_value_returns_float(self):
        h = Hypothesis(p_value_fn=lambda: 0.05, params={"alpha": 0.1})
        assert h.p_value() == 0.05
        assert isinstance(h.p_value(), float)

    def test_params_metadata(self):
        h = Hypothesis(p_value_fn=lambda: 0.5, params={"threshold": 0.7, "alpha": 0.3})
        assert h.params == {"threshold": 0.7, "alpha": 0.3}

    def test_default_empty_params(self):
        h = Hypothesis(p_value_fn=lambda: 0.1)
        assert h.params == {}

    def test_compute_p_values(self):
        hypotheses = [Hypothesis(p_value_fn=lambda v=v: v) for v in [0.01, 0.05, 0.9]]
        pv = compute_p_values(hypotheses)
        np.testing.assert_array_almost_equal(pv, [0.01, 0.05, 0.9])

    def test_compute_p_values_empty(self):
        pv = compute_p_values([])
        assert len(pv) == 0

    def test_hypothesis_with_graphical_testing(self):
        """Hypothesis objects integrate with graphical_testing via compute_p_values."""
        hypotheses = [
            Hypothesis(p_value_fn=lambda: 0.001, params={"i": 0}),
            Hypothesis(p_value_fn=lambda: 0.002, params={"i": 1}),
            Hypothesis(p_value_fn=lambda: 0.9, params={"i": 2}),
        ]
        pv = compute_p_values(hypotheses)
        w, g = chain_graph(len(hypotheses))
        result = graphical_testing(pv, w, g, delta=0.1)
        assert result.rejected == [0, 1]
