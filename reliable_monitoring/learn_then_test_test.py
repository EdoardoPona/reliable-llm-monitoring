"""Tests for learn_then_test.py — graphical testing and FST equivalence."""

import numpy as np
import pytest

from reliable_monitoring.bounds import binomial, hb_p_value
from reliable_monitoring.graphical_test_graphs import chain_graph, lattice_graph
from reliable_monitoring.learn_then_test import (
    Hypothesis,
    compute_p_values,
    fixed_sequence_testing,
    graphical_testing,
    joint_p_value,
)
from reliable_monitoring.risks import Risk

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


# ---------------------------------------------------------------------------
# joint_p_value (union-bound multi-risk p-value)
# ---------------------------------------------------------------------------


def _make_risk(name: str, bound_fn) -> Risk:
    """Build a Risk object whose empirical computation is unused in these tests."""
    return Risk(
        name=name,
        description=f"test risk {name}",
        empirical_computation=lambda ctx: 0.0,
        p_value_bound_fn=bound_fn,
    )


class TestJointPValue:
    @pytest.fixture
    def two_binomial_risks(self):
        return [_make_risk("budget", binomial), _make_risk("safety", binomial)]

    def test_array_input_takes_pointwise_max(self, two_binomial_risks):
        """Joint p-value is the per-threshold max over each risk's p-value."""
        risks = two_binomial_risks
        n = 1000
        empirical = {
            "budget": np.array([0.10, 0.20, 0.30]),
            "safety": np.array([0.05, 0.40, 0.10]),
        }
        alphas = {"budget": 0.3, "safety": 0.3}

        joint = joint_p_value(empirical, risks, alphas, n)
        per_budget = binomial(empirical["budget"], n, alphas["budget"])
        per_safety = binomial(empirical["safety"], n, alphas["safety"])
        expected = np.maximum(per_budget, per_safety)

        np.testing.assert_array_equal(joint, expected)
        assert joint.shape == (3,)

    def test_scalar_input_returns_length_one_array(self, two_binomial_risks):
        risks = two_binomial_risks
        n = 200
        joint = joint_p_value(
            empirical_risks={"budget": 0.20, "safety": 0.05},
            risks=risks,
            alphas={"budget": 0.3, "safety": 0.3},
            n=n,
        )
        assert joint.shape == (1,)
        expected = float(np.maximum(binomial(0.20, n, 0.3)[0], binomial(0.05, n, 0.3)[0]))
        assert float(joint[0]) == pytest.approx(expected)

    def test_dominant_risk_drives_joint(self, two_binomial_risks):
        """When one risk's p-value is uniformly larger, joint == that risk's p-value."""
        risks = two_binomial_risks
        n = 500
        # safety always closer to alpha => larger p-values than budget
        empirical = {
            "budget": np.array([0.05, 0.10, 0.15]),  # well below alpha=0.3
            "safety": np.array([0.28, 0.29, 0.30]),  # near alpha=0.3 => large p
        }
        alphas = {"budget": 0.3, "safety": 0.3}
        joint = joint_p_value(empirical, risks, alphas, n)
        safety_only = binomial(empirical["safety"], n, alphas["safety"])
        np.testing.assert_array_equal(joint, safety_only)

    def test_single_risk_matches_underlying_bound(self):
        """One-risk joint p-value reduces to that risk's bound function."""
        risk = _make_risk("budget", binomial)
        n = 1000
        empirical = {"budget": np.array([0.1, 0.2, 0.3])}
        alphas = {"budget": 0.3}
        joint = joint_p_value(empirical, [risk], alphas, n)
        np.testing.assert_array_equal(joint, binomial(empirical["budget"], n, 0.3))

    def test_mixed_bound_functions(self):
        """Mixed binomial / HB bound functions both feed into the max."""
        budget = _make_risk("budget", binomial)
        accuracy = _make_risk("accuracy", hb_p_value)
        n = 800
        empirical = {
            "budget": np.array([0.20, 0.21]),
            "accuracy": np.array([0.10, 0.40]),
        }
        alphas = {"budget": 0.3, "accuracy": 0.3}
        joint = joint_p_value(empirical, [budget, accuracy], alphas, n)
        b = binomial(empirical["budget"], n, 0.3)
        a = hb_p_value(empirical["accuracy"], n, 0.3)
        np.testing.assert_array_equal(joint, np.maximum(b, a))

    def test_empty_risks_raises(self):
        with pytest.raises(ValueError, match="empty"):
            joint_p_value({}, [], {}, n=100)

    def test_missing_alpha_raises(self, two_binomial_risks):
        with pytest.raises(KeyError):
            joint_p_value(
                empirical_risks={"budget": 0.2, "safety": 0.1},
                risks=two_binomial_risks,
                alphas={"budget": 0.3},  # missing "safety"
                n=100,
            )
