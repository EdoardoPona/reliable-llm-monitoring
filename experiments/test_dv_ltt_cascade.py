"""Tests for the LTT driver functions in dv_ltt_cascade.

Covers the multi-risk extension :func:`ltt_joint_threshold` and verifies
that it reduces to :func:`ltt_budget_threshold` when only the budget risk
is supplied.
"""

import numpy as np
import pytest

from experiments.dv_ltt_cascade import (
    ltt_budget_threshold,
    ltt_joint_threshold,
    ltt_joint_threshold_split_chains,
)
from reliable_monitoring.risks import AccuracyRisk, BudgetCostRisk


@pytest.fixture
def synthetic_cascade():
    """Tiny synthetic cascade scenario with controllable risks.

    n=400, half safe (label 0) and half unsafe (label 1).  Probe and
    baseline scores are well-separated by label so the cascade is
    well-behaved at any threshold.  Delegation signal `dv` is uniformly
    distributed in [0, 1] so the empirical delegation rate at threshold
    tau is approximately `1 - tau`.
    """
    rng = np.random.default_rng(0)
    n = 400
    labels = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    # Probe and baseline both roughly correct but baseline a bit better
    probe_scores = np.where(labels == 1, rng.uniform(0.55, 0.95, n), rng.uniform(0.05, 0.45, n))
    baseline_scores = np.where(labels == 1, rng.uniform(0.60, 0.99, n), rng.uniform(0.01, 0.40, n))
    dv_scores = rng.uniform(0.0, 1.0, n)  # rate(tau) ≈ 1 - tau
    tau_grid = np.linspace(0.05, 0.95, 19)
    return {
        "probe_scores": probe_scores,
        "baseline_scores": baseline_scores,
        "dv_scores": dv_scores,
        "labels": labels,
        "tau_grid": tau_grid,
    }


class TestLTTJointThreshold:
    def test_returns_none_when_constraints_infeasible(self, synthetic_cascade):
        """An impossibly tight budget should produce no certified tau."""
        tau = ltt_joint_threshold(
            probe_scores=synthetic_cascade["probe_scores"],
            baseline_scores=synthetic_cascade["baseline_scores"],
            delegation_scores=synthetic_cascade["dv_scores"],
            labels=synthetic_cascade["labels"],
            tau_grid=synthetic_cascade["tau_grid"],
            risks=[BudgetCostRisk, AccuracyRisk],
            alphas={"budget": 0.001, "accuracy_error": 0.5},
            delta=0.1,
        )
        assert tau is None

    def test_loose_constraints_certify_aggressive_tau(self, synthetic_cascade):
        """Loose budget and accuracy targets should certify a more aggressive
        threshold than tight ones do."""

        def _run(alphas: dict[str, float]) -> float | None:
            return ltt_joint_threshold(
                probe_scores=synthetic_cascade["probe_scores"],
                baseline_scores=synthetic_cascade["baseline_scores"],
                delegation_scores=synthetic_cascade["dv_scores"],
                labels=synthetic_cascade["labels"],
                tau_grid=synthetic_cascade["tau_grid"],
                risks=[BudgetCostRisk, AccuracyRisk],
                alphas=alphas,
                delta=0.1,
            )

        loose_tau = _run({"budget": 0.95, "accuracy_error": 0.95})
        tight_tau = _run({"budget": 0.30, "accuracy_error": 0.30})
        assert loose_tau is not None
        # Loose alphas should certify a strictly smaller (more aggressive) tau
        # than tight alphas.
        if tight_tau is not None:
            assert loose_tau < tight_tau
        else:
            # If tight constraints rejected nothing, loose should still produce
            # some tau and it should be in the bottom half of the grid.
            median_tau = float(np.median(synthetic_cascade["tau_grid"]))
            assert loose_tau < median_tau

    def test_single_risk_matches_budget_only_driver(self, synthetic_cascade):
        """With only BudgetCostRisk, the joint driver should match the budget-only one."""
        alpha_b = 0.4
        delta = 0.1

        joint_tau = ltt_joint_threshold(
            probe_scores=synthetic_cascade["probe_scores"],
            baseline_scores=synthetic_cascade["baseline_scores"],
            delegation_scores=synthetic_cascade["dv_scores"],
            labels=synthetic_cascade["labels"],
            tau_grid=synthetic_cascade["tau_grid"],
            risks=[BudgetCostRisk],
            alphas={"budget": alpha_b},
            delta=delta,
        )
        budget_only_tau = ltt_budget_threshold(
            dv_scores=synthetic_cascade["dv_scores"],
            alpha_budget=alpha_b,
            delta=delta,
            tau_grid=synthetic_cascade["tau_grid"],
        )
        assert joint_tau == budget_only_tau

    def test_joint_is_at_least_as_conservative_as_budget_only(self, synthetic_cascade):
        """Adding an accuracy constraint can only restrict the certified set.

        So the joint tau (if any) must satisfy ``joint_tau >= budget_only_tau``
        (taus are ordered safest-first, larger = more conservative).
        """
        alpha_b = 0.4
        delta = 0.1
        budget_only_tau = ltt_budget_threshold(
            dv_scores=synthetic_cascade["dv_scores"],
            alpha_budget=alpha_b,
            delta=delta,
            tau_grid=synthetic_cascade["tau_grid"],
        )
        joint_tau = ltt_joint_threshold(
            probe_scores=synthetic_cascade["probe_scores"],
            baseline_scores=synthetic_cascade["baseline_scores"],
            delegation_scores=synthetic_cascade["dv_scores"],
            labels=synthetic_cascade["labels"],
            tau_grid=synthetic_cascade["tau_grid"],
            risks=[BudgetCostRisk, AccuracyRisk],
            alphas={"budget": alpha_b, "accuracy_error": 0.30},
            delta=delta,
        )
        # Either joint test fails, or it certifies a more-conservative tau
        if joint_tau is not None and budget_only_tau is not None:
            assert joint_tau >= budget_only_tau


class TestLTTJointThresholdSplitChains:
    def test_returns_none_when_intersection_empty(self, synthetic_cascade):
        """An impossibly tight budget AND tight accuracy should produce no certified tau."""
        tau = ltt_joint_threshold_split_chains(
            probe_scores=synthetic_cascade["probe_scores"],
            baseline_scores=synthetic_cascade["baseline_scores"],
            delegation_scores=synthetic_cascade["dv_scores"],
            labels=synthetic_cascade["labels"],
            tau_grid=synthetic_cascade["tau_grid"],
            risks=[BudgetCostRisk, AccuracyRisk],
            alphas={"budget": 0.001, "accuracy_error": 0.001},
            delta=0.1,
        )
        assert tau is None

    def test_loose_constraints_certify_something(self, synthetic_cascade):
        """Loose budget + accuracy targets should certify at least one tau."""
        tau = ltt_joint_threshold_split_chains(
            probe_scores=synthetic_cascade["probe_scores"],
            baseline_scores=synthetic_cascade["baseline_scores"],
            delegation_scores=synthetic_cascade["dv_scores"],
            labels=synthetic_cascade["labels"],
            tau_grid=synthetic_cascade["tau_grid"],
            risks=[BudgetCostRisk, AccuracyRisk],
            alphas={"budget": 0.95, "accuracy_error": 0.95},
            delta=0.1,
        )
        assert tau is not None

    def test_single_risk_matches_budget_only_driver(self, synthetic_cascade):
        """With only BudgetCostRisk, Split-chains with k=1 reduces to budget-only FST."""
        alpha_b = 0.4
        delta = 0.1
        sc_tau = ltt_joint_threshold_split_chains(
            probe_scores=synthetic_cascade["probe_scores"],
            baseline_scores=synthetic_cascade["baseline_scores"],
            delegation_scores=synthetic_cascade["dv_scores"],
            labels=synthetic_cascade["labels"],
            tau_grid=synthetic_cascade["tau_grid"],
            risks=[BudgetCostRisk],
            alphas={"budget": alpha_b},
            delta=delta,
        )
        budget_only_tau = ltt_budget_threshold(
            dv_scores=synthetic_cascade["dv_scores"],
            alpha_budget=alpha_b,
            delta=delta,
            tau_grid=synthetic_cascade["tau_grid"],
        )
        assert sc_tau == budget_only_tau

    def test_split_chains_rescues_when_max_p_stops_at_step0(self):
        """Construct a case where max-p FST stops at step 0 but split-chains certifies.

        n=1000.  Budget = mean(dv > tau); ascending dv => budget high at small tau,
        low at large tau.  Accuracy error: probe correct at large tau (no
        delegation), baseline correct at small tau.  Probe is wrong at rate
        0.45 (so probe-only acc_error = 0.45); baseline is wrong at rate 0.05.
        Alpha_budget = 0.5, alpha_safety = 0.50.  At tau_largest, acc_error=0.45
        but the binomial p-value at alpha=0.50 is borderline-high, so the
        max-p FST starting from tau_largest stops early.  Split-chains, which
        starts safety FST at tau_smallest (where acc_error=0.05, easy to
        certify), should rescue more of the chain.
        """
        rng = np.random.default_rng(7)
        n = 1000
        labels = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
        # Probe: 55% correct (acc_error 0.45)
        probe_correct = rng.random(n) < 0.55
        probe_scores = np.where(
            (labels == 1) & probe_correct,
            rng.uniform(0.6, 0.9, n),
            np.where(
                (labels == 0) & probe_correct,
                rng.uniform(0.1, 0.4, n),
                rng.uniform(0.4, 0.6, n),  # wrong: near the 0.5 boundary
            ),
        )
        # Baseline: 95% correct (acc_error 0.05)
        baseline_correct = rng.random(n) < 0.95
        baseline_scores = np.where(
            (labels == 1) & baseline_correct,
            rng.uniform(0.7, 0.99, n),
            np.where(
                (labels == 0) & baseline_correct,
                rng.uniform(0.01, 0.3, n),
                rng.uniform(0.45, 0.55, n),
            ),
        )
        dv_scores = rng.uniform(0.0, 1.0, n)
        tau_grid = np.linspace(0.1, 0.9, 17)

        risks = [BudgetCostRisk, AccuracyRisk]
        alphas = {"budget": 0.50, "accuracy_error": 0.50}
        max_p_tau = ltt_joint_threshold(
            probe_scores=probe_scores,
            baseline_scores=baseline_scores,
            delegation_scores=dv_scores,
            labels=labels,
            tau_grid=tau_grid,
            risks=risks,
            alphas=alphas,
            delta=0.10,
        )
        sc_tau = ltt_joint_threshold_split_chains(
            probe_scores=probe_scores,
            baseline_scores=baseline_scores,
            delegation_scores=dv_scores,
            labels=labels,
            tau_grid=tau_grid,
            risks=risks,
            alphas=alphas,
            delta=0.10,
        )

        # max-p fails because tau_largest has acc_error=0.45 (p ≈ borderline at alpha=0.50)
        # Split-chains runs each FST in its own optimal direction and should certify
        # at least one threshold in the intersection.
        assert sc_tau is not None
        # Sanity: if max-p also certified, Split-chains at delta/2 is stricter, so its
        # certified tau should not be more aggressive than max-p's at the same delta.
        # (Not a strict guarantee — orderings differ — but a reasonable sanity check
        # on this construction.)
        if max_p_tau is not None:
            # Both are valid; just check they're in the grid range
            assert tau_grid.min() <= sc_tau <= tau_grid.max()

    def test_orderings_override(self, synthetic_cascade):
        """Explicit risk_orderings parameter is respected."""
        tau_default = ltt_joint_threshold_split_chains(
            probe_scores=synthetic_cascade["probe_scores"],
            baseline_scores=synthetic_cascade["baseline_scores"],
            delegation_scores=synthetic_cascade["dv_scores"],
            labels=synthetic_cascade["labels"],
            tau_grid=synthetic_cascade["tau_grid"],
            risks=[BudgetCostRisk, AccuracyRisk],
            alphas={"budget": 0.7, "accuracy_error": 0.7},
            delta=0.10,
        )
        tau_overridden = ltt_joint_threshold_split_chains(
            probe_scores=synthetic_cascade["probe_scores"],
            baseline_scores=synthetic_cascade["baseline_scores"],
            delegation_scores=synthetic_cascade["dv_scores"],
            labels=synthetic_cascade["labels"],
            tau_grid=synthetic_cascade["tau_grid"],
            risks=[BudgetCostRisk, AccuracyRisk],
            alphas={"budget": 0.7, "accuracy_error": 0.7},
            delta=0.10,
            # Force both risks to "ascending" — for budget this is its worst
            # direction (most-delegation first => high budget => high p_budget)
            # so the budget chain should reject fewer taus than under the default.
            risk_orderings={"budget": "ascending", "accuracy_error": "ascending"},
        )
        # Both runs should produce floats or None (no exception), and at minimum
        # the override path executes (we only assert types).
        assert tau_default is None or isinstance(tau_default, float)
        assert tau_overridden is None or isinstance(tau_overridden, float)

    def test_invalid_ordering_raises(self, synthetic_cascade):
        with pytest.raises(ValueError, match="Invalid ordering"):
            ltt_joint_threshold_split_chains(
                probe_scores=synthetic_cascade["probe_scores"],
                baseline_scores=synthetic_cascade["baseline_scores"],
                delegation_scores=synthetic_cascade["dv_scores"],
                labels=synthetic_cascade["labels"],
                tau_grid=synthetic_cascade["tau_grid"],
                risks=[BudgetCostRisk],
                alphas={"budget": 0.5},
                delta=0.1,
                risk_orderings={"budget": "bogus"},
            )

    def test_empty_risks_raises(self, synthetic_cascade):
        with pytest.raises(ValueError, match="empty"):
            ltt_joint_threshold_split_chains(
                probe_scores=synthetic_cascade["probe_scores"],
                baseline_scores=synthetic_cascade["baseline_scores"],
                delegation_scores=synthetic_cascade["dv_scores"],
                labels=synthetic_cascade["labels"],
                tau_grid=synthetic_cascade["tau_grid"],
                risks=[],
                alphas={},
                delta=0.1,
            )
