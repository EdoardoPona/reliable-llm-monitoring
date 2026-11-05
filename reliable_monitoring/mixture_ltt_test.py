import numpy as np
import pytest

from reliable_monitoring.learn_then_test import build_risk_pareto_ltt
from reliable_monitoring.mixture_ltt import (
    MixtureLTTResult,
    build_mixture_pareto_ltt,
    evaluate_domain_risks,
    mixture_weights,
    simplex_grid,
)
from reliable_monitoring.risks import AccuracyRisk, BudgetCostRisk, RocAucRisk, ThresholdEvaluationResult


def scores(seed, n=300) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    probe = np.clip(0.15 + 0.65 * labels + rng.normal(0, 0.3, n), 0, 1)
    expert = np.clip(0.1 + 0.8 * labels + rng.normal(0, 0.2, n), 0, 1)
    return probe, expert, labels, rng.uniform(-0.5, 0.5, n)


@pytest.mark.parametrize("guaranteed,objective", [(AccuracyRisk, BudgetCostRisk), (BudgetCostRisk, AccuracyRisk)])
def test_one_domain_matches_existing_selector(guaranteed, objective):
    ht, opt = scores(1), scores(2)
    taus = np.linspace(-0.51, 0.51, 20)
    existing = build_risk_pareto_ltt(*ht, *opt, taus, guaranteed, objective)
    mixture = build_mixture_pareto_ltt(
        *ht,
        np.full(300, "only"),
        *opt,
        np.full(300, "only"),
        taus,
        guaranteed,
        objective,
        domains=["only"],
        reference_weights={"only": 1},
    )
    np.testing.assert_allclose(existing.taus, mixture.taus)
    np.testing.assert_allclose(existing.opt_risks, mixture.opt_risks)
    for alpha in [0.1, 0.2, 0.3, 0.8]:
        assert existing.select_threshold(alpha, 0.1) == mixture.select_threshold(alpha, 0.1)


def test_a_bad_small_domain_cannot_hide_behind_the_pool():
    # Zero budget use in 900 examples, 100% use in a separate 100-example domain.
    # The pooled rate can be certified below 20%; the second domain cannot.
    n = 1000
    groups = np.array(["large"] * 900 + ["small"] * 100)
    table = evaluate_domain_risks(
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.r_[np.zeros(900), np.ones(100)],
        groups,
        np.array([0.5]),
        domains=["large", "small"],
        risks=[BudgetCostRisk],
    )
    selector = MixtureLTTResult(np.array([0.5]), np.array([0.0]), table, BudgetCostRisk)
    assert float(BudgetCostRisk.p_value_bound_fn(0.1, n, 0.2)[0]) < 0.1
    assert selector.select_threshold(0.2, 0.1) is None
    per_group = selector.domain_p_values(0.2)
    assert per_group["large"][0] == pytest.approx(0.8**900)
    assert per_group["small"][0] == 1
    np.testing.assert_allclose(selector.p_values(0.2), np.maximum(per_group["large"], per_group["small"]))


def test_missing_hypothesis_testing_domain_cannot_certify():
    data = scores(3)
    table = evaluate_domain_risks(
        *data, np.full(300, "present"), np.array([1.0]), domains=["present", "absent"], risks=[BudgetCostRisk]
    )
    selector = MixtureLTTResult(np.array([1.0]), np.zeros(1), table, BudgetCostRisk)
    assert table["absent"].n_samples == 0
    assert selector.domain_p_values(0.5)["absent"][0] == 1
    assert selector.select_threshold(0.5, 0.1) is None


def test_candidate_order_does_not_read_testing_outcomes():
    ht, opt = scores(4), scores(5)
    groups = np.array(["a", "b"] * 150)
    kwargs: dict = dict(domains=["a", "b"], reference_weights={"a": 2, "b": 1})
    a = build_mixture_pareto_ltt(
        *ht, groups, *opt, groups, np.linspace(-0.5, 0.5, 20), AccuracyRisk, BudgetCostRisk, **kwargs
    )
    changed = (np.ones(300), np.zeros(300), 1 - ht[2], -ht[3])
    b = build_mixture_pareto_ltt(
        *changed, groups[::-1], *opt, groups, np.linspace(-0.5, 0.5, 20), AccuracyRisk, BudgetCostRisk, **kwargs
    )
    np.testing.assert_array_equal(a.taus, b.taus)
    np.testing.assert_array_equal(a.opt_risks, b.opt_risks)


def test_sequence_stops_at_first_failure(monkeypatch):
    table = {"a": ThresholdEvaluationResult(np.array([3.0, 2.0, 1.0]), 100, {"budget": np.zeros(3)}, [BudgetCostRisk])}
    selector = MixtureLTTResult(np.array([3.0, 2.0, 1.0]), np.array([0.9, 0.5, 0.1]), table, BudgetCostRisk)
    monkeypatch.setattr(selector, "p_values", lambda alpha: np.array([0.01, 0.5, 0.001]))
    assert selector.select_threshold(0.5, 0.1) == 3.0


def test_mixture_grid_has_vertices_and_linear_worst_case():
    grid = simplex_grid(3, 20)
    assert grid.shape == (231, 3)
    np.testing.assert_allclose(grid.sum(axis=1), 1)
    for vertex in np.eye(3):
        assert np.any(np.all(grid == vertex, axis=1))
    risks = np.array([0.1, 0.3, 0.2])
    assert np.max(grid @ risks) == pytest.approx(np.max(risks))
    np.testing.assert_allclose(
        mixture_weights({"a": 500, "b": 586, "c": 500}, ["a", "b", "c"]), np.array([500, 586, 500]) / 1586
    )


def test_rejects_auc_and_invalid_domain_mapping():
    data = scores(6)
    with pytest.raises(ValueError, match="accuracy_error and budget"):
        evaluate_domain_risks(*data, np.full(300, "a"), np.array([0.0]), domains=["a"], risks=[RocAucRisk])
    with pytest.raises(ValueError, match="missing from domains"):
        evaluate_domain_risks(*data, np.full(300, "a"), np.array([0.0]), domains=["b"], risks=[BudgetCostRisk])
    with pytest.raises(ValueError, match="nonnegative"):
        mixture_weights({"a": 1, "b": -1}, ["a", "b"])
