"""Tests for cascade selection strategies."""

import numpy as np
import pytest

from reliable_monitoring.cascade import (
    get_selection_strategy,
    select_examples_for_baseline,
    select_fixed_budget_amount,
    select_fixed_budget_rate,
    select_fixed_threshold,
)


@pytest.fixture
def scores():
    """Simple synthetic probe scores."""
    return np.array([0.1, 0.3, 0.5, 0.7, 0.9])


class TestFixedThreshold:
    """Test fixed_threshold selection strategy."""

    def test_returns_boolean_mask(self, scores):
        """Return boolean array of correct shape."""
        mask = select_fixed_threshold(scores, threshold=0.7)
        assert mask.dtype == bool
        assert mask.shape == scores.shape

    def test_selects_in_range(self):
        """Select examples with score < threshold AND score > 1-threshold."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        mask = select_fixed_threshold(scores, threshold=0.7)
        # Select if score < 0.7 AND score > 0.3: only 0.5 qualifies
        assert np.array_equal(scores[mask], [0.5])

    def test_raises_on_invalid_threshold(self, scores):
        """Raise error if threshold out of bounds [0.5, 1]."""
        with pytest.raises(ValueError, match="must be in"):
            select_fixed_threshold(scores, threshold=0.3)
        with pytest.raises(ValueError, match="must be in"):
            select_fixed_threshold(scores, threshold=1.1)

    def test_raises_on_missing_threshold(self):
        """Raise error if threshold not provided via kwargs."""
        scores = np.array([0.5])
        with pytest.raises(TypeError):
            select_fixed_threshold(scores)


class TestFixedBudgetRate:
    """Test fixed_budget_rate selection strategy."""

    def test_selects_middle_portion(self, scores):
        """Select middle examples based on percentile rate."""
        mask = select_fixed_budget_rate(scores, rate=1.0)
        # rate=1.0 should select all (0th to 100th percentile)
        assert mask.sum() == 5

    def test_selects_correct_count(self):
        """Select approximately correct fraction."""
        scores = np.arange(100) / 100.0  # 0.0 to 0.99
        mask = select_fixed_budget_rate(scores, rate=0.5)
        # Should select ~50% (25th to 75th percentile)
        assert 40 <= mask.sum() <= 60

    def test_raises_on_missing_rate(self):
        """Raise error if rate not provided via kwargs."""
        scores = np.array([0.5])
        with pytest.raises(TypeError):
            select_fixed_budget_rate(scores)

    def test_ranking_scores_selects_topk(self):
        """With ranking_scores, select top fraction by that signal."""
        scores = np.arange(100) / 100.0
        ranking = np.random.default_rng(42).random(100)
        mask = select_fixed_budget_rate(scores, rate=0.2, ranking_scores=ranking)
        assert mask.sum() == 20
        # Selected examples should have the highest ranking scores
        assert ranking[mask].min() >= np.sort(ranking)[-20]


class TestFixedBudgetAmount:
    """Test fixed_budget_amount selection strategy."""

    def test_selects_exact_amount(self):
        """Select specified number of examples."""
        scores = np.arange(100) / 100.0
        mask = select_fixed_budget_amount(scores, amount=30)
        assert mask.sum() == 30

    def test_centers_on_median(self):
        """Examples centered around median score."""
        scores = np.arange(100) / 100.0
        mask = select_fixed_budget_amount(scores, amount=20)
        selected_scores = scores[mask]
        # Should be centered around 0.5
        assert 0.4 < selected_scores.mean() < 0.6

    def test_clamps_to_bounds(self):
        """Handle amount larger than dataset."""
        scores = np.array([0.1, 0.5, 0.9])
        mask = select_fixed_budget_amount(scores, amount=1000)
        assert mask.sum() == 3  # All selected, not more

    def test_raises_on_missing_amount(self):
        """Raise error if amount not provided via kwargs."""
        scores = np.array([0.5])
        with pytest.raises(TypeError):
            select_fixed_budget_amount(scores)

    def test_ranking_scores_selects_topk(self):
        """With ranking_scores, select top-k by that signal instead of median-centered."""
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.5])
        ranking = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
        mask = select_fixed_budget_amount(scores, amount=2, ranking_scores=ranking)
        assert mask.sum() == 2
        # Should select indices 2 and 4 (highest ranking scores)
        assert np.array_equal(np.where(mask)[0], [2, 4])

    def test_ranking_scores_zero_amount(self):
        """Amount <= 0 returns empty mask even with ranking_scores."""
        scores = np.array([0.1, 0.5, 0.9])
        ranking = np.array([1.0, 0.0, 0.5])
        mask = select_fixed_budget_amount(scores, amount=0, ranking_scores=ranking)
        assert mask.sum() == 0


class TestSelectionRegistry:
    """Test selection strategy registry and helper."""

    def test_get_builtin_strategies(self):
        """Can retrieve built-in strategies by name."""
        for name in ["fixed_threshold", "fixed_budget_rate", "fixed_budget_amount"]:
            strategy = get_selection_strategy(name)
            assert callable(strategy)

    def test_unknown_strategy_raises_error(self, scores):
        """Raise error for unknown strategy name."""
        with pytest.raises(ValueError, match="Unknown selection strategy"):
            select_examples_for_baseline(scores, strategy="nonexistent")

    def test_helper_with_string_strategy(self, scores):
        """Helper function works with strategy names."""
        mask = select_examples_for_baseline(scores, strategy="fixed_threshold", threshold=0.7)
        assert mask.sum() >= 0

    def test_helper_with_custom_callable(self, scores):
        """Helper function accepts custom callables."""

        def custom(probe_scores: np.ndarray, **kwargs) -> np.ndarray:
            return probe_scores > 0.5

        mask = select_examples_for_baseline(scores, strategy=custom)
        assert (scores[mask] > 0.5).all()

    def test_invalid_strategy_type_raises_error(self, scores):
        """Raise error if strategy is not string or callable."""
        with pytest.raises(TypeError, match="string or callable"):
            select_examples_for_baseline(scores, strategy=123)  # type: ignore
