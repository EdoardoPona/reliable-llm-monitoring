"""Unit tests for activation reduction strategies."""

import pytest
import torch

from reliable_monitoring.reductions import (
    apply_reduction_batched,
    get_reduction_function,
    reduce_first,
    reduce_last,
    reduce_max,
    reduce_mean,
    register_reduction,
)


@pytest.fixture
def simple_sequence():
    """Small test sequence for basic tests."""
    activations = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # seq_len=3
            [[2.0, 3.0], [4.0, 5.0], [0.0, 0.0]],  # seq_len=2 (last masked)
        ]
    )
    attention_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return activations, attention_mask


@pytest.fixture
def masked_sequence():
    """Sequence with masked token to test mask handling."""
    activations = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]])
    attention_mask = torch.tensor([[1.0, 1.0, 0.0]])
    return activations, attention_mask


@pytest.fixture
def large_batch():
    """Large batch for batching tests."""
    n_samples, seq_len, hidden_dim = 100, 10, 32
    activations = torch.randn(n_samples, seq_len, hidden_dim)
    attention_mask = torch.ones(n_samples, seq_len)
    return activations, attention_mask


class TestReductionFunctions:
    """Test individual reduction functions."""

    def test_reduce_mean_respects_mask(self, simple_sequence):
        """Test that mean reduction respects attention mask."""
        activations, attention_mask = simple_sequence
        result = reduce_mean(activations, attention_mask)

        # First sample: mean of all 3 tokens
        expected_first = torch.tensor([3.0, 4.0])  # (1+3+5)/3, (2+4+6)/3
        assert torch.allclose(result[0], expected_first)

        # Second sample: mean of first 2 tokens only
        expected_second = torch.tensor([3.0, 4.0])  # (2+4)/2, (3+5)/2
        assert torch.allclose(result[1], expected_second)

    def test_reduce_max_respects_mask(self, masked_sequence):
        """Test that max reduction respects attention mask."""
        activations, attention_mask = masked_sequence
        result = reduce_max(activations, attention_mask)

        # Should be max of first two, not the masked 999
        expected = torch.tensor([3.0, 4.0])
        assert torch.allclose(result[0], expected)

    def test_reduce_last_extracts_last_valid_token(self, simple_sequence):
        """Test that last reduction extracts the last valid token."""
        activations, attention_mask = simple_sequence
        result = reduce_last(activations, attention_mask)

        # First sample: last valid is index 2
        assert torch.allclose(result[0], torch.tensor([5.0, 6.0]))

        # Second sample: last valid is index 1
        assert torch.allclose(result[1], torch.tensor([4.0, 5.0]))

    def test_reduce_first_extracts_first_token(self):
        """Test that first reduction extracts the first token."""
        activations = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )

        result = reduce_first(activations, attention_mask)

        assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(result[1], torch.tensor([7.0, 8.0]))


class TestReductionRegistry:
    """Test reduction registry functionality."""

    def test_get_builtin_reductions(self):
        """Test retrieving built-in reduction functions."""
        builtin_names = ["mean", "max", "last", "first"]

        for name in builtin_names:
            fn = get_reduction_function(name)
            assert callable(fn)

    def test_get_nonexistent_reduction_raises_error(self):
        """Test that requesting unknown reduction raises ValueError."""
        with pytest.raises(ValueError, match="Unknown reduction strategy"):
            get_reduction_function("nonexistent_strategy")

    def test_register_custom_reduction(self):
        """Test registering a custom reduction function."""

        @register_reduction("test_custom")
        def custom_reduction(activations, attention_mask):
            # Simple custom reduction: just take first token and double it
            return activations[:, 0, :] * 2

        # Should be able to retrieve it
        fn = get_reduction_function("test_custom")
        assert fn is custom_reduction

        # Test it works
        activations = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        attention_mask = torch.ones(1, 2)
        result = fn(activations, attention_mask)
        expected = torch.tensor([[2.0, 4.0]])
        assert torch.allclose(result, expected)


class TestBatchedApplication:
    """Test batched reduction application."""

    def test_batched_application_consistency(self, large_batch):
        """Test that batched application gives same result with different batch sizes."""
        activations, attention_mask = large_batch
        reduction_fn = reduce_mean

        # Apply with large batch (effectively single batch)
        result_single = apply_reduction_batched(
            activations, attention_mask, reduction_fn, batch_size=100, show_progress=False
        )

        # Apply with small batches
        result_batched = apply_reduction_batched(
            activations, attention_mask, reduction_fn, batch_size=10, show_progress=False
        )

        assert torch.allclose(result_single, result_batched, rtol=1e-5)

    def test_batched_application_device_handling(self):
        """Test that batched application handles device placement correctly."""
        n_samples, seq_len, hidden_dim = 20, 5, 8
        activations = torch.randn(n_samples, seq_len, hidden_dim)
        attention_mask = torch.ones(n_samples, seq_len)

        # Force CPU
        result = apply_reduction_batched(
            activations,
            attention_mask,
            reduce_mean,
            batch_size=10,
            device_override=torch.device("cpu"),
            show_progress=False,
        )

        # Result should be on CPU
        assert result.device.type == "cpu"
        assert result.shape == (n_samples, hidden_dim)
