"""
Test to verify that offline cascade implementation produces identical results to online cascade.

This test compares:
- run_online_cascade: Calls baseline only for uncertain examples (efficient, online)
- compute_all_cascade_scores + run_offline_cascade: Precomputes all scores then applies cascade logic

Both should produce identical final predictions.
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from reliable_monitoring.cascade import run_llm_baseline, run_offline_cascade, run_online_cascade
from reliable_monitoring.dataset import ActivationConfig, load_dataset, sample_from_dataset
from reliable_monitoring.probes import SequenceProbe

# Load environment
load_dotenv()

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LAYER = 11
DATA_DIR = os.environ["DATA_DIR"]
TRAIN_SAMPLE_SIZE = 50  # Small sample for probe training
TEST_SAMPLE_SIZE = 50  # Small sample for fast testing


def test_online_offline_equivalence():
    """
    Test that offline cascade produces identical results to online cascade.
    """
    print("=" * 80)
    print("Testing Cascade Equivalence: Online vs Offline")
    print("=" * 80)

    # Setup activation config
    activation_config = ActivationConfig(model_name=MODEL_NAME, layer=LAYER)

    # Load and prepare datasets
    print("\n1. Loading datasets...")
    train_dataset = sample_from_dataset(
        load_dataset(
            Path(f"{DATA_DIR}/training/prompts_4x/train.jsonl"),
            activation_config=activation_config,
        ),
        TRAIN_SAMPLE_SIZE,
    )

    test_dataset_full = load_dataset(
        Path(f"{DATA_DIR}/evals/test/anthropic_test_balanced_apr_23.jsonl"),
        activation_config=activation_config,
    )

    # Use a smaller sample for testing
    test_dataset = sample_from_dataset(test_dataset_full, TEST_SAMPLE_SIZE)
    print(f"   ✓ Loaded test dataset with {len(test_dataset)} examples")

    # Train probe
    print("\n2. Training probe...")
    probe = SequenceProbe(reduction_strategy="mean")
    probe.fit(train_dataset)
    print("   ✓ Probe trained")

    # Test parameters - just 2 test cases to keep it fast
    test_cases = [
        {"threshold": 0.5, "merge_strategy": "avg", "baseline_batch_size": 8},
        {"threshold": 0.5, "merge_strategy": "replace", "baseline_batch_size": 8},
    ]

    all_tests_passed = True

    for i, params in enumerate(test_cases, 1):
        threshold = params["threshold"]
        merge_strategy = params["merge_strategy"]
        baseline_batch_size = params["baseline_batch_size"]

        print(f"\n3. Test Case {i}/{len(test_cases)}")
        print(f"   Threshold: {threshold}, Merge Strategy: {merge_strategy}")
        print("-" * 80)

        # Run online cascade
        print("\n   Running ONLINE cascade...")
        online_results = run_online_cascade(
            probe=probe,
            baseline_model_name=MODEL_NAME,
            dataset=test_dataset,
            selection_strategy="fixed_threshold",
            threshold=threshold,
            baseline_batch_size=baseline_batch_size,
            merge_strategy=merge_strategy,
        )
        print("   ✓ Online cascade completed")

        # Run offline cascade
        print("\n   Running OFFLINE cascade...")
        print("   - Computing all cascade scores...")
        probe_scores = probe.predict(test_dataset)
        baseline_scores = run_llm_baseline(
            baseline_model_name=MODEL_NAME,
            dataset=test_dataset,
            baseline_batch_size=baseline_batch_size,
        )
        print("   - Applying offline cascade logic...")
        offline_results = run_offline_cascade(
            probe_scores=probe_scores,
            baseline_scores=baseline_scores,
            selection_strategy="fixed_threshold",
            threshold=threshold,
            merge_strategy=merge_strategy,
        )
        print("   ✓ Offline cascade completed")

        # Compare results
        print("\n   Comparing results...")
        test_passed = True

        # Compare probe scores
        probe_scores_match = np.allclose(online_results.probe_scores, offline_results.probe_scores)
        if probe_scores_match:
            print("   ✓ Probe scores match")
        else:
            max_diff = np.max(np.abs(online_results.probe_scores - offline_results.probe_scores))
            print(f"   ✗ Probe scores differ! Max difference: {max_diff}")
            test_passed = False

        # Compare used_baseline flags
        baseline_flags_match = np.array_equal(online_results.used_baseline, offline_results.used_baseline)
        if baseline_flags_match:
            print(f"   ✓ Baseline usage flags match ({online_results.used_baseline.sum()} examples used baseline)")
        else:
            print("   ✗ Baseline usage flags differ!")
            test_passed = False

        # Compare baseline scores (only where baseline was used)
        # Note: baseline_scores will be NaN where baseline wasn't called
        baseline_used_mask = online_results.used_baseline
        if baseline_used_mask.any():
            online_baseline_used = online_results.baseline_scores[baseline_used_mask]
            offline_baseline_used = offline_results.baseline_scores[baseline_used_mask]
            baseline_scores_match = np.allclose(online_baseline_used, offline_baseline_used, rtol=1e-5, atol=1e-8)
            if baseline_scores_match:
                print("   ✓ Baseline scores match (where baseline was used)")
            else:
                max_diff = np.max(np.abs(online_baseline_used - offline_baseline_used))
                print(f"   ✗ Baseline scores differ (where baseline was used)! Max difference: {max_diff}")
                test_passed = False

        # Compare final scores
        final_scores_match = np.allclose(
            online_results.final_scores, offline_results.final_scores, rtol=1e-5, atol=1e-8
        )
        if final_scores_match:
            print("   ✓ Final scores match")
        else:
            max_diff = np.max(np.abs(online_results.final_scores - offline_results.final_scores))
            print(f"   ✗ Final scores differ! Max difference: {max_diff}")
            test_passed = False

        # Use pytest assertions for proper test failure reporting
        assert probe_scores_match, f"Probe scores differ (threshold={threshold}, merge={merge_strategy})"
        assert baseline_flags_match, f"Baseline flags differ (threshold={threshold}, merge={merge_strategy})"
        if baseline_used_mask.any():
            assert baseline_scores_match, f"Baseline scores differ (threshold={threshold}, merge={merge_strategy})"
        assert final_scores_match, f"Final scores differ (threshold={threshold}, merge={merge_strategy})"

        # Summary for this test case
        if test_passed:
            print(f"\n   ✅ Test Case {i} PASSED")
        else:
            print(f"\n   ❌ Test Case {i} FAILED")
            all_tests_passed = False

    # Final summary
    print("\n" + "=" * 80)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - Offline cascade is equivalent to online cascade")
    else:
        print("❌ SOME TESTS FAILED - Offline cascade differs from online cascade")
    print("=" * 80)

    assert all_tests_passed, "Some test cases failed"


def test_fixed_threshold_batched_non_batched_equivalence():
    """
    Test that the batched offline cascade produces identical results to the non-batched offline cascade for fixed-threshold selection strategy.
    This result should extend to all selection strategies that do not depend on global statistics.
    """

    print("=" * 80)
    print("Testing Cascade Equivalence: Batched vs Non-Batched Offline Cascade")
    print("=" * 80)

    # Setup activation config
    activation_config = ActivationConfig(model_name=MODEL_NAME, layer=LAYER)

    # Load and prepare datasets
    print("\n1. Loading datasets...")
    train_dataset = sample_from_dataset(
        load_dataset(
            Path(f"{DATA_DIR}/training/prompts_4x/train.jsonl"),
            activation_config=activation_config,
        ),
        TRAIN_SAMPLE_SIZE,
    )

    test_dataset_full = load_dataset(
        Path(f"{DATA_DIR}/evals/test/anthropic_test_balanced_apr_23.jsonl"),
        activation_config=activation_config,
    )

    # Use a smaller sample for testing
    test_dataset = sample_from_dataset(test_dataset_full, TEST_SAMPLE_SIZE)
    print(f"   ✓ Loaded test dataset with {len(test_dataset)} examples")

    # Train probe
    print("\n2. Training probe...")
    probe = SequenceProbe(reduction_strategy="mean")
    probe.fit(train_dataset)
    print("   ✓ Probe trained")

    # Test parameters - just 2 test cases to keep it fast
    test_cases = [
        {"threshold": 0.5, "merge_strategy": "avg", "baseline_batch_size": 8},
        {"threshold": 0.5, "merge_strategy": "replace", "baseline_batch_size": 8},
    ]

    all_tests_passed = True

    for i, params in enumerate(test_cases, 1):
        threshold = params["threshold"]
        merge_strategy = params["merge_strategy"]
        baseline_batch_size = params["baseline_batch_size"]

        print(f"\n3. Test Case {i}/{len(test_cases)}")
        print(f"   Threshold: {threshold}, Merge Strategy: {merge_strategy}")
        print("-" * 80)

        # Run non-batched offline cascade
        print("\n   Running NON-BATCHED OFFLINE cascade...")
        probe_scores = probe.predict(test_dataset)
        baseline_scores = run_llm_baseline(
            baseline_model_name=MODEL_NAME,
            dataset=test_dataset,
            baseline_batch_size=baseline_batch_size,
        )
        non_batched_results = run_offline_cascade(
            probe_scores=probe_scores,
            baseline_scores=baseline_scores,
            selection_strategy="fixed_threshold",
            threshold=threshold,
            merge_strategy=merge_strategy,
        )
        print("   ✓ Non-batched offline cascade completed")
        # Run batched offline cascade
        print("\n   Running BATCHED OFFLINE cascade...")
        batched_results = run_offline_cascade(
            probe_scores=probe_scores,
            baseline_scores=baseline_scores,
            selection_strategy="fixed_threshold",
            threshold=threshold,
            merge_strategy=merge_strategy,
            batch_size=16,
        )
        print("   ✓ Batched offline cascade completed")
        # Compare results
        print("\n   Comparing results...")
        test_passed = True
        # Compare used_baseline flags
        baseline_flags_match = np.array_equal(non_batched_results.used_baseline, batched_results.used_baseline)
        if baseline_flags_match:
            print(f"   ✓ Baseline usage flags match ({non_batched_results.used_baseline.sum()} examples used baseline)")
        else:
            print("   ✗ Baseline usage flags differ!")
            test_passed = False

        # Compare baseline scores (only where baseline was used)
        # Note: baseline_scores will be NaN where baseline wasn't called
        baseline_used_mask = non_batched_results.used_baseline
        if baseline_used_mask.any():
            non_batched_baseline_used = non_batched_results.baseline_scores[baseline_used_mask]
            batched_baseline_used = batched_results.baseline_scores[baseline_used_mask]
            baseline_scores_match = np.allclose(non_batched_baseline_used, batched_baseline_used, rtol=1e-5, atol=1e-8)
            if baseline_scores_match:
                print("   ✓ Baseline scores match (where baseline was used)")
            else:
                max_diff = np.max(np.abs(non_batched_baseline_used - batched_baseline_used))
                print(f"   ✗ Baseline scores differ (where baseline was used)! Max difference: {max_diff}")
                test_passed = False
        # Compare final scores
        final_scores_match = np.allclose(
            non_batched_results.final_scores, batched_results.final_scores, rtol=1e-5, atol=1e-8
        )
        if final_scores_match:
            print("   ✓ Final scores match")
        else:
            max_diff = np.max(np.abs(non_batched_results.final_scores - batched_results.final_scores))
            print(f"   ✗ Final scores differ! Max difference: {max_diff}")
            test_passed = False
        # Use pytest assertions for proper test failure reporting
        assert baseline_flags_match, f"Baseline flags differ (threshold={threshold}, merge={merge_strategy})"
        if baseline_used_mask.any():
            assert baseline_scores_match, f"Baseline scores differ (threshold={threshold}, merge={merge_strategy})"
        assert final_scores_match, f"Final scores differ (threshold={threshold}, merge={merge_strategy})"
        # Summary for this test case
        if test_passed:
            print(f"\n   ✅ Test Case {i} PASSED")
        else:
            print(f"\n   ❌ Test Case {i} FAILED")
            all_tests_passed = False
    # Final summary
    print("\n" + "=" * 80)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - Batched offline cascade is equivalent to non-batched offline cascade")
    else:
        print("❌ SOME TESTS FAILED - Batched offline cascade differs from non-batched offline cascade")
    print("=" * 80)
    assert all_tests_passed, "Some test cases failed"
