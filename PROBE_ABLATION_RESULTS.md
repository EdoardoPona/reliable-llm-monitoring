# Probe architecture and accuracy-guarantee results

This note records the experiments added after the COLM 2026 submission. It covers the probe
architecture ablation, the reverse accuracy-guarantee formulation, empirical validation against
uncorrected calibration, and the reproduction check for the submitted budget-guarantee results.

## 1. Probe architecture ablation

### Setup

The experiment uses the same data, activation model, layer, experts, calibration split, and cascade
settings as the submitted paper:

- activation model: Llama-3.2-1B-Instruct, layer 11;
- strong expert: Gemma-3-27B-IT;
- weak expert: Llama-3.2-1B-Instruct;
- confidence: \(1-\delta=0.9\);
- guaranteed risk: delegation budget;
- Pareto objective: accuracy error;
- batch size for top-\(k\) baselines: 128;
- seeds: 42, 43, and 44.

We ran four matched safety/DV probe families:

| Name | Safety probe | DV probe |
|---|---|---|
| `mean_ridge` | Logistic regression on mean-pooled activations | Ridge regression on mean-pooled activations |
| `attention_attention` | AttnLite attention pooling with a classification head | AttnLite attention pooling with a regression head |
| `softmax_softmax` | Softmax-weighted aggregation of per-token linear scores | The corresponding softmax-weighted regressor |
| `mlp_mlp` | MLP on mean-pooled activations | MLP regressor on mean-pooled activations |

We also ran three ridge-anchor cells: `attention_ridge`, `softmax_ridge`, and `mlp_ridge`. These keep
the new safety probe but replace the matched DV probe with the original ridge DV probe.

The full sweep contains 7 architecture pairs, 2 experts, and 3 seeds, for 42 runs.

### Matched-architecture results

The following values are means over three seeds. “Gain” is CTD AUROC minus calibrated-uncertainty
AUROC at a target delegation budget of 30%.

#### Strong expert

| Architecture | Probe AUROC | DV Spearman | Delegation capacity | CTD AUROC | Gain |
|---|---:|---:|---:|---:|---:|
| Mean | 0.810 | 0.272 | 0.842 | 0.886 | +0.050 |
| Attention | 0.848 | 0.329 | 0.842 | 0.910 | +0.070 |
| Softmax | 0.810 | 0.497 | 0.849 | 0.862 | +0.026 |
| MLP | 0.811 | 0.303 | 0.823 | 0.889 | +0.042 |

#### Weak expert

| Architecture | Probe AUROC | DV Spearman | Delegation capacity | CTD AUROC | Gain |
|---|---:|---:|---:|---:|---:|
| Mean | 0.810 | 0.503 | 0.332 | 0.859 | +0.074 |
| Attention | 0.848 | 0.571 | 0.343 | 0.892 | +0.061 |
| Softmax | 0.810 | 0.585 | 0.449 | 0.868 | +0.091 |
| MLP | 0.811 | 0.597 | 0.327 | 0.874 | +0.078 |

The main observations are:

1. Attention is the strongest safety probe and usually gives the highest absolute cascade AUROC.
2. CTD improves over calibrated uncertainty for every matched architecture and both experts.
3. With the weak expert, the largest relative gain is obtained by softmax at high budgets:
   +0.153 AUROC at a 50% target budget. Attention still has the highest absolute cascade AUROC.
4. The matched DV architecture does not consistently outperform the ridge anchor. At a 30% budget,
   the matched probe is slightly better for all three new families, but the ordering is mixed at
   other budgets. A simple ridge DV probe remains competitive.
5. The qualitative behaviour from the submitted paper is preserved: CTD delegates selectively and
   plateaus when further delegation is harmful, while fixed-quota methods continue to spend the
   available budget.

### Files

- Full result directory: `results/probe_ablation/cuda_run/`
- Per-run scores and metadata:
  `results/probe_ablation/cuda_run/<expert>/<architecture>/seed_<seed>/`
- Aggregate table: `results/probe_ablation/cuda_run/summary_aggregate.csv`
- Aggregate JSON: `results/probe_ablation/cuda_run/summary.json`
- Run manifest: `results/probe_ablation/cuda_run/manifest.json`
- Paper-style figures: `results/probe_ablation/cuda_run/paper_figures/`
- Four-architecture grid:
  `results/probe_ablation/cuda_run/paper_figures/architecture_comparison_grid_B128.pdf`
- Strong-expert architecture comparison:
  `results/probe_ablation/cuda_run/paper_figures/strong_architecture_comparison_B128.pdf`
- Weak-expert architecture comparison:
  `results/probe_ablation/cuda_run/paper_figures/weak_architecture_comparison_B128.pdf`

The per-run `scores.npz` files contain the probe, expert, DV, label, and group arrays needed to
recompute all metrics and plots without retraining.

## 2. Accuracy guarantee with budget minimisation

### Setup

This experiment reverses the submitted formulation:

- guaranteed risk: accuracy error;
- objective: delegation budget;
- confidence: \(1-\delta=0.9\);
- architectures: the four matched families;
- experts: strong and weak;
- seeds: 42, 43, and 44.

The minimum accuracy targets are:

- strong expert: 0.74, 0.76, 0.78, 0.80, and 0.82;
- weak expert: 0.72, 0.74, 0.76, 0.78, and 0.80.

### Results

At a guaranteed minimum accuracy of 0.78 with the strong expert:

| Architecture | CTD delegation | Calibrated-uncertainty delegation |
|---|---:|---:|
| Mean | 0.388 | 0.634 |
| Attention | 0.269 | 0.387 |
| Softmax | 0.408 | 0.728 |
| MLP | 0.314 | 0.650 |

All three seeds certified this target. Attention requires the least delegation.

At a guaranteed minimum accuracy of 0.76 with the weak expert:

| Architecture | CTD delegation | Certified seeds |
|---|---:|---:|
| Mean | 0.208 | 3/3 |
| Attention | 0.148 | 3/3 |
| Softmax | 0.282 | 3/3 |
| MLP | 0.200 | 2/3 |

Calibrated uncertainty did not certify any threshold for the weak expert at any tested target.

The highest targets are not always feasible. At strong-expert accuracy 0.82, CTD certified 6 of the
12 architecture/seed runs. At weak-expert accuracy 0.80, it certified 4 of 12. Figures at these
targets must report the number of selected runs rather than averaging successful runs without
qualification.

### Files

- Configuration: `experiments/configs/probe_ablation_accuracy_guarantee.yaml`
- Result directory: `results/probe_ablation_accuracy_guarantee/20260723_215102/`
- Full JSON: `results/probe_ablation_accuracy_guarantee/20260723_215102/results.json`
- Figure: `results/probe_ablation_accuracy_guarantee/20260723_215102/budget_vs_guaranteed_accuracy.pdf`

## 3. LTT versus empirical calibration

### Setup

We compared LTT with a plug-in empirical calibrator over 500 random calibration/evaluation splits
for each architecture and expert.

The two methods use:

- the same threshold grid;
- the same Pareto candidates;
- the same optimization split;
- the same hypothesis-testing split;
- the same held-out evaluation split.

LTT applies the one-sided accuracy-risk test. The empirical method keeps every threshold whose
empirical accuracy on the hypothesis-testing split exceeds the target and chooses the one with the
lowest budget on the optimization split. It applies no statistical correction.

The targets are 0.80 for the strong expert and 0.78 for the weak expert. A target of 0.85 was not
used because it is above expert-only accuracy and would produce almost no LTT selections.

### Results

“Failure” is the fraction of all 500 trials whose selected cascade falls below the target on the
held-out split. Non-selection is not counted as a failure. The JSON also records failure rates
conditional on selection.

| Expert | Architecture | LTT selected | LTT failure | Empirical selected | Empirical failure |
|---|---|---:|---:|---:|---:|
| Strong | Mean | 80.0% | 5.2% | 99.6% | 41.6% |
| Strong | Attention | 66.0% | 5.6% | 99.2% | 43.6% |
| Strong | Softmax | 58.6% | 5.2% | 98.2% | 43.4% |
| Strong | MLP | 83.0% | 6.4% | 99.8% | 47.2% |
| Weak | Mean | 18.4% | 4.6% | 95.4% | 44.6% |
| Weak | Attention | 84.2% | 4.8% | 100.0% | 44.0% |
| Weak | Softmax | 95.8% | 4.4% | 99.8% | 41.2% |
| Weak | MLP | 88.8% | 4.2% | 100.0% | 42.6% |

The LTT failure rate is below \(\delta=0.1\) in all eight settings. The empirical method fails in
41--47% of trials. The LTT accuracy distribution is shifted to the right in every panel.

Selection rates are essential for interpretation. In particular, weak-expert mean pooling selects
only 92 of 500 trials. Its unconditional LTT failure rate is 4.6%, but its failure rate conditional
on selection is 25%. The formal guarantee concerns the unconditional probability that the procedure
returns a violating policy.

LTT also delegates more than empirical calibration because it requires an accuracy margin. This is
the cost of obtaining the guarantee.

### Files

- Result directory: `results/ltt_coverage_validation/20260723_221318/`
- Full trials and summary:
  `results/ltt_coverage_validation/20260723_221318/accuracy_guarantee_validation.json`
- PDF figure:
  `results/ltt_coverage_validation/20260723_221318/accuracy_guarantee_validation.pdf`
- PNG figure:
  `results/ltt_coverage_validation/20260723_221318/accuracy_guarantee_validation.png`

This is an empirical validation using finite held-out splits. It is not an additional proof of the
population guarantee.

## 4. Reproduction of the submitted budget-guarantee results

The generalized risk-control implementation was checked by replaying the exact strong- and
weak-expert COLM configurations in local-cache-only mode.

The replay matches the archived results in:

- all reported AUROC and accuracy values;
- selected policies and realized budgets;
- top-\(k\) and oracle results;
- FAIC scores;
- saved configurations.

The only differences are threshold values at approximately \(10^{-15}\), caused by floating-point
rounding. They do not change any selected examples or reported results.

The archived outputs are:

- strong expert: `results/dv_cascade_comparison/20260401_111720/`;
- weak expert: `results/dv_cascade_comparison/20260401_111850/`.

The corresponding main figures are:

- `results/dv_cascade_comparison/20260401_111720/continuous_ranking_comparison_B128.pdf`;
- `results/dv_cascade_comparison/20260401_111850/continuous_llama1b_ranking_comparison_B128.pdf`.

## 5. Reproduction commands

Rebuild the architecture summaries and figures from saved scores:

```bash
uv run experiments/probe_ablation_analysis.py results/probe_ablation/cuda_run
```

Run the reverse accuracy-guarantee analysis:

```bash
uv run experiments/probe_ablation_analysis.py \
  --config experiments/configs/probe_ablation_accuracy_guarantee.yaml
```

Run the repeated-split LTT versus empirical validation:

```bash
uv run experiments/ltt_coverage_validation.py \
  --config experiments/configs/probe_ablation_accuracy_guarantee.yaml
```

Replay the submitted strong-expert experiment without allowing cache computation:

```bash
uv run experiments/dv_cascade_comparison.py \
  --config experiments/configs/dv_cascade/dv_cascade_comparison_continuous_strong_acc.yaml \
  --output-dir results/dv_cascade_replay_strong \
  --file-prefix continuous_ \
  --local-only
```

Use `dv_cascade_comparison_continuous_weak_acc.yaml` and the prefix `continuous_llama1b_` for the
weak-expert replay.
