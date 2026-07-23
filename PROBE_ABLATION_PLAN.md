# Phase 1: probe architecture ablations

Goal: substantiate the claim that CTD is agnostic to the probe architecture. We currently only show
results for a single probe (logistic regression on mean-pooled activations) and a single DV probe
(ridge on the same features). McKenzie et al. ablate safety probe architectures; we extend this to
the cascade setting, where the DV probe architecture is a new axis nobody has studied.

## Architectures

Four safety probe architectures, each with a matched DV regressor:

| Safety probe | DV probe (matched) |
|---|---|
| Logistic regression on mean-pooled activations (current) | Ridge on mean-pooled activations (current) |
| Attention (AttnLite, McKenzie's best) | AttnLite with regression head |
| Softmax aggregation (LinearThenSoftmax) | Softmax-aggregated linear regressor |
| MLP on mean-pooled activations | MLP regressor |

Max and last-token probes are excluded: they underperform in McKenzie et al. and add nothing
conceptually. The MLP is the one architecture McKenzie et al. did not cover; it tests whether the
capacity class of the probe matters for CTD.

## Design

We do not run the full safety x DV product. The DV target v(x, y) = P_expert - P_probe depends on
the safety probe, so a full grid would not be a clean factorisation anyway. Instead:

- **Diagonal (main design):** DV architecture matched to the safety architecture, i.e. the method
  instantiated entirely within one architecture family. This is how a practitioner would deploy it.
- **Ridge anchor:** every safety probe also run with the current ridge-on-mean DV probe. The
  diagonal-vs-anchor gap at fixed safety probe isolates the DV architecture effect. It also guards
  against high-capacity DV probes (AttnLite, MLP) overfitting the small dev split (~1.4k samples):
  if a matched DV cell underperforms, the anchor tells us whether CTD or the DV probe is at fault.

Total: 7 cells (4 diagonal + 3 anchor) x 2 experts (Gemma-3-27B strong, Llama-3.2-1B weak)
x 3 seeds for the torch probes. All cells run on cached Llama-3.2-1B layer-11 activations and
cached baseline scores, so the sweep itself is CPU-cheap.

Everything else follows the paper: same splits (dev for DV training, test half calibration / half
evaluation), 20 budget levels, delta = 0.1, batch size 128 for the top-k baselines. Torch probe
hyperparameters from McKenzie et al. Appendix A.1 (LR 5e-3 to 1e-4, 200 epochs, early stop,
softmax temperature 5).

## Per-cell metrics

- Probe AUROC and the CTD-vs-uncertainty gain at 3-4 budget levels (headline).
- Delegation capacity (fraction of samples with v > 0). A strong attention probe against the weak
  expert should shrink this towards zero; the prediction is that CTD certifies a near-zero
  delegation rate while uncertainty top-k spends its full budget for nothing.
- DV Spearman and mean-v-at-k, to check that better DV ranking tracks bigger cascade gains.

## Engineering steps

1. **Probe registry.** Add a factory and registry in `reliable_monitoring/probes.py` (mirroring the
   existing reduction / selection / risk registries) with a `probe` config block (`type`,
   `hyperparams`). Replace the ~8 hardcoded `SequenceProbe(...)` construction sites.
2. **Torch probes.** A `TorchSequenceProbe` satisfying the `Probe` protocol, wrapping the
   architectures from the models-under-pressure fork (`AttnLite`, `LinearThenSoftmax`,
   `MeanThenLinear`) plus a new MLP module. Sigmoid outputs feed the existing platt/isotonic
   calibration unchanged.
3. **Raw activation cache.** Sequence-aggregating probes need per-token activations, but the
   activation cache only stores reduced ones. Extend `dataset.py` / `activation_registry.py` with a
   `raw` cache entry (fp16), keyed `model__L{layer}__raw__{dataset}`. A few GB at 1B scale; do not
   enable for 70B.
4. **DV probe factory.** Lift `train_dv_probe` / `predict_dv_scores` out of
   `experiments/delegation_value_probe.py` into the package, with its own registry and a `dv_probe`
   config block. Include a flag for the attention DV probe to reuse the safety probe's frozen
   attention weights (fallback if from-scratch training overfits; default is from-scratch).
5. **Sweep configs.** Sweep configs over the 7 cells x 2 experts, reusing saved score artifacts.
   Empirical validation of the budget guarantee is already covered by
   `ltt_coverage_validation.py` and is independent of the probe architecture.

## Deliverables

- Summary figure: CTD-vs-uncertainty AUC gain at fixed budgets, grouped by architecture,
  strong/weak expert panels.
- Table: probe AUROC, DV Spearman, delegation capacity, CTD gain per cell.
- Per-architecture replicas of the main figure in the appendix.

## Running the experiment

```bash
uv run experiments/probe_ablation_sweep.py --config experiments/configs/probe_ablation.yaml
uv run experiments/probe_ablation_analysis.py results/probe_ablation/<run>
```

The sweep resumes from existing `scores.npz` files. Each file contains the probe, expert, DV,
label, and group arrays needed to rebuild the metrics and figures. The analysis command does not
load a language model or retrain a probe.
