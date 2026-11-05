# Distribution-mixture shift experiments

Implementation and data decision, 5 November 2026.

The primary experiment reuses the exact official attention/attention score artifacts. It changes only the domain labels used for calibration and deployment-mixture evaluation: `mt` and `mts` both map to `medical`. It does not change, deduplicate, relabel, expand or move any benchmark examples between development and test.

## What is already computed, and what the new experiment computes

There are two distinct stages:

1. **Model scoring, already complete:** fit the safety and DV probes, obtain expert predictions, and save one safety score, expert score and DV score per example. These scores do not depend on a delegation threshold. Existing artifacts cover both experts and all three seeds.
2. **Risk control and evaluation, implemented here:** split the saved test rows into ordering, hypothesis-testing and evaluation partitions; select a threshold with pooled or domainwise LTT; evaluate that fixed threshold under deployment mixtures.

`--inventory-only` performs neither training nor risk control. A normal run does select thresholds. No refitting, activation extraction, Modal access or GPU is needed for these saved-score experiments.

## Fixed benchmark inventory

| Original source | DV development rows | Test rows | Deployment domain |
|---|---:|---:|---|
| Anthropic HH | 500 | 500 | Anthropic |
| MTSamples (`mt`) | 278 | 500 | Medical |
| MTS-Dialog (`mts`) | 274 | 86 | Medical |
| ToolACE | 328 | 500 | ToolACE |
| **Total** | **1,380** | **1,586** | **Three domains** |

The safety probe was fitted on the same 8,000 synthetic training examples as the official experiments. The original per-source 500-row cap and seed-specific selections are preserved by using the saved scores. In particular, this primary experiment does not expand MT to all 604 released balanced test rows. The medical domain contains **586** test examples; the earlier proposal to remove source families and use 468 medical examples is superseded.

Keep the original 50% calibration / 50% evaluation split. Of the 793 calibration examples, 237 are used for ordering (D_est) and 556 for hypothesis testing (D_cal). These are global random splits; the medical domain contributes approximately 205 hypothesis-testing examples, with exact domain and source counts recorded in every trial.

Original JSONL filenames:

| Source | Under `evals/dev/` | Under `evals/test/` |
|---|---|---|
| Anthropic | `anthropic_balanced_apr_23.jsonl` | `anthropic_test_balanced_apr_23.jsonl` |
| MT | `mt_balanced_apr_30.jsonl` | `mt_test_balanced_apr_30.jsonl` |
| MTS | `mts_balanced_apr_22.jsonl` | `mts_test_balanced_apr_22.jsonl` |
| ToolACE | `toolace_balanced_apr_22.jsonl` | `toolace_test_balanced_apr_22.jsonl` |

### Inherited overlaps are reported, not repaired

The full balanced source files have disjoint development/test IDs and no identical complete inputs shared between their respective development and test splits. Related source content nevertheless exists: for example, MT records 1097 and 4913 repeat the same clinical note under different specialties, and MTS test record `test_chat_19` contains a note section present in MT development record 4027. MTS-Dialog was constructed from MTSamples note sections; see the [original paper](https://aclanthology.org/2023.eacl-main.168/).

The inventory reports exact input overlap, repeated normalised MT note texts, shared Anthropic chosen/rejected preference-pair families, and long MTS note sections contained in MT development notes. It does not infer patient identity or claim exhaustive semantic deduplication. **No rows are excluded.** Since the source score arrays and original dev/test membership are unchanged, merging domain labels introduces no additional examples or cross-split reassignment. These inherited dependencies remain a limitation of interpreting the benchmark through the paper's sampling assumptions.

Legacy score artifacts do not store example IDs or original JSONL hashes. With `--data-root`, the inventory reconstructs IDs using the metadata's sampling rule, checks all labels and test-source ordering, and records the current file hashes. This is transparent reconstruction, not a claim that the original artifact contained IDs.

## Methods and guarantees

The runner evaluates three methods on identical partitions and model scores:

- **`pooled`:** the existing `build_risk_pareto_ltt` implementation, with its original pooled Pareto frontier, ordering and objective.
- **`mixture`:** domainwise risk evaluation, worst-domain constraint estimates for the Pareto frontier, a reference-mixture complementary objective, and the maximum domain p-value at each threshold.
- **`pooled_matched`:** pooled testing with exactly the mixture method's candidates, ordering and complementary objective. This isolates the effect of domainwise testing from candidate construction.

For domain risks R_g and deployment weights q,

\[
R_q(\lambda)=\sum_g q_g R_g(\lambda),\qquad
\sup_{q\in\Delta_G}R_q(\lambda)=\max_g R_g(\lambda).
\]

For each candidate, apply the existing accuracy Hoeffding–Bentkus or budget binomial test using each domain's own hypothesis-testing sample size. Combine them as

\[
p_{\mathrm{mix}}(\lambda)=\max_g p_g(\lambda).
\]

A missing hypothesis-testing domain gets p=1 and prevents certification. Fixed-sequence testing runs at δ=0.10 and stops at the first non-rejection. There is no extra correction by the number of domains. Candidate filtering, ordering and the complementary objective use D_est only; the 100-point DV threshold grid comes from development scores only. Budget thresholds run largest to smallest; accuracy thresholds are ordered by estimated worst-domain error, then estimated complementary cost.

The reference objective uses q₀=(500,586,500)/1586 in Anthropic/Medical/ToolACE order. Evaluation includes the three-domain simplex at resolution 10 (66 grid points in 10-percentage-point increments), plus q₀ and the uniform mixture, giving 68 mixtures in total. Held-out domain risks are computed once for each selected policy and weighted to obtain mixture risks. The threshold is not recalibrated at different deployment mixtures.

Medical source composition is fixed as part of the medical domain; arbitrary MT-versus-MTS shifts are not covered by this three-domain claim. Original source metrics remain in the output as diagnostics. Accuracy error and expert-call rate are the supported per-example risks; AUROC is not certified by this implementation. Returning no certified threshold is explicit, with no uncalibrated fallback.

Each target/expert configuration has its own guarantee under the paper's sampling assumptions. The output's held-out exceedance flag is an empirical measurement, not direct evidence of a population-guarantee violation. Repeated splits reuse the same finite dataset and are not independent fresh datasets. Coverage must not be reported only conditional on successful selection.

## Separate configurations

- `experiments/configs/mixture_shift/accuracy.yaml`: attention/attention, both experts, seeds 42/43/44, existing minimum-accuracy sweeps; one official split per seed.
- `experiments/configs/mixture_shift/budget.yaml`: the same model artifacts and seeds, with 20 budget targets from 0.05 to 0.95.
- `experiments/configs/mixture_shift/accuracy_resplits.yaml`: seed-42 models, 500 paired splits starting at seed 1000; minimum accuracy 0.80 for the strong expert and 0.78 for the weak expert.

Accuracy configuration targets are **minimum accuracies**; the runner converts them to error bounds. Budget configuration targets are **maximum expert-call rates**. Other groupings can be supplied through the source-to-domain mapping and reference weights; original score arrays remain immutable. Joint accuracy/budget certification is not part of this first implementation.

## Commands and outputs

### How to present the mixtures

Use simplex maps instead of a table with one row per deployment mixture. The vertices are pure Anthropic, Medical and ToolACE; mark the uniform mixture and original benchmark mixture explicitly. Each selected policy stays fixed across the triangle. For these per-example risks, its empirical risk is affine in the mixture weights, so the map can be rendered exactly from the three held-out domain risks. The denser colour rendering adds no observations or calibration runs.

The main quantities are:

| Quantity | Accuracy-controlled experiment | Budget-controlled experiment |
|---|---|---|
| Target margin (positive is good) | Achieved accuracy minus required accuracy | Allowed call rate minus actual call rate |
| Complementary performance | Expert-call rate: the cost of robustness | Achieved accuracy: the performance cost of robustness |
| Worst deployment mixture | Minimum domain accuracy | Maximum domain call rate |
| Selection availability | Number of seeds/splits returning a certified threshold | Number of seeds/splits returning a certified threshold |

Show the target margin on a shared diverging colour scale, with a zero contour separating mixtures whose empirical risk meets the target from those exceeding it. Compare pooled and domainwise calibration side by side; include the matched-candidate pooled control in supplementary output. Show the complementary metric in a second row. Report individual seeds or an explicitly labelled summary; do not interpret an average over seeds as a guarantee for every fitted policy, or silently drop configurations that return no threshold.

Use only five named mixtures in a compact table: the three vertices, uniform and original benchmark. Record the selected threshold and selection availability once per configuration, since they do not vary across deployment mixtures. A separate target-sweep summary should show selection availability and worst-domain risk, including strict targets for which no policy is certified. The fraction of grid points meeting a target is optional and descriptive: it depends on how the grid is sampled and is not a deployment probability or a coverage estimate.

Initial fixed-target views use the already specified minimum accuracies (strong 0.80, weak 0.78) and the configured budget target nearest 0.50. These targets are selected before inspecting held-out results. All configured targets and both experts remain in the outputs. The 500-resplit configuration is a separate stability analysis; the initial figures use the three official model seeds.

The [initial results](mixture_shift_initial_results.md) record the completed 68-mixture runs, including nonselection at the configured accuracy targets.

Run from the repository root. Set the artifact root to the directory containing `strong/` and `weak/`, normally `results/probe_ablation/cuda_run`.

```bash
# Read-only input inventory; no thresholds are selected.
uv run experiments/mixture_shift.py \
  --config experiments/configs/mixture_shift/accuracy.yaml \
  --artifact-root "$CTD_ARTIFACT_ROOT" \
  --data-root "$DATA_DIR" \
  --inventory-only

# Full risk-control/evaluation stage; omit --data-root if only scores are available.
uv run experiments/mixture_shift.py \
  --config experiments/configs/mixture_shift/accuracy.yaml \
  --artifact-root "$CTD_ARTIFACT_ROOT"

# A limited repeated-split run before the full 500-split configuration.
uv run experiments/mixture_shift.py \
  --config experiments/configs/mixture_shift/accuracy_resplits.yaml \
  --artifact-root "$CTD_ARTIFACT_ROOT" --max-trials 1

# Empirical simplex maps and summaries from one completed run.
uv run experiments/mixture_shift_report.py \
  results/mixture_shift/<completed-run> \
  --output-dir results/mixture_shift/<completed-run>/report_seed42
```

Outputs go to a new directory under `results/mixture_shift/` by default. `--output-dir` can set a new destination; existing directories and destinations inside the source artifact root are rejected.

- `config.json`: exact effective configuration.
- `inventory.json`: artifact identity, hashes and counts; optional reconstructed row manifests and inherited-overlap report; completion status.
- `mixtures.json`: domain order, deployment weights and reference weights.
- `trials.jsonl`: one record per expert/model-seed/split, containing exact split indices, domain/source counts, candidate thresholds, p-values, certified indices, selected threshold, held-out domain/source risks and mixture risks. Absent in inventory-only mode.

The report command exports PNG/SVG simplex figures for seed 42, including a supplementary matched-candidate comparison; all-target certification-availability plots; and `summary.json` with all seeds, targets, explicit nonselection, worst-domain metrics and the five named mixtures. It never recalibrates thresholds. Pass `--seed` to display another official seed. The default budget view uses 47.6316%, the nearest configured target to 50% with ties resolved toward the lower target.

Large inputs and outputs are not committed. The existing experiment entry points and shared calibration implementations are unchanged. The new `reliable_monitoring/mixture_ltt.py` composes their risk evaluation, bounds, Pareto mask and fixed-sequence testing; the dedicated experiment driver handles configuration and reporting.

## Remote readiness

The lab server's original score artifacts and full token-activation caches were located and verified. The primary workflow needs only the score files. The older remote checkout should be left in place; use a separate checkout of the PR branch and the existing environment with that checkout on `PYTHONPATH`. The inherited benchmark files and activation caches are read-only inputs.

The checkpoint search found legacy difference-of-means and Gemma-based probe files, not verified attention/attention checkpoints. This does not block the score-based experiment. New score generation or probe refitting is only needed if we later change the model or example selection, and should be a separate recorded run.
