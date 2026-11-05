# Reliable LLM Monitoring

Code for the experiments in
[Calibrate-Then-Delegate: Safety Monitoring with Risk and Budget Guarantees via Model Cascades](https://arxiv.org/abs/2604.14251).

## Setup

The project uses Python 3.12 and `uv`:

```bash
uv sync
```

Set `DATA_DIR` to the directory containing the training and evaluation JSONL
files. Large caches and outputs can be placed on a separate disk:

```bash
export DATA_DIR=/path/to/data
export RELIABLE_MONITORING_CACHE_DIR=/path/to/cache
export RESULTS_DIR=/path/to/results
```

`ACTIVATION_CACHE_DIR` and `BASELINE_CACHE_DIR` override the two cache
locations separately. Experiments fetch cached activations and expert scores
when available; a cache miss may run model inference locally or on Modal,
according to the selected configuration.

The training and evaluation JSONL files are released by McKenzie et al. in the
[models-under-pressure dataset table](https://github.com/arrrlex/models-under-pressure#datasets).
Download the balanced training, Anthropic HH, MT, MTS, and ToolACE files and
preserve the directory layout used by the experiment configurations.

## Main probe-architecture experiments

Train the seven probe-ablation cells for both experts and three seeds. Each
cell saves `scores.npz` and `metadata.json`; all subsequent architecture
analyses use these files without loading language models.

```bash
uv run experiments/probe_ablation_sweep.py \
  --config experiments/configs/probe_ablation.yaml
```

Use the resulting run directory as the input to downstream analyses. For
config-driven analyses, set `artifact_root` in the corresponding YAML file.

Reconstruct the budget-control results, method-specific score-merging
comparison, fixed-merging controls, architecture comparison, weak-expert
overwrite stress test, and budget-utilisation figure:

```bash
uv run experiments/probe_ablation_analysis.py \
  <run-directory> \
  --merge-strategy method-specific
```

Outputs are stored under
`<run-directory>/fusion_method_specific/`. The `paper_figures` directory
contains the main and supplementary cascade figures.

Run the saved-score analyses used for the DV-probe metrics and group table:

```bash
uv run experiments/probe_analysis.py \
  --config experiments/configs/probe_ablation_probe_analysis.yaml

uv run experiments/group_analysis.py \
  --config experiments/configs/probe_ablation_group_analysis.yaml
```

These commands create timestamped directories beneath the `output_dir` in
each YAML file. The probe analysis exports `attention_mean_v_at_k.pdf`; the
group analysis exports per-seed and aggregate CSV/JSON results.

## Performance control

Certify minimum accuracy while minimising delegation, using the saved
architecture scores:

```bash
uv run experiments/probe_ablation_analysis.py \
  --config experiments/configs/probe_ablation_accuracy_guarantee.yaml
```

Validate CTD against the plug-in empirical selector over repeated random
splits:

```bash
uv run experiments/ltt_coverage_validation.py \
  --config experiments/configs/probe_ablation_accuracy_guarantee.yaml
```

The first command exports `budget_vs_guaranteed_accuracy.pdf`. The second
exports `attention_accuracy_guarantee_validation.pdf` and
`accuracy_guarantee_validation_all_architectures.pdf`, together with the JSON
trial data needed to replot them.

## Distribution-mixture shift

Reuse the official attention/attention scores to compare pooled and domainwise
risk control, grouping MT and MTS as medical while preserving every benchmark
example. The [experiment plan](docs/mixture_shift_experiment_plan.md) describes
the data inventory, guarantees, configurations and output files.

```bash
uv run experiments/mixture_shift.py \
  --config experiments/configs/mixture_shift/accuracy.yaml \
  --artifact-root results/probe_ablation/cuda_run
```

Add `--inventory-only` to check the inputs without selecting thresholds.

## Mean+Ridge supplementary experiments

The six configurations in `experiments/configs/dv_cascade/` reproduce the
Mean+Ridge binary/continuous-DV runs, strong/weak experts, batch-size sweep,
and accuracy- versus AUROC-objective comparisons. For example:

```bash
uv run experiments/dv_cascade_comparison.py \
  --config experiments/configs/dv_cascade/dv_cascade_comparison_continuous_strong_acc.yaml \
  --file-prefix continuous_
```

Run the corresponding weak-expert and AUROC configurations in the same way.
The binary configurations provide the inputs for:

```bash
uv run experiments/dv_binary_vs_continuous.py \
  --binary-strong /path/to/binary-strong/results.json \
  --binary-weak /path/to/binary-weak/results.json \
  --cont-strong /path/to/continuous-strong/results.json \
  --cont-weak /path/to/continuous-weak/results.json \
  --output-dir results/dv_binary_vs_continuous
```

This exports the binary-versus-continuous tables and `dv_delta_strong.pdf` /
`dv_delta_weak.pdf` used in the supplement.

## Budget-guarantee validation

The original Mean+Ridge budget guarantee is validated over 500 random splits
for each expert:

```bash
uv run experiments/ltt_coverage_validation.py \
  --config experiments/configs/ltt_coverage/strong_expert.yaml

uv run experiments/ltt_coverage_validation.py \
  --config experiments/configs/ltt_coverage/weak_expert.yaml
```

The output directories contain `coverage_strong_expert.pdf` or
`coverage_weak_expert.pdf` and the underlying `coverage_results.json`.

## Verification

The test suite is model-free:

```bash
uv run pytest -q
```

Formatting, linting, and type checks are run with:

```bash
./lint.sh
```
