# Reliable LLM Monitoring

Code for the Calibrate-Then-Delegate experiments.

## Setup

The project uses Python 3.12 and `uv`:

```bash
uv sync
```

Set `DATA_DIR` to the directory containing the training and evaluation JSONL files. Large caches and outputs can be placed on a separate disk:

```bash
export DATA_DIR=/path/to/data
export RELIABLE_MONITORING_CACHE_DIR=/path/to/cache
export RESULTS_DIR=/path/to/results
```

`ACTIVATION_CACHE_DIR` and `BASELINE_CACHE_DIR` can be set separately if needed.

## Probe architecture experiments

Train the probe-ablation models and save their scores:

```bash
uv run experiments/probe_ablation_sweep.py \
  --config experiments/configs/probe_ablation.yaml \
  --output-dir results/probe_ablation/run
```

The sweep resumes from existing `scores.npz` files. Rebuild the paper analyses and figures without loading language models:

```bash
uv run experiments/probe_ablation_analysis.py \
  results/probe_ablation/run \
  --merge-strategy method-specific
```

Run the performance-control experiment, which guarantees accuracy and minimises delegation:

```bash
uv run experiments/probe_ablation_analysis.py \
  --config experiments/configs/probe_ablation_accuracy_guarantee.yaml
```

Run the repeated-split validation against empirical calibration:

```bash
uv run experiments/ltt_coverage_validation.py \
  --config experiments/configs/probe_ablation_accuracy_guarantee.yaml
```

Additional saved-score analyses are configured in:

- `experiments/configs/probe_ablation_probe_analysis.yaml`
- `experiments/configs/probe_ablation_group_analysis.yaml`
- `experiments/configs/probe_ablation_batch_analysis.yaml`

These analyses require the score artifacts produced by the sweep, but do not retrain probes or run language models.

## Verification

Run the model-free test suite with:

```bash
uv run pytest -q
```

Formatting, linting, and type checks are run with:

```bash
./lint.sh
```
