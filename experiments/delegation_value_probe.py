"""Prototype: Train a delegation value (DV) probe.

Given a config specifying activations model, baseline model, and datasets:

1. Trains a safety probe on training data.
2. Fetches cached baseline scores via the baseline registry.
3. Computes delegation value v(x) = 1[probe wrong AND baseline correct].
4. Trains a DV probe on the dev split to predict v from activations.
5. Evaluates on the test split: compares delegation strategies across budgets.
6. Generates figures and optionally logs to ClearML.

See ``experiments/notes/delegation_value.md`` for theory.

Usage::

    uv run experiments/delegation_value_probe.py --config configs/delegation_value_probe.yaml
    uv run experiments/delegation_value_probe.py --config configs/delegation_value_probe.yaml --use-clearml
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from activation_registry import compute_or_fetch_activations
from config import load_config
from dotenv import load_dotenv
from mixed_dataset import fetch_per_source_activations, fetch_per_source_baselines, load_mixed_dataset_with_baselines
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from reliable_monitoring.cascade import probe_uncertainty
from reliable_monitoring.dataset import ActivationConfig, load_dataset
from reliable_monitoring.probes import SequenceProbe

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_delegation_value(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-example delegation value: 1 if delegation flips wrong to right, else 0.

    v(x) = 1 when the probe is incorrect and the baseline is correct.
    All other cases (both correct, both wrong, probe correct baseline wrong) are 0.
    """
    probe_correct = (probe_scores >= 0.5).astype(int) == labels
    baseline_correct = (baseline_scores >= 0.5).astype(int) == labels
    return (~probe_correct & baseline_correct).astype(int)


def compute_continuous_delegation_value(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Continuous delegation value: improvement in probability of the correct class.

    v(x) = P_baseline(y|x) - P_probe(y|x) = (2y - 1)(b(x) - p(x)).

    Positive when delegation improves the score toward the correct label.
    """
    sign = 2 * labels - 1  # +1 for y=1, -1 for y=0
    return sign * (baseline_scores - probe_scores)


def cascade_scores_at_budget(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    ranking_scores: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Return final scores after delegating top-``budget`` fraction by ``ranking_scores``."""
    n = len(probe_scores)
    k = int(budget * n)
    if k >= n:
        return baseline_scores.copy()
    final = probe_scores.copy()
    if k > 0:
        order = np.argsort(-ranking_scores)
        delegate_mask = np.zeros(n, dtype=bool)
        delegate_mask[order[:k]] = True
        final[delegate_mask] = baseline_scores[delegate_mask]
    return final


def cascade_metrics_at_budget(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    ranking_scores: np.ndarray,
    budget: float,
) -> tuple[float, float]:
    """Return (AUC, accuracy) after delegating top-``budget`` fraction."""
    final = cascade_scores_at_budget(probe_scores, baseline_scores, ranking_scores, budget)
    auc = float(roc_auc_score(labels, final))
    acc = float(accuracy_score(labels, (final >= 0.5).astype(int)))
    return auc, acc


def cascade_at_dv_threshold(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    dv_scores: np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Cascade metrics when delegating all examples with DV score >= threshold.

    Returns:
        (budget_used, cascade_auc, cascade_accuracy).
    """
    delegate_mask = dv_scores >= threshold
    budget = float(delegate_mask.mean())
    final_scores = probe_scores.copy()
    final_scores[delegate_mask] = baseline_scores[delegate_mask]
    auc = float(roc_auc_score(labels, final_scores))
    acc = float(accuracy_score(labels, (final_scores >= 0.5).astype(int)))
    return budget, auc, acc


def train_dv_probe(
    X_train: np.ndarray,
    v_train: np.ndarray,
    mode: str = "binary",
) -> LogisticRegression | Ridge:
    """Train delegation value probe on dev data.

    Args:
        mode: "binary" for LogisticRegression on binary v,
              "continuous" for Ridge regression on continuous v.
    """
    if mode == "binary":
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, v_train)
        train_scores = clf.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(v_train, train_scores)
        logger.info(f"  Train AUC: {train_auc:.4f}")
    elif mode == "continuous":
        clf = Ridge(alpha=1.0)
        clf.fit(X_train, v_train)
        train_scores = clf.predict(X_train)
        # Evaluate against binarized target for comparability
        v_binary = (v_train > 0).astype(int)
        if len(np.unique(v_binary)) > 1:
            train_auc = roc_auc_score(v_binary, train_scores)
            logger.info(f"  Train AUC (vs v>0): {train_auc:.4f}")
        from scipy.stats import spearmanr

        rho, _ = spearmanr(v_train, train_scores)
        logger.info(f"  Train Spearman rho: {rho:.4f}")
    else:
        raise ValueError(f"Unknown DV target mode: {mode}")
    return clf


def predict_dv_scores(
    clf: LogisticRegression | Ridge,
    X: np.ndarray,
    mode: str = "binary",
) -> np.ndarray:
    """Get delegation scores from a trained DV probe."""
    if mode == "binary" and isinstance(clf, LogisticRegression):
        return clf.predict_proba(X)[:, 1]
    return clf.predict(X)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_delegation_value_rates(
    v: np.ndarray,
    groups: np.ndarray | None,
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> plt.Figure:
    """Bar chart of delegation value rate (v=1) and error breakdown per group."""
    probe_correct = (probe_scores >= 0.5).astype(int) == labels
    baseline_correct = (baseline_scores >= 0.5).astype(int) == labels

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: v=1 rate per group
    ax = axes[0]
    if groups is not None:
        unique_groups = list(np.unique(groups))
        rates = [float(v[groups == g].mean()) for g in unique_groups]
        bars = ax.bar(unique_groups, rates, alpha=0.7, edgecolor="black")
        for bar, rate in zip(bars, rates, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax.bar(["all"], [float(v.mean())], alpha=0.7, edgecolor="black")
    ax.set_ylabel("Rate")
    ax.set_title(f"v=1 rate (overall: {v.mean():.1%})")
    ax.set_ylim(0, 1)

    # Right: error breakdown per group (stacked bar)
    ax = axes[1]
    if groups is not None:
        unique_groups = list(np.unique(groups))
        both_correct = [
            float((probe_correct[groups == g] & baseline_correct[groups == g]).mean()) for g in unique_groups
        ]
        v1 = [float(v[groups == g].mean()) for g in unique_groups]  # probe wrong, baseline correct
        probe_only = [
            float((probe_correct[groups == g] & ~baseline_correct[groups == g]).mean()) for g in unique_groups
        ]
        both_wrong = [
            float((~probe_correct[groups == g] & ~baseline_correct[groups == g]).mean()) for g in unique_groups
        ]

        x = np.arange(len(unique_groups))
        w = 0.6
        ax.bar(x, both_correct, w, label="Both correct", color="C2", alpha=0.7)
        ax.bar(x, v1, w, bottom=both_correct, label="v=1 (delegation helps)", color="C0", alpha=0.7)
        bottoms = [a + b for a, b in zip(both_correct, v1, strict=True)]
        ax.bar(x, probe_only, w, bottom=bottoms, label="Probe correct only", color="C1", alpha=0.7)
        bottoms2 = [a + b for a, b in zip(bottoms, probe_only, strict=True)]
        ax.bar(x, both_wrong, w, bottom=bottoms2, label="Both wrong", color="C3", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(unique_groups)
        ax.legend(fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_title("Prediction outcome breakdown")

    fig.tight_layout()
    fig.savefig(output_dir / "delegation_value_rates.pdf", bbox_inches="tight")
    return fig


def plot_dv_probe_roc(
    v: np.ndarray,
    dv_scores: np.ndarray,
    groups: np.ndarray | None,
    output_dir: Path,
) -> plt.Figure:
    """ROC curve for DV probe predicting positive delegation value."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    fpr, tpr, _ = roc_curve(v, dv_scores)
    auc = roc_auc_score(v, dv_scores)
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("DV probe ROC: predicting v(x) > 0")
    ax.legend()

    ax = axes[1]
    if groups is not None:
        unique_groups = np.unique(groups)
        group_aucs: dict[str, float] = {}
        for g in unique_groups:
            mask = groups == g
            if len(np.unique(v[mask])) < 2:
                group_aucs[g] = float("nan")
            else:
                group_aucs[g] = roc_auc_score(v[mask], dv_scores[mask])
        bars = ax.bar(group_aucs.keys(), group_aucs.values(), alpha=0.7, edgecolor="black")
        for bar, val in zip(bars, group_aucs.values(), strict=True):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_ylabel("ROC AUC")
        ax.set_title("Per-group DV prediction")
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No group labels", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(output_dir / "dv_probe_roc.pdf", bbox_inches="tight")
    return fig


def plot_cascade_budget_sweep(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    dv_scores: np.ndarray,
    v: np.ndarray,
    dv_threshold_point: tuple[float, float, float],
    output_dir: Path,
) -> plt.Figure:
    """Cascade AUC and accuracy vs delegation budget for different strategies."""
    budgets = np.linspace(0, 1, 21)
    uncertainty = probe_uncertainty(probe_scores)

    # Compute both metrics for each strategy
    strategies = [("Uncertainty", uncertainty), ("DV probe", dv_scores)]
    strategy_metrics: dict[str, tuple[list[float], list[float]]] = {}
    for name, scores in strategies:
        aucs, accs = [], []
        for b in budgets:
            a, c = cascade_metrics_at_budget(probe_scores, baseline_scores, labels, scores, b)
            aucs.append(a)
            accs.append(c)
        strategy_metrics[name] = (aucs, accs)

    oracle_aucs, oracle_accs = [], []
    for b in budgets:
        a, c = cascade_metrics_at_budget(probe_scores, baseline_scores, labels, v, b)
        oracle_aucs.append(a)
        oracle_accs.append(c)

    # Reference values
    probe_auc = roc_auc_score(labels, probe_scores)
    baseline_auc = roc_auc_score(labels, baseline_scores)
    probe_acc = accuracy_score(labels, (probe_scores >= 0.5).astype(int))
    baseline_acc = accuracy_score(labels, (baseline_scores >= 0.5).astype(int))

    dt_budget, dt_auc, dt_acc = dv_threshold_point

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

    for row, (metric_name, ref_probe, ref_base, dt_val, oracle_vals) in enumerate(
        [
            ("ROC AUC", probe_auc, baseline_auc, dt_auc, oracle_aucs),
            ("Accuracy", probe_acc, baseline_acc, dt_acc, oracle_accs),
        ]
    ):
        ax = axes[row]
        metric_idx = row  # 0=auc, 1=acc

        for name, _ in strategies:
            vals = strategy_metrics[name][metric_idx]
            ax.plot(budgets, vals, label=name, marker="o", markersize=3)

        ax.plot(budgets, oracle_vals, label="Oracle", marker="o", markersize=3, linestyle="--", alpha=0.6)

        # DV threshold star
        ax.plot(dt_budget, dt_val, "*", color="C1", markersize=15, zorder=5)
        ax.annotate(
            f"DV threshold\n({dt_budget:.0%}, {dt_val:.3f})",
            xy=(dt_budget, dt_val),
            xytext=(dt_budget + 0.08, dt_val - 0.02),
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "C1"},
            color="C1",
        )

        ax.axhline(ref_probe, color="gray", linestyle=":", alpha=0.5, label=f"Probe only ({ref_probe:.3f})")
        ax.axhline(ref_base, color="gray", linestyle="--", alpha=0.5, label=f"Baseline only ({ref_base:.3f})")
        ax.set_ylabel(f"Cascade {metric_name}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_title("Cascade performance vs delegation budget")
    axes[1].set_xlabel("Delegation budget (fraction)")

    fig.tight_layout()
    fig.savefig(output_dir / "cascade_budget_sweep.pdf", bbox_inches="tight")
    return fig


def plot_per_group_budget_sweep(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    dv_scores: np.ndarray,
    v: np.ndarray,
    output_dir: Path,
) -> plt.Figure | None:
    """Per-group cascade AUC and accuracy vs budget for uncertainty, DV, and oracle."""
    if groups is None:
        return None

    unique_groups = np.unique(groups)
    budgets = np.linspace(0, 1, 21)
    uncertainty = probe_uncertainty(probe_scores)

    n_groups = len(unique_groups)
    metric_names = ["ROC AUC", "Accuracy"]
    fig, axes = plt.subplots(2, n_groups, figsize=(5 * n_groups, 7), squeeze=False)

    for i, g in enumerate(unique_groups):
        mask = groups == g
        ps, bs, lb = probe_scores[mask], baseline_scores[mask], labels[mask]
        ds = dv_scores[mask]
        g_v = v[mask]

        # Compute metrics for each strategy
        strategy_metrics: dict[str, tuple[list[float], list[float]]] = {}
        for name, scores in [("Uncertainty", uncertainty[mask]), ("DV probe", ds)]:
            aucs, accs = [], []
            for b in budgets:
                try:
                    a, c = cascade_metrics_at_budget(ps, bs, lb, scores, b)
                except ValueError:
                    a, c = float("nan"), float("nan")
                aucs.append(a)
                accs.append(c)
            strategy_metrics[name] = (aucs, accs)

        oracle_aucs, oracle_accs = [], []
        for b in budgets:
            try:
                a, c = cascade_metrics_at_budget(ps, bs, lb, g_v, b)
            except ValueError:
                a, c = float("nan"), float("nan")
            oracle_aucs.append(a)
            oracle_accs.append(c)

        # DV threshold operating point
        try:
            g_budget, g_dt_auc, g_dt_acc = cascade_at_dv_threshold(ps, bs, lb, ds)
        except ValueError:
            g_budget, g_dt_auc, g_dt_acc = None, None, None

        oracle_vals_list = [oracle_aucs, oracle_accs]
        dt_vals = [g_dt_auc, g_dt_acc]

        for row, metric_name in enumerate(metric_names):
            ax = axes[row, i]
            metric_idx = row

            for name in strategy_metrics:
                ax.plot(budgets, strategy_metrics[name][metric_idx], label=name)

            ax.plot(budgets, oracle_vals_list[row], label="Oracle", linestyle="--", alpha=0.6)

            if g_budget is not None and dt_vals[row] is not None:
                ax.plot(g_budget, dt_vals[row], "*", color="C1", markersize=12, zorder=5)

            ax.set_xlabel("Budget")
            ax.set_ylabel(metric_name)
            if row == 0:
                ax.set_title(f"{g} (n={mask.sum()})")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle("Per-group: cascade performance vs delegation budget")
    fig.tight_layout()
    fig.savefig(output_dir / "per_group_budget_sweep.pdf", bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_split(
    split: str,
    sources: list[dict],
    baseline_model: str,
    activation_config: ActivationConfig,
    balance: str | int,
    seed: int,
    safety_probe: SequenceProbe,
    *,
    local: bool = True,
    gpu: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a split, returning (probe_scores, baseline_scores, labels, activations, groups)."""
    strategy = activation_config.aggregation_strategy
    reduction = strategy if isinstance(strategy, str) else "mean"

    logger.info(f"Fetching cached baselines for {split}...")
    per_source_bl = fetch_per_source_baselines(sources, split, baseline_model, local=local, gpu=gpu)

    logger.info(f"Fetching cached activations for {split}...")
    per_source_acts = fetch_per_source_activations(
        sources,
        split,
        activation_config.model_name,
        activation_config.layer,
        reduction,
        local=local,
        gpu=gpu,
    )

    logger.info(f"Loading {split} datasets with cached activations and baselines...")
    activation_field = f"activations_{reduction}"
    dataset, baseline_scores = load_mixed_dataset_with_baselines(
        sources,
        split,
        per_source_bl,
        activation_config=None,
        balance_strategy=balance,
        seed=seed,
        per_source_activations=per_source_acts,
        activation_field_name=activation_field,
    )

    logger.info(f"Computing safety probe scores on {split}...")
    probe_scores = safety_probe.predict(dataset)
    labels = dataset.labels_numpy()
    groups = np.array(dataset.other_fields["group"]) if "group" in dataset.other_fields else None

    X = dataset.other_fields[activation_field]
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    X = np.asarray(X)

    logger.info(f"  {split}: n={len(labels)}, activations={X.shape}")
    if groups is not None:
        for g in np.unique(groups):
            logger.info(f"    {g}: n={int((groups == g).sum())}")

    return probe_scores, baseline_scores, labels, X, groups


def parse_args():
    parser = argparse.ArgumentParser(description="Delegation value probe")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/delegation_value_probe.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/delegation_value_probe",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--use-clearml",
        action="store_true",
        help="Log results to ClearML.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    seed = config.seed

    # --- ClearML init (early, so the task is visible while running) ---
    clearml_logger = None
    if args.use_clearml:
        import os

        from clearml_logger import ClearMLLogger

        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="delegation_value_probe",
            enabled=True,
        )
        clearml_logger.add_tags(["delegation-value-probe"])
        clearml_logger.connect_configuration(
            {
                "config_path": args.config,
                "baseline_model": config.baseline_model_name,
                "activations_model": config.activations_model_name,
                "activations_layer": config.activations_layer,
            }
        )

    activation_config = ActivationConfig(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
    )
    mixed_cfg = config.mixed_datasets
    sources = mixed_cfg["sources"]
    balance = mixed_cfg.get("balance_strategy", "min_size")
    use_modal = getattr(config, "use_modal", False)
    modal_gpu = getattr(config, "modal_gpu", None)

    # --- Train safety probe ---
    logger.info("Loading training data and fitting safety probe...")
    train_dataset = load_dataset(Path(config.train_dataset_path), activation_config=None)
    reduction = config.reduction_strategy
    train_acts = compute_or_fetch_activations(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
        reduction=reduction,
        dataset=train_dataset,
        dataset_path=config.train_dataset_path,
        local=not use_modal,
        gpu=modal_gpu,
    )
    train_dataset = train_dataset.assign(**{f"activations_{reduction}": train_acts})
    safety_probe = SequenceProbe(reduction_strategy=reduction)
    safety_probe.fit(train_dataset)
    del train_dataset, train_acts

    # --- Load dev split (DV probe training) ---
    dev_probe_scores, dev_baseline_scores, dev_labels, X_dev, dev_groups = _load_split(
        "dev",
        sources,
        config.baseline_model_name,
        activation_config,
        balance,
        seed,
        safety_probe,
        local=not use_modal,
        gpu=modal_gpu,
    )
    dv_target = getattr(config, "dv_target", "binary")
    logger.info(f"DV target mode: {dv_target}")

    v_dev_binary = compute_delegation_value(dev_probe_scores, dev_baseline_scores, dev_labels)
    if dv_target == "continuous":
        v_dev = compute_continuous_delegation_value(dev_probe_scores, dev_baseline_scores, dev_labels)
        logger.info(
            f"Dev delegation value: v>0 rate={float((v_dev > 0).mean()):.1%}, "
            f"mean={float(v_dev.mean()):.3f}, std={float(v_dev.std()):.3f} (n={len(v_dev)})"
        )
    else:
        v_dev = v_dev_binary.astype(float)
        logger.info(f"Dev delegation value: v=1 rate={v_dev.mean():.1%} (n={len(v_dev)})")

    # --- Train DV probe on dev ---
    logger.info("Training DV probe on dev split...")
    dv_clf = train_dv_probe(X_dev, v_dev, mode=dv_target)
    del X_dev  # free memory

    # --- Load test split (evaluation) ---
    probe_scores, baseline_scores, labels, X_test, groups = _load_split(
        "test",
        sources,
        config.baseline_model_name,
        activation_config,
        balance,
        seed,
        safety_probe,
        local=not use_modal,
        gpu=modal_gpu,
    )
    v_binary = compute_delegation_value(probe_scores, baseline_scores, labels)
    if dv_target == "continuous":
        v = compute_continuous_delegation_value(probe_scores, baseline_scores, labels)
        logger.info(
            f"Test delegation value: v>0 rate={float((v > 0).mean()):.1%}, "
            f"mean={float(v.mean()):.3f}, std={float(v.std()):.3f} (n={len(v)})"
        )
    else:
        v = v_binary.astype(float)
        logger.info(f"Test delegation value: v=1 rate={v.mean():.1%} (n={len(v)})")
    if groups is not None:
        for g in np.unique(groups):
            mask = groups == g
            if dv_target == "continuous":
                logger.info(f"  {g}: v>0 rate={float((v[mask] > 0).mean()):.1%}, mean={float(v[mask].mean()):.3f}")
            else:
                logger.info(f"  {g}: v=1 rate={v[mask].mean():.1%}")

    # --- Predict DV scores on test ---
    dv_scores = predict_dv_scores(dv_clf, X_test, mode=dv_target)
    del X_test

    # Evaluate DV probe quality against binarized target (for comparability)
    dv_auc = roc_auc_score(v_binary, dv_scores)
    dv_acc = accuracy_score(v_binary, (dv_scores >= 0.5 if dv_target == "binary" else dv_scores > 0).astype(int))
    logger.info(f"DV probe (test): AUC={dv_auc:.4f}, Acc={dv_acc:.4f}")

    # --- DV threshold operating point ---
    dv_threshold = 0.0 if dv_target == "continuous" else 0.5
    dt_budget, dt_auc, dt_acc = cascade_at_dv_threshold(
        probe_scores, baseline_scores, labels, dv_scores, threshold=dv_threshold
    )
    logger.info(
        f"DV threshold (>= {dv_threshold}): budget={dt_budget:.1%}, cascade AUC={dt_auc:.4f}, accuracy={dt_acc:.4f}"
    )
    if groups is not None:
        for g in np.unique(groups):
            mask = groups == g
            g_budget, g_auc, g_acc = cascade_at_dv_threshold(
                probe_scores[mask], baseline_scores[mask], labels[mask], dv_scores[mask], threshold=dv_threshold
            )
            logger.info(f"  {g}: budget={g_budget:.1%}, AUC={g_auc:.4f}, accuracy={g_acc:.4f}")

    # --- Budget sweep summary ---
    budget_levels = [0.1, 0.2, 0.3, 0.5]
    uncertainty = probe_uncertainty(probe_scores)

    logger.info("Cascade metrics at key budgets:")
    logger.info(
        f"  {'Budget':>6}  {'Unc AUC':>9}  {'DV AUC':>9}  {'Orc AUC':>9}  {'Unc Acc':>9}  {'DV Acc':>9}  {'Orc Acc':>9}"
    )
    for b in budget_levels:
        unc_auc, unc_acc = cascade_metrics_at_budget(probe_scores, baseline_scores, labels, uncertainty, b)
        dv_b_auc, dv_b_acc = cascade_metrics_at_budget(probe_scores, baseline_scores, labels, dv_scores, b)
        orc_auc, orc_acc = cascade_metrics_at_budget(probe_scores, baseline_scores, labels, v, b)
        logger.info(
            f"  {b:>6.0%}  {unc_auc:>9.4f}  {dv_b_auc:>9.4f}  {orc_auc:>9.4f}  {unc_acc:>9.4f}  {dv_b_acc:>9.4f}  {orc_acc:>9.4f}"
        )

    # --- Generate figures ---
    logger.info("Generating figures...")
    figs: dict[str, plt.Figure | None] = {}
    figs["delegation_value"] = plot_delegation_value_rates(
        v_binary, groups, probe_scores, baseline_scores, labels, output_dir
    )
    figs["dv_roc"] = plot_dv_probe_roc(v_binary, dv_scores, groups, output_dir)
    figs["budget_sweep"] = plot_cascade_budget_sweep(
        probe_scores, baseline_scores, labels, dv_scores, v, (dt_budget, dt_auc, dt_acc), output_dir
    )
    figs["per_group_sweep"] = plot_per_group_budget_sweep(
        probe_scores, baseline_scores, labels, groups, dv_scores, v, output_dir
    )
    plt.close("all")
    logger.info(f"Figures saved to {output_dir}")

    # --- ClearML results ---
    if clearml_logger is not None:
        clearml_logger.log_scalars(
            {
                "dv_probe_auc": dv_auc,
                "dv_probe_accuracy": dv_acc,
                "delegation_rate": float(v.mean()),
                "dv_threshold_budget": dt_budget,
                "dv_threshold_cascade_auc": dt_auc,
                "dv_threshold_cascade_accuracy": dt_acc,
            }
        )
        for name, fig in figs.items():
            if fig is not None:
                clearml_logger.log_figure("DV Probe", name, fig)
        clearml_logger.finalize()
        logger.info("Results logged to ClearML.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
