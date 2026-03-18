"""Binary vs Continuous delegation value comparison.

Reads structured JSON results from dv_cascade_comparison runs and produces:
  1. A LaTeX table comparing binary and continuous DV at B=128 (both baselines)
  2. A line chart of Δ AUC (continuous − binary) vs budget, one line per batch size

Inputs are the `{prefix}results.json` files saved by dv_cascade_comparison.py.

Usage::

    uv run experiments/dv_binary_vs_continuous.py \
        --binary-strong  figures/results.json \
        --binary-weak    figures/llama1b_results.json \
        --cont-strong    figures/continuous_results.json \
        --cont-weak      figures/continuous_llama1b_results.json \
        --output-dir     figures/
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def _interp_at(budget_fractions: list[float], values: list[float], targets: list[float]) -> list[float]:
    """Interpolate values at target budget fractions."""
    return list(np.interp(targets, budget_fractions, values))


def make_table(
    binary: dict,
    continuous: dict,
    batch_size: int = 128,
    budget_targets: list[float] | None = None,
) -> str:
    """Build a LaTeX table comparing binary vs continuous at a given batch size."""
    if budget_targets is None:
        budget_targets = [0.10, 0.20, 0.30, 0.50]

    bs_key = str(batch_size)
    lines: list[str] = []

    signals = ["Probe uncertainty (top-k)", "DV probe (top-k)", "Oracle (top-k)"]

    lines.append(r"\begin{tabular}{l cc cc cc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c}{Uncertainty} & \multicolumn{2}{c}{DV probe} & \multicolumn{2}{c}{Oracle} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"Budget & AUC & Acc & AUC & Acc & AUC & Acc \\")
    lines.append(r"\midrule")

    for label, data in [
        (r"Binary $v(x)$", binary),
        (r"Continuous $v_c(x)$", continuous),
    ]:
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{label}}}}} \\")
        topk = data["topk"][bs_key]
        bf = topk["budget_fractions"]

        for target in budget_targets:
            cells = [f"{int(target * 100)}\\%"]
            for sig in signals:
                auc_val = np.interp(target, bf, topk["signals"][sig]["auc"])
                acc_val = np.interp(target, bf, topk["signals"][sig]["accuracy"])
                cells.append(f"{auc_val:.3f}"[1:])
                cells.append(f"{acc_val:.3f}"[1:])
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    # Remove last \midrule and replace with \bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def plot_delta_auc(
    binary: dict,
    continuous: dict,
    output_path: Path,
    signal: str = "DV probe (top-k)",
) -> plt.Figure:
    """Plot Δ AUC (continuous − binary) vs budget, one line per batch size."""
    batch_sizes = binary["config"]["batch_sizes"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = {32: "#1f77b4", 64: "#ff7f0e", 128: "#2ca02c"}
    markers = {32: "s", 64: "^", 128: "o"}

    for metric_idx, (metric, metric_label) in enumerate([("auc", "ROC AUC"), ("accuracy", "Accuracy")]):
        ax = axes[metric_idx]
        for bs in batch_sizes:
            bs_key = str(bs)
            bf_bin = np.array(binary["topk"][bs_key]["budget_fractions"])
            bf_cont = np.array(continuous["topk"][bs_key]["budget_fractions"])
            vals_bin = np.array(binary["topk"][bs_key]["signals"][signal][metric])
            vals_cont = np.array(continuous["topk"][bs_key]["signals"][signal][metric])

            # Interpolate to common grid
            common_bf = np.linspace(0.0, 1.0, 50)
            interp_bin = np.interp(common_bf, bf_bin, vals_bin)
            interp_cont = np.interp(common_bf, bf_cont, vals_cont)
            delta = interp_cont - interp_bin

            # Subsample for markers
            marker_idx = np.linspace(0, len(common_bf) - 1, 10, dtype=int)
            ax.plot(
                common_bf,
                delta,
                color=colors[bs],
                label=f"B={bs}",
                linewidth=1.5,
            )
            ax.plot(
                common_bf[marker_idx],
                delta[marker_idx],
                color=colors[bs],
                marker=markers[bs],
                linestyle="none",
                markersize=5,
            )

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Delegation budget (k/B)")
        ax.set_ylabel(f"Δ {metric_label} (continuous − binary)")
        ax.set_title(f"DV probe: Δ {metric_label}")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    logger.info(f"Delta plot saved to {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Binary vs Continuous DV comparison")
    parser.add_argument("--binary-strong", type=Path, required=True, help="Binary strong baseline results.json")
    parser.add_argument("--binary-weak", type=Path, required=True, help="Binary weak baseline results.json")
    parser.add_argument("--cont-strong", type=Path, required=True, help="Continuous strong baseline results.json")
    parser.add_argument("--cont-weak", type=Path, required=True, help="Continuous weak baseline results.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dv_comparison"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    binary_strong = load_results(args.binary_strong)
    binary_weak = load_results(args.binary_weak)
    cont_strong = load_results(args.cont_strong)
    cont_weak = load_results(args.cont_weak)

    # --- Table: B=128, both baselines ---
    logger.info("=== Strong baseline (Gemma-27B) ===")
    table_strong = make_table(binary_strong, cont_strong, batch_size=128)
    logger.info("\n" + table_strong)

    logger.info("\n=== Weak baseline (Llama-1B) ===")
    table_weak = make_table(binary_weak, cont_weak, batch_size=128)
    logger.info("\n" + table_weak)

    # Save tables
    (args.output_dir / "dv_comparison_table_strong.tex").write_text(table_strong)
    (args.output_dir / "dv_comparison_table_weak.tex").write_text(table_weak)
    logger.info("Tables saved.")

    # --- Delta AUC plots ---
    plot_delta_auc(
        binary_strong,
        cont_strong,
        args.output_dir / "dv_delta_strong.pdf",
    )
    plot_delta_auc(
        binary_weak,
        cont_weak,
        args.output_dir / "dv_delta_weak.pdf",
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
