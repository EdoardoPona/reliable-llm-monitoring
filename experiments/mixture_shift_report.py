"""Summarise completed mixture experiments and draw empirical simplex maps.

Maps show one explicitly identified seed and fixed policy. All model seeds and
targets remain in the summary, including nonselection. Rendering uses the exact
linear mixture identity, not additional data or recalibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.tri import Triangulation
from mixture_shift import write_json

from reliable_monitoring.mixture_ltt import simplex_grid

METHOD_LABELS = {"pooled": "Pooled CTD", "pooled_matched": "Pooled, matched candidates", "mixture": "Domainwise CTD"}
DOMAIN_LABELS = {"anthropic": "Anthropic", "medical": "Medical", "toolace": "ToolACE"}
VERTICES = np.array([[0, 0], [0.5, np.sqrt(3) / 2], [1, 0]])


def read_run(directory: Path):
    inventory = json.loads((directory / "inventory.json").read_text())
    if inventory["status"] != "complete" or inventory["inventory_only"]:
        raise ValueError("Report requires a completed calibration run")
    mixtures = json.loads((directory / "mixtures.json").read_text())
    if len(mixtures["domains"]) != 3:
        raise ValueError("Simplex figures require three domains")
    trials = [json.loads(line) for line in (directory / "trials.jsonl").read_text().splitlines() if line.strip()]
    records = []
    for trial in trials:
        for result in trial["results"]:
            records.append(
                {**result, **{k: trial[k] for k in ("expert", "model_seed", "split_seed", "trial", "guaranteed_risk")}}
            )
    return mixtures, records, inventory["code_commit"]


def domain_metrics(record: dict, domains: list[str]):
    evaluation = record["evaluation"]
    if evaluation is None:
        return None
    risks = np.array([evaluation["domains"][g][record["guaranteed_risk"]] for g in domains], dtype=float)
    complementary = "budget" if record["guaranteed_risk"] == "accuracy_error" else "accuracy_error"
    utility = np.array([evaluation["domains"][g][complementary] for g in domains], dtype=float)
    if complementary == "accuracy_error":
        utility = 1 - utility
    # Positive margin means empirical accuracy/call rate meets its target.
    return record["alpha"] - risks, utility


def special_mixtures(mixtures: dict):
    return {
        **{DOMAIN_LABELS.get(g, g): q for g, q in zip(mixtures["domains"], np.eye(3), strict=True)},
        "Uniform": np.full(3, 1 / 3),
        "Original benchmark": np.array(mixtures["reference"]),
    }


def summarise(records: list[dict], mixtures: dict):
    summary = []
    keys = sorted({(r["expert"], r["alpha"], r["method"]) for r in records})
    for expert, alpha, method in keys:
        rows = [r for r in records if (r["expert"], r["alpha"], r["method"]) == (expert, alpha, method)]
        selected = [r for r in rows if r["selection"]["threshold"] is not None]
        points = {}
        for name, weights in special_mixtures(mixtures).items():
            points[name] = [
                {
                    "model_seed": r["model_seed"],
                    "split_seed": r["split_seed"],
                    "accuracy": 1
                    - weights
                    @ np.array([r["evaluation"]["domains"][g]["accuracy_error"] for g in mixtures["domains"]]),
                    "expert_call_rate": weights
                    @ np.array([r["evaluation"]["domains"][g]["budget"] for g in mixtures["domains"]]),
                }
                for r in selected
            ]
        summary.append(
            {
                "expert": expert,
                "alpha": alpha,
                "method": method,
                "n_trials": len(rows),
                "n_certified": len(selected),
                "unselected": [
                    {"model_seed": r["model_seed"], "split_seed": r["split_seed"]} for r in rows if r not in selected
                ],
                "per_selected_policy": [
                    {
                        "model_seed": r["model_seed"],
                        "split_seed": r["split_seed"],
                        "threshold": r["selection"]["threshold"],
                        "worst_domain_accuracy": 1 - r["evaluation"]["accuracy_error"]["worst_domain"],
                        "worst_domain_call_rate": r["evaluation"]["budget"]["worst_domain"],
                    }
                    for r in selected
                ],
                "named_mixtures": points,
            }
        )
    return summary


def _frame(ax, domains):
    closed = np.vstack([VERTICES, VERTICES[0]])
    ax.plot(*closed.T, color="#30343b", lw=1.1)
    for (x, y), name, offset, align in zip(
        VERTICES, domains, [(0, -12), (0, 9), (0, -12)], ["top", "bottom", "top"], strict=True
    ):
        ax.annotate(
            DOMAIN_LABELS.get(name, name), (x, y), xytext=offset, textcoords="offset points", ha="center", va=align
        )
    ax.set(xlim=(-0.14, 1.14), ylim=(-0.10, 1.01), aspect="equal")
    ax.axis("off")


def plot_simplex(records, mixtures, expert, alpha, seed, output: Path, *, include_matched=False):
    methods = ["pooled", "pooled_matched", "mixture"] if include_matched else ["pooled", "mixture"]
    target_rows = [r for r in records if r["expert"] == expert and np.isclose(r["alpha"], alpha)]
    chosen = {}
    for method in methods:
        rows = [r for r in target_rows if r["method"] == method and r["model_seed"] == seed]
        if len(rows) != 1:
            raise ValueError("Simplex report requires exactly one split per requested model seed")
        chosen[method] = rows[0]
    domains = mixtures["domains"]
    values = {m: domain_metrics(r, domains) for m, r in chosen.items()}
    accuracy_control = target_rows[0]["guaranteed_risk"] == "accuracy_error"
    kind = "accuracy" if accuracy_control else "budget"
    target = 1 - alpha if accuracy_control else alpha
    margins = [v[0] for v in values.values() if v is not None]
    span = max(1.0, np.ceil(100 * max((np.abs(v).max() for v in margins), default=0.05)))
    norms = [Normalize(-span, span), Normalize(0, 100)]
    maps = ["RdBu", "viridis"]
    # Fine rendering is an affine function of three measured domain risks.
    dense = simplex_grid(3, 60)
    xy = dense @ VERTICES
    mesh = Triangulation(xy[:, 0], xy[:, 1])
    evaluated_points = np.array(mixtures["weights"]) @ VERTICES
    fig, axes = plt.subplots(2, len(methods), figsize=(4.2 * len(methods) + 1, 7.3))
    fig.subplots_adjust(left=0.03, right=0.87, top=0.80, bottom=0.12, hspace=0.25, wspace=0.08)
    for col, method in enumerate(methods):
        for row in range(2):
            ax = axes[row, col]
            _frame(ax, domains)
            if values[method] is None:
                ax.fill(*VERTICES.T, color="#eceff1", zorder=-1)
                ax.text(0.5, 0.32, "No certified\nthreshold", ha="center", va="center", color="#61666d", fontsize=11)
            else:
                z = 100 * (dense @ values[method][row])
                ax.tricontourf(
                    mesh,
                    z,
                    levels=np.linspace(-span, span, 101) if row == 0 else np.linspace(0, 100, 101),
                    cmap=maps[row],
                    norm=norms[row],
                )
                if row == 0 and z.min() < 0 < z.max():
                    ax.tricontour(mesh, z, levels=[0], colors="#202020", linewidths=1.3)
                ax.scatter(*evaluated_points.T, s=4, c="#30343b", alpha=0.18, zorder=3)
                for weights, marker, size in [(np.full(3, 1 / 3), "o", 45), (mixtures["reference"], "*", 110)]:
                    ax.scatter(
                        *(np.asarray(weights) @ VERTICES),
                        marker=marker,
                        s=size,
                        c="white",
                        edgecolor="black",
                        linewidth=1,
                        zorder=4,
                    )
        axes[0, col].set_title(METHOD_LABELS[method], fontsize=12, fontweight="bold", pad=10)
    labels = [
        "Target margin (percentage points)\nPositive meets the empirical target",
        "Expert-call rate (%)" if accuracy_control else "Achieved accuracy (%)",
    ]
    for row, bottom in enumerate([0.505, 0.145]):
        cax = fig.add_axes((0.89, bottom, 0.017, 0.245))
        fig.colorbar(ScalarMappable(norm=norms[row], cmap=maps[row]), cax=cax).set_label(labels[row], fontsize=9)
    target_label = "Minimum accuracy" if accuracy_control else "Maximum expert-call rate"
    fig.suptitle(
        f"{expert.capitalize()} expert · {target_label} {100 * target:.1f}%\nFixed policy per panel · model/split seed {seed}",
        y=0.98,
        fontsize=14,
    )
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", label="Uniform"),
            Line2D(
                [],
                [],
                marker="*",
                color="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=11,
                label="Original benchmark",
            ),
            Line2D([], [], color="#202020", label="Target boundary (zero margin)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.48, 0.885),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    available = []
    for method in methods:
        rows = [r for r in target_rows if r["method"] == method]
        available.append(
            f"{METHOD_LABELS[method]}: {sum(r['selection']['threshold'] is not None for r in rows)}/{len(rows)}"
        )
    fig.text(0.45, 0.055, "Certified across model seeds — " + "; ".join(available), ha="center", fontsize=9)
    fig.text(
        0.45,
        0.025,
        "Held-out empirical measurements. Mixture weights change; thresholds stay fixed.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    stem = output / f"{kind}_{expert}_seed{seed}{'_matched' if include_matched else ''}"
    for extension in ("png", "svg"):
        fig.savefig(stem.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


def plot_availability(summary, risk_name, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True, layout="constrained")
    for ax, expert in zip(axes, ["strong", "weak"], strict=True):
        for method, marker in zip(METHOD_LABELS, ["o", "s", "^"], strict=True):
            rows = [r for r in summary if r["expert"] == expert and r["method"] == method]
            rows.sort(key=lambda r: -r["alpha"] if risk_name == "accuracy_error" else r["alpha"])
            targets = [100 * (1 - r["alpha"] if risk_name == "accuracy_error" else r["alpha"]) for r in rows]
            ax.plot(targets, [r["n_certified"] for r in rows], marker=marker, label=METHOD_LABELS[method], alpha=0.8)
        ax.set(
            title=f"{expert.capitalize()} expert",
            xlabel="Required accuracy (%)" if risk_name == "accuracy_error" else "Allowed expert-call rate (%)",
        )
        ax.set_ylim(-0.12, max(r["n_trials"] for r in summary) + 0.12)
        ax.set_yticks(range(max(r["n_trials"] for r in summary) + 1))
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Seeds with a certified threshold")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Certification availability · all targets, all three model seeds")
    for extension in ("png", "svg"):
        fig.savefig(output / f"certification_{risk_name}.{extension}", dpi=180)
    plt.close(fig)


def configured_target(records: list[dict], expert: str, requested: float) -> float:
    """Require calibration at the requested target instead of substituting a neighbour."""
    matches = sorted(
        {r["alpha"] for r in records if r["expert"] == expert and np.isclose(r["alpha"], requested, rtol=0, atol=1e-12)}
    )
    if len(matches) != 1:
        raise ValueError(
            f"No unique calibration at target {requested:g} for {expert}; calibrate this exact target first"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strong-accuracy", type=float, default=0.80)
    parser.add_argument("--weak-accuracy", type=float, default=0.78)
    parser.add_argument(
        "--budget", type=float, default=0.40, help="Exact calibrated budget target to plot (default: 0.40)"
    )
    args = parser.parse_args()
    mixtures, records, code_commit = read_run(args.run)
    summary = summarise(records, mixtures)
    risk_name = records[0]["guaranteed_risk"]
    targets = {}
    for expert in ("strong", "weak"):
        requested = 1 - getattr(args, expert + "_accuracy") if risk_name == "accuracy_error" else args.budget
        targets[expert] = configured_target(records, expert, requested)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for expert in ("strong", "weak"):
        for matched in (False, True):
            plot_simplex(
                records, mixtures, expert, targets[expert], args.seed, args.output_dir, include_matched=matched
            )
    plot_availability(summary, risk_name, args.output_dir)
    write_json(
        args.output_dir / "summary.json",
        {
            "run": str(args.run.resolve()),
            "calibration_commit": code_commit,
            "figure_model_seed": args.seed,
            "figure_error_or_budget_targets": targets,
            "guaranteed_risk": risk_name,
            "summary": summary,
        },
    )
    print(f"Wrote simplex figures and all-target summary to {args.output_dir}")


if __name__ == "__main__":
    main()
