# Initial distribution-mixture results

Completed on the lab server on 5 November 2026, using calibration commit `ffd313d`. Both primary configurations ran with the official attention/attention scores: strong/weak experts, model and split seeds 42/43/44, unchanged 1,586-row benchmark, and 68 deployment mixtures. The original model scores and data were read-only inputs. No activations, refitting or new model inference were required.

The grid contains 66 combinations in 10-percentage-point increments, including all three single-domain extremes, plus the uniform and original benchmark mixtures. Each calibrated threshold is fixed across mixtures. The runs contain 90 accuracy-control and 360 budget-control method/target/seed records. The 500-resplit stability analysis has not been run.

## Accuracy control: certification is the limitation

Domainwise calibration returned **no certified threshold at any configured accuracy target in any model seed**: strong targets 74–82%, weak targets 72–80%, five targets each. This is nonselection, not a measured risk violation by a selected domainwise policy. It does not establish that no alternative policy could meet these targets.

At the two predefined accuracy views:

| Expert / minimum accuracy | Method | Certified seeds | Accuracy at original mixture | Worst-domain accuracy |
|---|---|---:|---:|---:|
| Strong / 80.0% | pooled | 3/3 | 81.9%–84.9% | 61.8%–71.4% |
| Strong / 80.0% | mixture | 0/3 | — | — |
| Weak / 78.0% | pooled | 3/3 | 79.1%–84.0% | 68.3%–75.1% |
| Weak / 78.0% | mixture | 0/3 | — | — |

The ranges are the three observed seed results, not confidence intervals. Pooled calibration meets the accuracy target empirically at the original mixture in these views, while its fixed policies fall below the target in some domain mixtures. At the first domainwise candidate, the ToolACE p-value is 1 in all six predefined accuracy configurations; some weak-expert candidates also fail other domain tests. This identifies a calibration diagnostic to investigate before changing targets or candidate construction. Do not infer from nonselection alone that the medical sample count is the limiting factor.

## Budget control: robustness and its performance cost

The predefined budget view uses the configured target nearest 50%, with a lower-target tie break: **47.6316%** maximum expert calls.

| Expert | Method | Certified seeds | Worst-domain call rate | Accuracy at original mixture |
|---|---|---:|---:|---:|
| Strong | pooled | 3/3 | 49.3%–60.2% | 82.5%–85.2% |
| Strong | mixture | 3/3 | 38.9%–42.1% | 79.8%–83.0% |
| Weak | pooled | 3/3 | 36.7%–41.2% | 81.2%–84.4% |
| Weak | mixture | 3/3 | 36.7%–43.1% | 81.4%–84.4% |

For the strong expert, each pooled policy exceeds the call-rate target in at least one domain. Each domainwise policy stays below it in all three measured domains, and hence all of their empirical mixtures, at a cost in accuracy. For the weak expert, both methods stay within the empirical budget at this target; a benefit should not be claimed uniformly across experts and targets. The full summary retains the matched-candidate pooled comparison.

Five reference mixtures illustrate the strong-expert budget result for the **preselected seed 42**:

| Mixture | Pooled calls | Domainwise calls | Pooled accuracy | Domainwise accuracy |
|---|---:|---:|---:|---:|
| Anthropic | 49.2% | 38.9% | 90.2% | 86.1% |
| Medical | 49.3% | 32.3% | 89.1% | 88.8% |
| ToolACE | 20.4% | 11.4% | 69.8% | 67.8% |
| Uniform | 39.6% | 27.5% | 83.0% | 80.9% |
| Original benchmark | 40.2% | 27.8% | 83.4% | 81.3% |

## Figures and interpretation

Use a pair of simplex maps for target margin, with a second row for the complementary metric. Mark the uniform and original mixtures; corners identify the single-domain extremes. The budget figure shows where the pooled call-rate target is exceeded and the performance cost of domainwise calibration. The accuracy figure explicitly shows the lack of a certified domainwise threshold. All-target certification plots prevent the fixed-target figures from hiding nonselection.

These are held-out empirical measurements, not direct observations of population-guarantee violations. The 68 mixtures reuse the same evaluation examples and are not 68 independent test sets. The guarantee is over mixtures of fixed domain distributions; the medical MT/MTS composition remains fixed. Three model seeds describe observed variability, not a calibrated estimate of coverage probability.

Run directories under `results/mixture_shift/` (outputs are not committed):

- `accuracy_grid10_ffd313d/`: accuracy runs, inventory and raw trial results.
- `budget_grid10_ffd313d/`: budget runs, inventory and raw trial results.
- Each `report_seed42/` directory: simplex PNG/SVG figures, certification plots and all-target `summary.json`.

Reproduce the figures with `experiments/mixture_shift_report.py`; see the [experiment plan](mixture_shift_experiment_plan.md) for commands and metrics. The calibration target grid was unchanged after viewing these results.
