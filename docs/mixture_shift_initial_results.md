# Distribution-mixture results

Calibration commit: [`b000dcb`](https://github.com/EdoardoPona/reliable-llm-monitoring/commit/b000dcb9768361acc260cc17e1f0d0b1305d87f7).

Results cover three model seeds and 68 deployment mixtures, with each threshold fixed across mixtures. The central comparisons are control under shift, its accuracy/call-rate tradeoff, and the availability of a certified policy.

## Budget control at 40%

Domainwise calibration returns a certified threshold for **3/3 seeds for each expert**. All six policies have held-out worst-domain call rates below 40%. Pooled policies exceed 40% in all three strong-expert seeds and two of the three weak-expert seeds.

| Expert | Method | Certified seeds | Worst-domain call rate | Accuracy at original mixture |
|---|---|---:|---:|---:|
| Strong | Pooled | 3/3 | 44.6%–60.2% | 80.9%–85.2% |
| Strong | Domainwise | 3/3 | 34.3%–34.8% | 78.4%–81.5% |
| Weak | Pooled | 3/3 | 36.7%–41.2% | 81.2%–84.4% |
| Weak | Domainwise | 3/3 | 35.5%–37.7% | 81.2%–84.4% |

Ranges show observed variation across the three seeds. For the strong expert, the extra budget margin comes with lower accuracy at the original mixture. For the weak expert, the accuracy cost is smaller; seed 44 selects the same policy under both methods. The matched-candidate pooled control is retained in the full summary.

Five named mixtures show the strong-expert comparison for seed 42:

| Mixture | Pooled calls | Domainwise calls | Pooled accuracy | Domainwise accuracy |
|---|---:|---:|---:|---:|
| Anthropic | 49.2% | 34.8% | 90.2% | 85.7% |
| Medical | 49.3% | 23.5% | 89.1% | 87.8% |
| ToolACE | 20.4% | 8.6% | 69.8% | 66.3% |
| Uniform | 39.6% | 22.3% | 83.0% | 79.9% |
| Original benchmark | 40.2% | 22.4% | 83.4% | 80.3% |

## Accuracy control: certification is the limitation

Domainwise calibration returns **no certified threshold at any configured accuracy target in any seed**: strong targets 74–82%, weak targets 72–80%, five targets each. Nonselection remains part of the result; it does not establish that no alternative policy could meet these targets.

| Expert / minimum accuracy | Method | Certified seeds | Accuracy at original mixture | Worst-domain accuracy |
|---|---|---:|---:|---:|
| Strong / 80.0% | Pooled | 3/3 | 81.9%–84.9% | 61.8%–71.4% |
| Strong / 80.0% | Domainwise | 0/3 | — | — |
| Weak / 78.0% | Pooled | 3/3 | 79.1%–84.0% | 68.3%–75.1% |
| Weak / 78.0% | Domainwise | 0/3 | — | — |

Pooled policies meet these accuracy targets empirically at the original mixture but fall below them in some domain mixtures. At the first domainwise candidate, the ToolACE p-value is 1 in all six displayed configurations; some weak-expert candidates also fail other domain tests. Nonselection alone does not identify medical sample size as the limiting factor.

## Presentation and interpretation

The main presentation combines paired simplex maps, the five-mixture table, a worst-domain summary and certification availability across targets. The maps show target margin and complementary performance, mark uniform and original mixtures, and display nonselection explicitly. All accuracy targets and the original budget sweep plus exact 40% remain in the summaries.

The measurements are held-out empirical risks, not direct observations of population-guarantee violations. The 68 mixtures reuse the same evaluation examples; the medical MT/MTS composition stays fixed. The seed ranges are descriptive, not confidence intervals. The 500-resplit stability analysis remains outstanding.

See the [experiment plan](mixture_shift_experiment_plan.md) for the metric definitions and reproduction commands.
