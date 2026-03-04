# DV-LTT Cascade — Slide Scratch

---

## Slide 1: Problem Setup

**Two-stage safety cascade** with a budget constraint.

- **Probe** $\rho$: cheap safety classifier (logistic regression on Llama-3.2-1B activations, layer 11). Fast, runs on every input.
- **Baseline** $M$: expensive but stronger model (Gemma-3-27B). Can only be called on a fraction of inputs.

**Goal:** delegate selectively to the baseline to maximize cascade performance while respecting a budget constraint on the delegation rate.

**Key question:** *which* examples should be delegated?

---

## Slide 2: Delegation Value

Define per-example **delegation value**:

$$v(x) = \mathbf{1}[\text{probe wrong} \wedge \text{baseline correct}]$$

- $v=1$: delegation flips the outcome from wrong to right.
- $v=0$: delegation doesn't help (both correct, both wrong, or probe correct / baseline wrong).

Measured on test data (4 groups, 500 examples each):
- Overall $v=1$ rate: **19.7%**
- Varies by group: anthropic 31.4%, mt 10.4%, mts 23.3%, toolace 16.6%
- "Both correct" dominates for mt (~90%); toolace has the most "both wrong" cases

---

## Slide 3: DV Probe

**Idea:** train a classifier to predict $v(x)$ from the *same activations* $z(x)$ used by the safety probe. The activations encode richer information than the scalar probe score — they can distinguish subpopulations where the baseline is strong vs weak.

**DV probe:** logistic regression on mean-pooled activations $z(x)$, predicting $P(v=1 \mid z)$.

- Trained on **dev split** (separate from test evaluation).
- Same feature space as the safety probe, different target.

**Results (out-of-sample, test set):**
- Overall AUC: **0.801**
- Per-group: anthropic 0.830, mt 0.771, mts 0.811, toolace 0.679

The DV probe meaningfully separates examples where delegation would help from those where it wouldn't.

---

## Slide 4: DV Probe vs Uncertainty — Budget Sweep

Compare delegation strategies by ranking examples and delegating the top-$k$ at varying budget fractions:

1. **Uncertainty:** rank by $\min(p, 1-p)$ — delegate the most uncertain probe predictions.
2. **DV probe:** rank by $d_z(x) = P(v=1 \mid z)$ — delegate where the model predicts delegation helps.
3. **Oracle:** rank by true $v(x)$ — upper bound (capped at $v=1$ rate).

**Key finding (AUC):** DV probe dominates uncertainty at every budget level. At 14% budget (the natural DV threshold operating point), DV achieves cascade AUC 0.869 vs ~0.80 for uncertainty at the same budget. Uncertainty needs ~60% budget to match.

**Key finding (Accuracy):** DV probe also dominates on accuracy. Oracle line shows perfect accuracy improvement with budget (by construction), confirming the DV probe captures genuine accuracy-relevant signal.

Reference points: probe-only AUC 0.797, baseline-only AUC 0.933.

---

## Slide 5: The LTT Experiment — Method

Previous slide used "delegate top-$k$" with a known budget. In practice, we need a **threshold** $\tau$ with a **PAC budget guarantee**.

**Learn Then Test (LTT)** procedure:
1. Split test data into **calibration** (50%) and **evaluation** (50%).
2. Define budget constraint: delegation rate $\leq \alpha_\text{budget}$.
3. Grid of candidate thresholds $\tau \in [0, 1]$ (200 steps).
4. For each $\alpha_\text{budget}$, use **fixed-sequence testing** with binomial p-value bounds on the calibration set to find the smallest valid $\tau$ (most aggressive threshold that still satisfies the budget guarantee at confidence $1-\delta = 0.9$).
5. Evaluate on held-out eval set: apply the DV threshold cascade ($d_z(x) > \tau \Rightarrow$ delegate).

**Baseline:** batched top-$k$ by uncertainty at the same $\alpha_\text{budget}$ (batch size 128, the realistic operational setting). Fixed-$k$ uses its full budget allocation every batch.

---

## Slide 6: Budget Control

LTT successfully controls the delegation rate.

- Across 10 $\alpha_\text{budget}$ levels from 5% to 50%, the realized DV delegation rate tracks the constraint closely.
- Realized rate stays at or slightly above $\alpha_\text{budget}$ — the guarantee holds (valid region).
- The DV threshold is **adaptive**: it uses less budget when the data doesn't need it, more when it does. Fixed-$k$ always uses its full allocation.

---

## Slide 7: Performance — DV Threshold vs Fixed-$k$ Uncertainty

At every budget level from 5% to 50%:

**AUC:**
- DV threshold consistently outperforms fixed-$k$ uncertainty.
- Gap is largest at low budgets (~5-15%): DV achieves ~0.83-0.85 AUC while uncertainty stays near probe-only (0.797).
- Even at high budgets (50%), DV maintains a ~0.01 edge.
- Probe-only AUC: 0.797. Baseline-only AUC: 0.933.

**Accuracy:**
- DV threshold outperforms fixed-$k$ uncertainty across the board.
- At low budgets the gap is striking: DV reaches ~0.78-0.80 accuracy while uncertainty stays near probe-only (0.70).
- Probe-only accuracy: 0.70. Baseline-only accuracy: 0.84.

The DV probe makes better use of the budget because it targets examples where delegation actually flips the outcome, rather than merely uncertain examples.

---

## Slide 8: Adaptivity Across Groups

At $\alpha_\text{budget} = 30\%$, per-group delegation rates reveal how the DV threshold adapts:

| Group | DV rate | Fixed-$k$ rate | $v=1$ rate |
|-------|---------|----------------|------------|
| anthropic | ~47% | ~19% | 31.4% |
| mt | ~15% | ~21% | 10.4% |
| mts | ~38% | ~28% | 23.3% |
| toolace | ~30% | ~30% | 16.6% |

- DV allocates **more** budget to anthropic and mts (where delegation helps most — high $v=1$ rates) and **less** to mt (where delegation rarely helps — low $v=1$ rate).
- Fixed-$k$ uncertainty spreads budget roughly uniformly regardless of where it's useful.
- The DV threshold achieves group-aware budget allocation **without using group labels** — the activations encode subpopulation identity implicitly.

---

## Slide 9: Summary

1. **DV probe works:** logistic regression on activations predicts delegation value with AUC 0.80 (out-of-sample). The signal is there in the activation space.

2. **Better than uncertainty:** DV-ranked delegation dominates uncertainty-ranked delegation at every budget level, for both AUC and accuracy. The gap is largest at low budgets where efficiency matters most.

3. **LTT gives valid guarantees:** fixed-sequence testing with binomial bounds successfully controls the delegation rate. Realized budget tracks the constraint.

4. **Adaptive without group labels:** the DV threshold naturally allocates more budget to subpopulations where delegation helps and less where it doesn't, using only the activation vector.

5. **Practical:** the DV probe is the same architecture as the safety probe (logistic regression on mean-pooled activations). No additional model needed — just a second set of weights on the same features.
