# Adaptive Rate Selection: Options Analysis

## The Problem

We have a probe that produces scores $p_i \in [0,1]$ for each example $i$. We define uncertainty as $u_i = \min(p_i, 1 - p_i)$. Our cascade sends some examples to an expensive baseline.

Currently we have two strategies:

**Fixed threshold** (our adaptive method): Send example $i$ to baseline iff $u_i > \tau$ for a fixed threshold $\tau = 1 - \text{threshold}$. The per-batch budget varies because different batches have different fractions of examples above $\tau$.

**Fixed budget rate** (our baseline): Sort examples by $u_i$, send the top $r$ fraction (most uncertain) to baseline. Budget is constant at $r$ per batch.

### Key observation

Since both methods rank examples by the same signal (probe uncertainty $u_i$), and since the threshold method selects exactly the examples with $u_i > \tau$, which are the top-$N$ most uncertain — **the fixed threshold and "top-N by percentile" select identical examples**. The threshold is just a percentile selection with a data-dependent rate.

Therefore: any scheme of the form "compute a per-batch rate, then select that many most-uncertain examples" is equivalent to a scheme of the form "apply a per-batch threshold." **You cannot improve per-example selection quality while using the same ranking signal.** The only degree of freedom is _how many_ to select per batch.

This means the current comparison is really: **variable rate (set by absolute threshold) vs constant rate** — and we showed the constant rate wins because the batch-to-batch variation in difficulty is too small for the variable rate to help, while the variable rate under-escalates on easy batches where even the "less uncertain" examples benefit from baseline.

## What would actually be different?

To improve on fixed-rate, we need one of:

1. A **different ranking signal** for per-example selection (not just probe uncertainty)
2. Conditions where **batch difficulty genuinely varies** enough for rate adaptation to matter
3. Both

Below we analyze options for (1). Option (2) is the stratified batching experiment (handled separately).

---

## Option 1: Learned Escalation Signal

### Idea

Train a binary classifier $g(x)$ that predicts: "will the baseline produce a better prediction than the probe for example $x$?" Use $g(x)$ instead of $u_i$ for the per-example ranking.

### Formalization

Let $\ell_i$ be the true label, $\hat{y}_i^P = \mathbb{1}[p_i \geq 0.5]$ the probe prediction, and $\hat{y}_i^B = \mathbb{1}[b_i \geq 0.5]$ the baseline prediction. Define:

$$z_i = \mathbb{1}[\hat{y}_i^B = \ell_i \text{ and } \hat{y}_i^P \neq \ell_i]$$

i.e., $z_i = 1$ when escalation would correct a probe error. On calibration data where we have both $p_i$ and $b_i$, fit a classifier $g(x_i) \approx P(z_i = 1 \mid x_i)$ using features derived from the probe (e.g., probe score $p_i$, probe activations, or other input features).

At test time, rank examples by $g(x_i)$ instead of $u_i$ and send the top-$r$ fraction.

### Assessment

- **Pro:** Uses a fundamentally different ranking signal. Could catch high-confidence probe errors if features other than $p_i$ are predictive.
- **Con:** Requires calibration data with baseline scores for every example (expensive). Risk of overfitting the routing to calibration distribution. Adds complexity and another trained component.
- **Con:** If $g$ only has access to $p_i$ as a feature (no additional features), it reduces to a monotonic transform of $u_i$ and is equivalent to what we already have. The signal must come from something _other_ than the probe score.
- **Feasibility:** Would need probe-internal features (activations) to be useful. These are available (we already extract activations for the probe), but this is a significant research direction on its own.

---

## Option 2: Budget-Constrained Proportional Allocation

### Idea

Allocate a per-batch budget rate proportional to batch difficulty, subject to a total budget constraint. Within each batch, use standard percentile selection.

### Formalization

Given $K$ batches with mean uncertainties $\bar{u}_1, \ldots, \bar{u}_K$ and a global budget target $B$, set:

$$r_k = \text{clip}\!\left(B \cdot \frac{\bar{u}_k}{\bar{u}},\ 0,\ 1\right)$$

where $\bar{u} = \frac{1}{K} \sum_k \bar{u}_k$. This ensures $\mathbb{E}[r_k] = B$ so the total budget is controlled, while harder batches get more.

Within batch $k$, select the top $r_k$ fraction by uncertainty (= standard percentile selection).

### Assessment

- **Pro:** Budget-controlled, adaptive, clean formulation.
- **Con:** Since the ranking is still by $u_i$, this selects the same examples as a threshold would for the same count. The only difference from the current threshold method is the _mapping_ from batch uncertainty to rate — linear-proportional vs the step function implied by a fixed threshold.
- **Con:** On our current data, $\bar{u}_k$ ranges from 0.113 to 0.178. With $B = 0.39$ and $\bar{u} \approx 0.148$, rates would range from $0.39 \times 0.113/0.148 = 0.298$ to $0.39 \times 0.178/0.148 = 0.469$. This is very similar to what the threshold already produces (0.28 to 0.53). The difference is marginal smoothing of the rate curve.
- **Key problem:** Same ranking signal → same examples selected for the same count. Cannot overcome the fundamental limitation that fixed-rate wins when batches are homogeneous.

---

## Option 3: Different Per-Example Signal via Probe-Baseline Disagreement Prediction

### Idea

Instead of routing based on how uncertain the probe is, route based on how likely the probe and baseline are to _disagree_. This is a refinement of Option 1, but with a specific estimable target.

### Formalization

On calibration data, compute $d_i = |p_i - b_i|$ (probe-baseline score disagreement). Fit a regression model $h(p_i) \approx \mathbb{E}[d_i \mid p_i]$ (or use richer features). At test time, escalate examples with high predicted disagreement.

More concretely, bin calibration examples by probe score $p_i$ and compute the mean $|p_i - b_i|$ per bin. This gives a calibrated lookup: "for a probe score of $p$, how much does the baseline typically disagree?"

### Assessment

- **Pro:** Directly targets the examples where escalation changes the outcome, not just where the probe is uncertain.
- **Con:** If $h$ is only a function of $p_i$, then $h(p_i)$ is a monotonic-ish function of $u_i$ (more uncertain → more disagreement), so the ranking might be very similar. The signal is only different if disagreement is non-monotonic in probe confidence (e.g., the baseline disagrees a lot at certain confidence levels but not others).
- **Feasibility:** Easy to compute on calibration data. Worth checking empirically whether $\mathbb{E}[|p_i - b_i| \mid p_i]$ is actually non-monotonic.

---

## Option 4: Threshold + Different Merge Strategy

### Idea

Keep the threshold-based selection as-is, but change what happens to escalated examples. Currently merge_strategy="replace" means we throw away the probe score entirely. If the baseline is wrong, we lose the probe's correct prediction.

### Formalization

Instead of $\text{final}_i = b_i$ (replace), use a confidence-weighted merge:

$$\text{final}_i = w_i \cdot p_i + (1 - w_i) \cdot b_i$$

where $w_i$ is a function of probe confidence. When the probe is very uncertain ($u_i$ high), $w_i \to 0$ (trust baseline). When moderately uncertain ($u_i$ near the threshold edge), $w_i \to 0.5$ (hedge).

Or more simply: use merge_strategy="avg" instead of "replace."

### Assessment

- **Pro:** Reduces the "breakage" problem (82 examples where escalation hurt). Probe-right-baseline-wrong examples would retain some probe signal.
- **Pro:** Minimal code change; already implemented as merge_strategy="avg".
- **Con:** Also dilutes corrections (probe-wrong-baseline-right gets averaged instead of fully corrected).
- **Con:** Doesn't change the selection mechanism, so doesn't address the core adaptivity question.
- **Worth testing:** Quick experiment — rerun with merge_strategy="avg" and compare.

---

## Summary

| Option | Changes ranking? | Changes rate? | Changes merge? | Complexity | Expected gain |
|--------|:---:|:---:|:---:|:---:|:---:|
| 1. Learned escalation | Yes (new signal) | No | No | High | High if features available |
| 2. Proportional allocation | No | Yes (smooth) | No | Low | Low (similar to threshold) |
| 3. Disagreement prediction | Maybe | No | No | Medium | Depends on non-monotonicity |
| 4. Different merge | No | No | Yes | Minimal | Moderate (reduces breakage) |

### Honest assessment

Options 2 and 3 (if using only probe score as feature) are unlikely to meaningfully differ from what we have, because the ranking signal is the same. **The fundamental constraint is: with only probe score as the routing signal, all methods select the same examples for the same count.** The only difference is count-per-batch, and our data shows that's not enough.

Options that could genuinely help:
- **Option 1 with probe activations** — use internal representations, not just the scalar output. This is the only way to get a genuinely different ranking.
- **Option 4** — quick win, orthogonal to the selection question.
- **Stratified batching** — rather than changing the method, change the experimental conditions to ones where the existing method's adaptivity actually matters.
