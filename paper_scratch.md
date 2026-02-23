
# Introduction
We want monitoring tools that provide guarantees on the safety and performance of LLMs. Probes are very good, but their ability to generalise is somewhat questionable, it is hard to deploy them with confidence.
At the same time, more powerful monitoring methods are expensive, and cannot be present all the time.
Our method finds a tradeoff with guaranteed performance.
In the context of batch inference, we need to adapt when the batches become harder than expected. Fixed amount per batch is not always optimal. Maintain overall budget competitiveness and performance guarantee.
Baseline is fixed percentile threshold

# Problem formulation
We need a monitoring system for LLM behaviour that is both cheap and effective. We need to find a good balance between probes' cheapness and LLM monitors' performance, and we need to be sure about its performance thanks to guarantees.
Additionally, we want our method to be adaptive - to not waste compute on easy batches, and to deploy more compute on hard batches
> would be nice to have a concise formal way to measure this

compare with 'same budget' fixed method

# Background
(maybe goes at the end?)
probes, monitoring, LTT

# Method
guaranteed performance, budget control, SGT (probably appendix)

# Experiments and Results
- same or better overall performance
- same or better overall budget
- better performance on hard batches and better efficiency on easy ones.

finding what level at which to guarantee is a problem in itself, so we use SGT over these parameters to find the best alpha we can guarantee for some budget.

---

# Extracted Methodology (from codebase)

## System Components

### 1. Probe (cheap classifier)
- A logistic regression classifier trained on internal activations of a small LLM (Llama-3.2-1B, layer 11).
- Activations are reduced via a strategy (e.g., mean-pooling over token positions) to a fixed-size vector.
- Optional calibration (isotonic regression or Platt scaling) on held-out data to improve probability estimates.
- Very cheap to run — just a forward pass through a small model + logistic regression inference.

### 2. Baseline (expensive monitor)
- A large LLM (Gemma-3-27B) used as a high-quality judge/monitor.
- Expensive to run, but more accurate. Serves as the "gold standard" monitor.

### 3. Cascade Strategy
The cascade decides **which examples** get sent to the expensive baseline:
- **Fixed-threshold strategy:** An example is sent to the baseline when the probe is uncertain — specifically when `probe_score < threshold AND (1 - probe_score) < threshold`. The threshold parameter controls the uncertainty band around 0.5.
- **Merge strategy:** For examples sent to baseline, the final score either replaces the probe score ("replace") or averages with it ("avg").
- **Fixed-budget-rate baseline:** For comparison, a strategy that simply sends a fixed fraction of examples to the baseline (ranked by confidence, sends the least confident).

**Prediction rule (single-equation form).** Let $x$ be an input, $p(x) \in [0,1]$ the probe probability for the positive class, and $b(x) \in [0,1]$ the baseline probability. Define probe uncertainty
$$
u(x) = \min\big(p(x), 1 - p(x)\big).
$$
Given a threshold $\tau \in [0.5, 1]$, the cascade sends $x$ to the baseline iff $p(x) \in (1-\tau, \tau)$, equivalently $u(x) > 1 - \tau$. The final score is
$$
s(x) =
\begin{cases}
\mathrm{merge}(p(x), b(x)), & \text{if } u(x) > 1 - \tau,\\
p(x), & \text{if } u(x) \le 1 - \tau.
\end{cases}
$$
with
$$
\mathrm{merge}(p,b) =
\begin{cases}
\frac{p + b}{2}, & \text{(avg)},\\
b, & \text{(replace)}.
\end{cases}
$$

**Fixed-budget-rate baseline (same merge form).** Let $r \in (0,1]$ be the budget rate and let $\mathcal{I}_r$ be the indices of the middle $r$ fraction of examples when sorting by $p(x)$ (i.e., least confident in the sense of being closest to $0.5$). Then
$$
s(x_i) =
\begin{cases}
\mathrm{merge}(p(x_i), b(x_i)), & i \in \mathcal{I}_r,\\
p(x_i), & i \notin \mathcal{I}_r.
\end{cases}
$$


### 4. Risk Guarantees via Learn-then-Test (LTT)
For a given cascade threshold, we want to guarantee that a risk metric (e.g., `1 - ROC-AUC`, or `1 - accuracy`) stays below a level alpha with probability at least `1 - delta`.

**Procedure:**
1. Compute empirical risk `r_hat` on calibration data for a given threshold.
2. Compute a p-value using a concentration inequality bound:
   - **Hoeffding-Bentkus (HB) bound:** `p = min(e * binom.cdf(ceil(n*r_hat), n, alpha), exp(-n * KL(min(r_hat, alpha) || alpha)))` — used for accuracy/ROC-AUC risks.
   - **Binomial bound:** Used for budget cost risk.
3. If `p <= delta`, we reject the null hypothesis (i.e., we can guarantee that the true risk is <= alpha).

### 5. Sequential Graphical Testing (SGT) over (threshold x alpha) Grid
The key methodological contribution: rather than testing a single (threshold, alpha) pair, we test a **2D grid** of hypotheses simultaneously while controlling the Family-Wise Error Rate (FWER).

**Algorithm:**
1. Define a grid: `n_thresholds` threshold values x `n_alphas` alpha values = `m` hypotheses.
2. Each hypothesis H_{i,j}: "The risk at threshold_i exceeds alpha_j" (null hypothesis).
3. Compute p-values for all hypotheses using the bounds above.
4. Order hypotheses strategically:
   - Thresholds ordered by risk difficulty (ascending for budget, descending for others).
   - Alphas ordered descending (most permissive first).
5. Construct a **graph** encoding how alpha-weight flows between hypotheses on rejection:
   - **Row-chain graph** (used in budget_target config): Each row (one per threshold) gets equal initial weight. Within a row, hypotheses are tested sequentially. Surplus flows to next row.
   - Also available: lattice graph, uniform lattice, chain graph.
6. Run graphical testing (Bretz et al. 2009): iteratively reject hypotheses, redistributing unused alpha-weight to remaining hypotheses, increasing power.
7. **Output:** The set of all valid (threshold, alpha) pairs for which we can guarantee risk <= alpha.

### 6. Selection Modes
After SGT identifies all valid pairs, we select the "headline" pair to use:
- **`best_alpha`:** Choose the tightest (smallest) alpha guarantee; break ties by lowest budget.
- **`best_threshold`:** Choose lowest budget cost; break ties by tightest alpha.
- **`budget_target`:** Choose the threshold closest to a desired budget target (e.g., 0.3); break ties by tightest alpha. This answers: "Given a budget I can afford, what's the best reliability guarantee I can get?"

### 7. Optional: Pareto Pre-filtering and Deduplication
- **Pareto testing:** Before SGT, filter thresholds to only those on the Pareto front of (primary risk, secondary risk) — e.g., keep only thresholds that aren't dominated in both ROC-AUC error and budget cost. Reduces hypothesis count, increasing SGT power.
- **Threshold deduplication:** Remove thresholds with identical empirical risk on calibration data. Avoids wasting hypotheses on indistinguishable thresholds.

## Experiment Pipeline (sgt_cascade_budget_target.yaml)

**Config summary:**
- Guaranteed risk: `roc_auc_error` (1 - ROC-AUC)
- Guarantee probability: 0.9 (delta = 0.1)
- Grid: 25 thresholds in [0.6, 1.0] x 15 alphas in [0.05, 0.7]
- Selection mode: `budget_target` with target 0.3
- Graph: row-chain (rows = thresholds)
- Probe: mean-pooled Llama-3.2-1B activations (layer 11), isotonic regression calibration
- Baseline: Gemma-3-27B
- Merge strategy: replace
- Pareto testing enabled (secondary risk: budget)
- Threshold deduplication enabled

**Pipeline steps (run_cascade_experiments_pipeline.py):**

1. **SGT Cascade Experiment** — Run the full SGT procedure:
   - Split data into train/calib/test
   - Train probe on training data
   - Calibrate probe (isotonic regression) on auxiliary data
   - Compute probe and baseline scores on all splits
   - Optionally Pareto-filter and deduplicate thresholds
   - Build (threshold x alpha) hypothesis grid
   - Run SGT to find all valid pairs
   - Select best pair via budget_target mode
   - Run cascade on test data with selected threshold
   - Record per-batch statistics (budget, accuracy, F1, ROC-AUC, uncertainty)

2. **Analyse SGT** — Generate single-experiment analysis:
   - Overall performance metrics
   - Batch-level distributions (budget, uncertainty, performance)
   - Uncertainty vs. metrics scatter plots
   - Cascade vs. probe comparison

3. **Fixed-Budget Cascade** — Run the baseline comparison:
   - Reuse the same test data from step 1
   - Apply fixed-budget-rate strategy (default: match the adaptive method's mean budget)
   - Compute same metrics for fair comparison

4. **Analyse Fixed** — Same analysis as step 2, for the fixed baseline.

5. **Comparison** — Head-to-head analysis:
   - Overall performance comparison (adaptive vs fixed)
   - Distribution comparison (per-batch budget and performance)
   - Paired batch-level comparison
   - Uncertainty analysis comparison

**Key outputs:**
- Rejection heatmap showing which (threshold, alpha) pairs were validated
- Per-batch budget and performance distributions for both methods
- Head-to-head comparison plots (adaptive vs fixed at same mean budget)

---

# Discussion: What's Missing for a Solid Paper

## Experiments We Have
1. SGT cascade with budget_target selection on ROC-AUC error risk
2. Fixed-budget cascade baseline (matched budget)
3. Per-batch comparison showing adaptive vs fixed behavior
4. Analysis plots (performance, budget distributions, uncertainty)

## What We Likely Need

### Critical Gaps

1. **Multiple datasets / domains:** Currently only Anthropic safety data. Need to show generalization:
   - Different safety benchmarks (ToxiGen, RealToxicityPrompts, HarmBench, etc.)
   - Different task types (toxicity, helpfulness, factuality)
   - Different distribution shifts between train and test

2. **Multiple risk metrics side-by-side:** Currently config targets ROC-AUC error. Should show results for:
   - Accuracy error
   - F1 error
   - Budget cost as the guaranteed risk (flip the question: "guarantee budget <= X, maximize performance")
   - Table showing all metrics for each configuration

3. **Ablation studies:**
   - Graph type comparison (row-chain vs lattice vs uniform-lattice) — impact on number of rejections and final alpha
   - Grid resolution (fewer vs more thresholds/alphas) — power vs computation tradeoff
   - Pareto filtering on vs off — how much power does it recover?
   - Deduplication on vs off
   - Calibration impact (none vs Platt vs isotonic)
   - Merge strategy (replace vs avg)

4. **Different probe/baseline pairs:**
   - Vary the probe model (different sizes, different layers)
   - Vary the baseline model (different capabilities/costs)
   - Show the method works across the capability spectrum

5. **Batch difficulty variation experiment:**
   - This is the core claimed advantage — the adaptive method should shine when batch difficulty varies.
   - Need a controlled experiment: create test batches with **intentionally varying difficulty** (e.g., mix easy/hard batches) and show the adaptive method reallocates budget while fixed doesn't.
   - Currently batches seem to be random splits — need to show what happens with non-uniform difficulty.

6. **Guarantee verification (coverage experiment):**
   - Run the method many times (different random seeds / data splits).
   - Verify that the risk guarantee actually holds at the claimed rate (e.g., guarantee holds in >= 90% of runs when delta=0.1).
   - This is the empirical validation that the statistical theory works in practice.

### Important but Less Critical

7. **Computational cost analysis:**
   - Wall-clock time / FLOPs comparison between probe-only, baseline-only, cascade methods
   - Show the actual cost savings in concrete terms

8. **Comparison with other adaptive methods:**
   - Conformal prediction-based cascades
   - Uncertainty quantification baselines (MC dropout, ensembles)
   - Other learned routing/gating mechanisms

9. **Sensitivity to calibration data size:**
   - How much calibration data is needed for the guarantees to be tight?
   - Plot: calibration set size vs achieved alpha (tightness of guarantee)

10. **Selection mode comparison:**
    - Show results for all three selection modes (best_alpha, best_threshold, budget_target)
    - Demonstrate the Pareto front of (budget, guarantee tightness) that the method can achieve

11. **Real deployment scenario:**
    - Streaming / online adaptation (rather than batch)
    - Or at minimum, a realistic batch-processing scenario with naturally varying difficulty

12. **Formal measure of adaptivity:**
    - The paper outline notes "would be nice to have a concise formal way to measure this"
    - Could define: variance of per-batch budget (adaptive should have higher variance than fixed, correlated with difficulty)
    - Or: correlation between batch difficulty and budget allocated
    - Or: Gini coefficient of budget allocation vs batch difficulty ranking

---

# Analysis: Why Fixed-Rate Beats Adaptive on i.i.d. Data (and How to Fix It)

## Jensen's Inequality Argument

Both the adaptive (fixed threshold) and fixed-rate methods rank examples by the same signal: probe uncertainty `u_i = min(p_i, 1-p_i)`. The only difference is how many examples per batch are escalated.

Define `V(K)` = total accuracy gain from escalating the `K` most uncertain examples. Since the marginal benefit of escalation decreases with uncertainty rank (verified empirically: the most uncertain examples benefit most from baseline correction), `V(K)` is **concave**.

The fixed method uses constant `K` per batch. The adaptive method uses variable `K_b` with `E[K_b] ≈ K`. By Jensen's inequality: `E[V(K_b)] ≤ V(E[K_b]) = V(K)`. **Constant allocation always beats variable allocation when marginal returns are diminishing and batches are i.i.d.**

### Empirical verification
- Marginal value is clearly decreasing: ~0.4 at rank 0, ~0 by rank 30, slightly negative beyond rank 50.
- `V(K)` is visibly concave.
- Adaptive `K_b` ranges [18, 34] with `Var(K_b) = 12.8` — sufficient variance for Jensen's gap to be real.
- Variance ratio of batch mean uncertainty = 0.901, confirming batches are pure i.i.d. noise.

### Intuition
When adaptive escalates **more** (hard batch): the extra examples are the least uncertain among those selected — lowest marginal value. When adaptive escalates **fewer** (easy batch): the skipped examples are the most uncertain among those not selected — highest marginal value among what's dropped. Adaptive systematically wastes budget on low-value escalations and skips high-value ones.

## Escaping Jensen's: Exchangeable but not i.i.d. Data

The LTT guarantee is **population-level**: it concerns the threshold selected during calibration, not how test data is batched. Stratified or structured batching of test data does not invalidate the guarantee, as long as calibration data is exchangeable from the same population.

Under exchangeable-but-not-i.i.d. data (De Finetti: mixture of i.i.d. with latent variable θ), examples are correlated through θ, inflating batch-level variance beyond `σ²/n`. Different batches realize different θ, giving each a different marginal value curve `V_b(K)`. The adaptive method's `K_b` is naturally correlated with the steepness of `V_b` — the covariance `Cov(steepness, K_b) > 0` can overcome Jensen's penalty.

### Realistic scenarios for exchangeable-but-not-i.i.d. data
1. **Multi-domain LLM monitoring**: Batches from user sessions within one domain (code, medical, legal). Probe difficulty varies by domain. Calibration spans all domains.
2. **Source-heterogeneous classification**: Sentiment across product categories, toxicity across subreddits. Batches from one source.
3. **Temporal clustering**: Query difficulty has autocorrelation but stationary marginals. Consecutive batches inherit temporal structure.

## Stratified Batching Experiment

To simulate exchangeable-but-not-i.i.d. conditions, we sort test examples by probe uncertainty and form batches from consecutive examples. This creates genuine batch-level difficulty variation while preserving the population-level LTT guarantee (same calibration data, same examples, just reordered).

### Results: Adaptive wins on accuracy under stratification

**Global metrics (over full test set):**

| Condition | Method | Accuracy | ROC-AUC | Budget |
|---|---|---|---|---|
| Random | Adaptive | 0.8845 | 0.9215 | 0.393 |
| Random | Fixed | 0.8944 | 0.9319 | 0.391 |
| Stratified | Adaptive | 0.8842 | 0.9215 | 0.385 |
| Stratified | Fixed | 0.8536 | 0.9206 | 0.391 |

- **Random batches**: Fixed wins on both accuracy (+1pp) and ROC-AUC (+1pp). Jensen's inequality in action.
- **Stratified batches**: **Adaptive wins on accuracy** (+3pp, p=0.026). Global ROC-AUC is ~identical for all conditions (~0.92), as expected since the same examples receive the same scores regardless of batching.
- The accuracy gap flips sign: from -1pp (random) to +3pp (stratified). This demonstrates that the adaptive method's advantage scales with batch-level structure.

### Budget allocation
- Random: Adaptive budget correlates with uncertainty (Spearman ρ=0.70) but range is narrow (0.28–0.53).
- Stratified: Adaptive budget ranges from 0 (easy batches, delegates nothing) to 1.0 (hard batches, delegates everything), ρ=0.86. Fixed stays flat at 0.39.

### Per-batch F1 and ROC-AUC are ill-conditioned under stratification

While accuracy shows a clear adaptive advantage, per-batch F1 and ROC-AUC degrade dramatically under stratification — many batches score 0 for F1 and near 0.5 for ROC-AUC. This is **not a method failure** but a metric artifact caused by the interaction between uncertainty-based stratification and class-conditional probe confidence.

**Root cause: Asymmetric probe confidence across classes.** Despite the dataset being perfectly balanced (50/50 positive/negative), the probe's confidence distribution is asymmetric:
- The probe is most confident about **negatives** (scores very near 0, uncertainty ~0.01).
- The probe is less confident about **positives** (scores near 0.8–0.9, uncertainty ~0.1–0.15).

When sorting by uncertainty ascending:
- **Easiest batches (index 0–5)**: Skewed negative (only 23–42% positive). The probe's most confident predictions are disproportionately negative.
- **Batches 6–9**: Dramatically skewed, only 3–9% positive. These are moderately confident negative predictions with almost no positives.
- **Hardest batches (index 42–46)**: Skewed positive (55–72% positive). The uncertain examples are disproportionately positive.

**Consequences for per-batch metrics:**
- **F1 = 0** in easy batches: With only 2–6 positives out of 64, missing even one tanks F1. Note that accuracy is still ~95–100% in these same batches (correctly predicting the dominant negatives).
- **ROC-AUC near 0.5**: With a handful of positives, a single misranked example destroys the ranking metric. The metric is essentially noise with so few positive examples.

**Conclusion**: Per-batch accuracy is the appropriate metric for stratified evaluation because it is robust to class imbalance within batches. Global F1 and ROC-AUC (computed over the full test set where class balance is preserved) should be reported to confirm no overall degradation. Per-batch F1 and ROC-AUC are ill-conditioned under any stratification scheme that correlates with class-conditional confidence.

### Summary: When each method wins

| Data regime | Variance ratio | Winner (accuracy) |
|---|---|---|
| i.i.d. batches | ≈ 1.0 | **Fixed-rate** (Jensen's inequality) |
| Stratified (strong batch structure) | >> 1.0 | **Adaptive threshold** (covariance overcomes Jensen's) |

The variance ratio test (`observed_var(batch_means) / expected_var_iid`) provides a direct diagnostic for which regime the data is in. The adaptive method's advantage scales with the strength of batch-level structure.
