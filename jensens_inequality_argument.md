# Why Fixed-Rate Beats Adaptive Threshold: A Jensen's Inequality Argument

## Setup

We have a cascade where a cheap probe routes uncertain examples to an expensive baseline. Both the **adaptive** (fixed threshold) and **fixed-rate** methods rank examples by the same signal: probe uncertainty $u_i = \min(p_i, 1 - p_i)$.

- **Fixed-rate**: In each batch, escalate the $K$ most uncertain examples, where $K = \lfloor r \cdot n \rfloor$ is constant across batches.
- **Adaptive threshold**: Escalate all examples with $u_i > \tau$ (equivalently, $p_i \in (1-\theta, \theta)$). The number escalated $K_b$ varies per batch.

Both methods select from the **same ranking** — the only difference is how many per batch.

## Marginal value of escalation

Sort examples within a batch by decreasing uncertainty. Let $v(k)$ be the expected accuracy change from escalating the $k$-th most uncertain example:

$$v(k) = \mathbb{E}\Big[\mathbb{1}[\text{baseline correct}, \text{probe wrong}] - \mathbb{1}[\text{probe correct}, \text{baseline wrong}] \;\Big|\; \text{rank} = k\Big]$$

**Key assumption (empirically verifiable):** $v(k)$ is decreasing in $k$. The most uncertain examples are most likely to be probe errors that the baseline corrects; confident examples are more likely to be correct already, so escalating them risks replacing a right answer with a wrong one.

## Cumulative value function

Define the total value of escalating the top $K$ examples:

$$V(K) = \sum_{k=1}^{K} v(k)$$

Since $v(k)$ is decreasing, $V(K)$ is a **concave** function of $K$: each additional escalation adds less value than the previous one.

## Applying Jensen's inequality

The fixed method achieves value $V(K)$ in every batch.

The adaptive method achieves value $V(K_b)$ in batch $b$, where $K_b$ is random (depends on how many examples exceed the threshold in that batch). Under i.i.d. batches, $\mathbb{E}[K_b] \approx K$ (both methods use roughly the same total budget).

Since $V$ is concave, **Jensen's inequality** gives:

$$\mathbb{E}[V(K_b)] \leq V(\mathbb{E}[K_b]) \approx V(K)$$

The inequality is **strict** whenever $\operatorname{Var}(K_b) > 0$, i.e., whenever the adaptive method's per-batch count actually varies. In our data, $K_b$ ranges from ~18 to ~34 across batches, so the gap is real.

## Intuition

When the adaptive method escalates **more** than $K$ (hard batch):
- The extra examples are the *least uncertain* among those selected — they have the **lowest marginal value**. These are examples near the threshold boundary where the probe is somewhat confident and baseline correction is least likely to help.

When the adaptive method escalates **fewer** than $K$ (easy batch):
- The skipped examples are the *most uncertain* among those not selected — they have the **highest marginal value** among what's dropped. The fixed method would escalate these and benefit from baseline correction.

The adaptive method systematically **wastes budget on low-value escalations** in hard batches and **skips high-value escalations** in easy batches. This is the opposite of efficient allocation.

## When would adaptive win?

The argument breaks down when $V(K)$ is **not the same function across batches** — i.e., when the marginal value curve $v_b(k)$ differs by batch. This happens when:

1. **Batches have genuine difficulty structure** (non-exchangeable data): hard batches might have steeper marginal benefit, making extra escalation there more valuable than equal escalation everywhere.
2. **The ranking signal differs across batches**: if uncertainty means different things in different batches (e.g., due to distribution shift), then a global threshold might capture this while a percentile selection cannot.

In our data, neither condition holds. The variance ratio test gives 0.901 (batches are pure i.i.d. samples), so the marginal value curves are statistically identical across batches. The adaptive method's variability is pure noise, and Jensen's inequality guarantees the fixed method wins.

## Summary

| Condition | Fixed-rate | Adaptive threshold |
|---|---|---|
| i.i.d. batches + concave $V(K)$ | **Optimal** (by Jensen's) | Strictly worse |
| Non-i.i.d. batches (genuine difficulty variation) | Suboptimal (ignores batch info) | **Can win** (adapts budget to difficulty) |
| Different ranking signal per method | N/A | N/A (both use probe uncertainty) |

The core constraint: **with the same per-example ranking signal and i.i.d. data, constant allocation always beats variable allocation when marginal returns are diminishing.** This is not a failure of the uncertainty signal — it's a mathematical certainty.

---

## Escaping Jensen's: exchangeable but not i.i.d. data

### The LTT guarantee is population-level

A key observation: the LTT/SGT guarantee in our framework operates at the **population level**, not per-batch. The p-value bounds use $n =$ total calibration set size, and the risk is computed as an aggregate mean over all calibration examples. Per-batch statistics are purely descriptive. The guarantee states:

$$P(R(\hat{\lambda}) \leq \alpha) \geq 1 - \delta$$

where $R$ is the **population risk**. How test data is grouped into batches is irrelevant to this guarantee — it concerns the threshold $\hat\lambda$ selected during calibration, not the structure of future test data.

This means stratified or structured batching of test data **does not invalidate the guarantee**, as long as calibration data is i.i.d./exchangeable from the same population. The exchangeability requirement is on the calibration examples, not on the batching scheme.

### De Finetti's theorem and genuine batch variation

**De Finetti's theorem**: an infinite exchangeable sequence can be represented as a **mixture of i.i.d. sequences**. There exists a latent variable $\theta$ such that, conditional on $\theta$, examples are i.i.d. from $P_\theta$:

$$P(X_1, \ldots, X_n) = \int \prod_{i=1}^n P_\theta(X_i) \, dF(\theta)$$

When $\theta$ is non-degenerate (i.e., the data is exchangeable but **not** i.i.d.), examples are positively correlated through the shared latent variable. This has a direct consequence for batch-level statistics: the variance of batch means **exceeds** $\sigma^2/n$, because:

$$\operatorname{Var}(\bar{X}_{\text{batch}}) = \frac{\sigma^2}{n} + \frac{n-1}{n} \operatorname{Var}_\theta(\mu_\theta)$$

The second term captures genuine between-batch variation driven by $\theta$. In our current data, the variance ratio is 0.901 (consistent with i.i.d.). Under exchangeable-but-not-i.i.d. data, this ratio would be **significantly above 1.0**, indicating real batch-level structure.

### Why this breaks the Jensen's argument

Under exchangeable-but-not-i.i.d. data, different batches realize different values of $\theta$, giving each batch a **different marginal value curve** $V_b(K)$. Hard batches (high $\theta$, more probe errors) have steeper value curves — each escalation is more likely to correct a genuine error.

The adaptive method's $K_b$ is naturally correlated with the steepness of $V_b$: when a batch is hard, more examples cross the uncertainty threshold, so $K_b$ is large precisely when extra escalations are most valuable. Formally, for a parametric approximation $V_b(K) \approx a_b K - c_b K^2$:

$$\mathbb{E}[V_b(K_b)] = \mathbb{E}[a_b] \cdot \mathbb{E}[K_b] + \operatorname{Cov}(a_b, K_b) - \mathbb{E}[c_b K_b^2]$$

The covariance term $\operatorname{Cov}(a_b, K_b) > 0$ captures the adaptive advantage: budget is allocated where it helps most. When this positive covariance exceeds the Jensen's penalty from variable $K_b$, the adaptive method wins.

### Realistic scenarios

The following settings produce exchangeable-but-not-i.i.d. data where batches have genuine difficulty variation:

1. **Multi-domain LLM monitoring**: Monitoring an LLM across domains (code review, medical QA, legal analysis). Each batch comes from a user session within one domain. The probe has domain-dependent accuracy — easy on code, hard on legal. Calibration data spans all domains (so individual examples are exchangeable), but test batches are domain-coherent. The latent $\theta$ is the domain.

2. **Source-heterogeneous classification**: Sentiment analysis across product categories, or toxicity detection across subreddits. Batches naturally come from one source. Some sources are harder than others. The latent $\theta$ is the source/category.

3. **Temporal clustering with stationary marginals**: A monitoring system where query difficulty has autocorrelation (hard queries cluster in time) but the long-run marginal distribution is stationary. Batches formed from consecutive queries inherit the temporal structure. The latent $\theta$ is the current "difficulty regime."

In all of these, calibration data sampled uniformly from the population is exchangeable with test examples (same marginal $\int P_\theta \, dF(\theta)$), preserving the LTT guarantee. But test batches exhibit genuine difficulty variation through the latent $\theta$, giving the adaptive threshold a structural advantage over fixed-rate allocation.

### Summary: when each method wins

| Data regime | Variance ratio | Jensen's binding? | Winner |
|---|---|---|---|
| i.i.d. batches | $\approx 1$ | Yes | **Fixed-rate** |
| Exchangeable, not i.i.d. (weak $\theta$) | $> 1$ but small | Partially | Depends on $\operatorname{Cov}(a_b, K_b)$ |
| Exchangeable, not i.i.d. (strong $\theta$) | $\gg 1$ | No — covariance dominates | **Adaptive threshold** |

The adaptive method's advantage scales with the strength of the latent batch-level structure: the more $\theta$ varies, the more the covariance term $\operatorname{Cov}(a_b, K_b)$ overcomes the Jensen's penalty. The variance ratio test provides a direct empirical diagnostic for which regime we're in.
