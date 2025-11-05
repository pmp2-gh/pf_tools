---
title: "📊 Portfolio Efficient Frontier, Rolling Analysis & Rebalancing Policy"
---

## 1. Overview

This document summarizes the **theory, equations, and practical procedures** for building and maintaining a diversified portfolio using **mean–variance optimization**, **efficient frontier analysis**, and **rolling 10-year evaluation** to inform **strategic asset allocation** and **rebalancing decisions**.

We consider four core assets:

- U.S. Total Stock Market  
- International Total Stock Market  
- U.S. Total Bond Market  
- International Total Bond Market (synthetic or proxy)

---

## 2. Modern Portfolio Theory (Markowitz 1952)

Given a set of assets with expected returns and covariances, a portfolio’s performance is described by two key quantities.

### 2.1 Expected Return

The expected portfolio return is

$$
E[R_p] = \mu_p = \sum_{i=1}^{n} w_i \mu_i = \mathbf{w}^\top \boldsymbol{\mu}
$$

where

- $\mathbf{w} = [w_1, w_2, \dots, w_n]^\top$ are asset weights (summing to 1),
- $\boldsymbol{\mu} = [\mu_1, \mu_2, \dots, \mu_n]^\top$ are expected returns.

### 2.2 Portfolio Variance

The portfolio variance (risk) is

$$
\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}
$$

where

- $\Sigma$ is the covariance matrix of asset returns,
- $\sigma_p = \sqrt{\sigma_p^2}$ is the portfolio volatility.

---

## 3. Efficient Frontier

The **efficient frontier** is the set of portfolios that are **not dominated** in risk–return space: for each point on the frontier, there is no other portfolio with the **same return and lower volatility** or **same volatility and higher return**.

### 3.1 Analytical Form (Unconstrained Case)

If shorting and any weights are allowed, the Markowitz problem

- minimize $\mathbf{w}^\top \Sigma \mathbf{w}$
- subject to $\mathbf{w}^\top \boldsymbol{\mu} = r$ and $\mathbf{w}^\top \mathbf{1} = 1$

has a closed-form solution.

Define

$$
A = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}, \quad
B = \mathbf{1}^\top \Sigma^{-1} \boldsymbol{\mu}, \quad
C = \boldsymbol{\mu}^\top \Sigma^{-1} \boldsymbol{\mu}, \quad
D = AC - B^2.
$$

Then, for a target return $r$, the minimum-variance weights are

$$
\mathbf{w}(r) = \Sigma^{-1} \left[
\frac{C - Br}{D} \mathbf{1} + \frac{Ar - B}{D} \boldsymbol{\mu}
\right].
$$

This formula **can produce negative weights**, i.e. short positions.

---

## 4. Long-Only Frontier (Practical Portfolios)

Real-world portfolios are often **long-only**:

- $w_i \ge 0$ for all $i$,
- $\sum_i w_i = 1$.

Because of these inequality constraints, the neat closed-form above no longer applies. A practical approach:

1. Randomly generate many long-only weight vectors $\mathbf{w} \ge 0$ with $\sum_i w_i = 1$.
2. For each, compute
   $$
   \mu_p = \mathbf{w}^\top \boldsymbol{\mu}, \qquad
   \sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}.
   $$
3. Keep only the **Pareto-efficient** portfolios.

The upper envelope of these points is the **practical (long-only) efficient frontier**.

---

## 5. Sharpe Ratio and the Tangent Portfolio

To compare portfolios with different risk levels, we use the **Sharpe ratio**:

$$
S = \frac{E[R_p] - R_f}{\sigma_p}
$$

where $R_f$ is the **risk-free rate**.

- Higher $S$ = more return per unit of risk.
- The frontier portfolio with the highest $S$ is the **tangent portfolio** (when a risk-free asset exists).

---

## 6. Rolling 10-Year Analysis

Historical estimates of $\boldsymbol{\mu}$ and $\Sigma$ **change over time**. To see how sensitive an optimizer is, we do a **rolling-window** study.

For each month $t$:

1. Take the past 120 months (10 years) of returns.
2. Estimate
   $$
   \boldsymbol{\mu}_t = 12 \cdot \overline{r}_{t-119:t}, \qquad
   \Sigma_t = 12 \cdot \text{Cov}_{t-119:t}.
   $$
3. Build a long-only frontier using $(\boldsymbol{\mu}_t, \Sigma_t)$.
4. Either pick:
   - the portfolio whose return is closest to a target $r^\*$, **or**
   - the max-Sharpe portfolio for that window.
5. Store the weights and diagnostics.

This gives a time series of “what the optimizer would have told me at that time.”

---

## 7. Interpreting Rolling Results

| Observation          | Interpretation                                                    |
|----------------------|-------------------------------------------------------------------|
| Weights are stable   | Inputs/structure are robust → static allocation is reasonable    |
| Weights jump around  | Optimizer is sensitive → add constraints / wider bands           |
| Sharpe drifts        | Market regime changes (rates, vol, correlations)                 |

The rolling data helps you build a **robust** policy allocation, not chase every month’s “optimal” weights.

---

## 8. Building a Policy Allocation

Let the rolling exercise give you $T$ sets of optimal weights:

$$
\mathbf{w}^{(1)}, \mathbf{w}^{(2)}, \dots, \mathbf{w}^{(T)}.
$$

For asset $i$, take the median across all windows:

$$
\tilde{w}_i = \text{median}\bigl(w_i^{(1)}, \dots, w_i^{(T)}\bigr).
$$

These medians may not sum to 1, so normalize:

$$
w_i^{(\text{policy})} = \frac{\tilde{w}_i}{\sum_j \tilde{w}_j}.
$$

This is your **policy portfolio** — the center of gravity of historical optimal allocations.

---

## 9. Defining Allocation Bands

To avoid over-trading, define a tolerance around each policy weight:

$$
\text{Band}_i = \bigl[\, w_i^{(\text{policy})} - \delta,\; w_i^{(\text{policy})} + \delta \,\bigr]
$$

where $\delta$ is often **0.05** (i.e. ±5 percentage points).

**Example**

| Asset      | Policy | Band       |
|------------|--------|------------|
| US Stock   | 0.40   | 0.35–0.45  |
| Intl Stock | 0.20   | 0.15–0.25  |
| US Bond    | 0.30   | 0.25–0.35  |
| Intl Bond  | 0.10   | 0.05–0.15  |

---

## 10. Rebalancing Strategy

### 10.1 Calendar-Based
- Rebalance on a fixed schedule (e.g. annually).
- Simple, but may trade unnecessarily.

### 10.2 Threshold-Based
- Check periodically.
- Rebalance **only** if a weight is **outside** its band.
- Usually more cost- and tax-efficient.

### 10.3 Hybrid
- Check monthly/quarterly.
- Act only if out of band.

---

## 11. Decision Procedure

Let $\mathbf{w}_{\text{current}}$ be your actual portfolio weights.

1. For each asset $i$, get $w_i^{(\text{policy})}$ and its band.
2. If
   $$
   w_{i,\text{current}} < w_i^{(\text{policy})} - \delta
   \quad \text{or} \quad
   w_{i,\text{current}} > w_i^{(\text{policy})} + \delta,
   $$
   then **rebalance asset $i$**.
3. Trade back toward the policy weight (or the midpoint of the band).

---

## 12. Interpreting Volatility

If a portfolio has $\mu_p = 8\%$ and $\sigma_p = 7\%$, then (roughly, assuming normality):

- About 68% of years fall in
  $$
  8\% \pm 7\% \quad \Rightarrow \quad 1\% \text{ to } 15\%
  $$
- About 95% of years fall in
  $$
  8\% \pm 14\% \quad \Rightarrow \quad -6\% \text{ to } 22\%
  $$

Volatility = typical fluctuation around the mean, not guaranteed bounds.

---

## 13. Risk-Free Rate

Used in Sharpe:

- $R_f$ is usually a Treasury yield (3m T-bill, 1y, etc.).
- Higher $R_f$ makes it harder for risky portfolios to look good on a Sharpe basis.

---

## 14. Handling Median Normalization

If
$$
\sum_i \tilde{w}_i \neq 1,
$$
then normalize:

$$
w_i^{(\text{normalized})} = \frac{\tilde{w}_i}{\sum_j \tilde{w}_j}.
$$

Now weights sum to 1 and you can define bands.

---

## 15. Summary: From Math to Action

| Step | Concept                      | Output                       |
|------|------------------------------|------------------------------|
| 1    | Get prices, make returns     | Monthly return series        |
| 2    | Estimate $\mu$, $\Sigma$     | Inputs for optimization      |
| 3    | Build frontiers              | Risk–return trade-off        |
| 4    | Rolling 10y analysis         | Stability over time          |
| 5    | Take medians, normalize      | Policy allocation            |
| 6    | Add bands                    | Rebalancing tolerance        |
| 7    | Compare to current weights   | Rebalance signals            |

---

## 16. Example Interpretation

If rolling results over 30 years suggest:

| Asset      | Median | 25–75% Range |
|------------|--------|--------------|
| US Stock   | 41%    | 35–48%       |
| Intl Stock | 18%    | 14–23%       |
| US Bond    | 31%    | 27–36%       |
| Intl Bond  | 10%    | 6–13%        |

then a policy of **40/20/30/10** with **±5%** bands is a sensible, data-backed long-term mix.

---

## 17. Key Principles

1. **Optimization is fragile** — don’t chase tiny differences.
2. **Diversification is robust** — own multiple assets.
3. **Discipline beats precision** — policy + bands > re-optimize monthly.
4. **Use models as maps, not oracles.**

---

## 18. References

- Markowitz, H. (1952). *Portfolio Selection*. **Journal of Finance**.  
- Sharpe, W. F. (1966). *Mutual Fund Performance*. **Journal of Business**.  
