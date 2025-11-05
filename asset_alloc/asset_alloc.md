# 📊 Portfolio Efficient Frontier, Rolling Analysis & Rebalancing Policy

## 1. Overview

This document summarizes the **theory, equations, and practical procedures** for building and maintaining a diversified portfolio using **mean–variance optimization**, **efficient frontier analysis**, and **rolling 10-year evaluation** to inform **strategic asset allocation** and **rebalancing decisions**.

We consider four core assets:

- U.S. Total Stock Market  
- International Total Stock Market  
- U.S. Total Bond Market  
- International Total Bond Market (synthetic or proxy)

---

## 2. Modern Portfolio Theory (Markowitz 1952)

Given a set of assets with expected returns and covariances, a portfolio’s performance is described by two key quantities:

### **Expected Return**

\[
E[R_p] = \mu_p = \sum_{i=1}^{n} w_i \mu_i = \mathbf{w}^\top \boldsymbol{\mu}
\]

where  
- \( \mathbf{w} = [w_1, w_2, \dots, w_n]^\top \) are asset weights (summing to 1)  
- \( \boldsymbol{\mu} = [\mu_1, \mu_2, \dots, \mu_n]^\top \) are expected returns

### **Portfolio Variance**

\[
\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}
\]

where  
- \( \Sigma \) is the covariance matrix of asset returns.  
- \( \sigma_p = \sqrt{\sigma_p^2} \) is the portfolio volatility.

---

## 3. Efficient Frontier

The **efficient frontier** represents all portfolios that, for a given level of risk, achieve the **maximum expected return**, or equivalently, for a given return, have **minimum variance**.

### **Analytical Form (Unconstrained Case)**

Given the moments:

\[
A = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}, \quad
B = \mathbf{1}^\top \Sigma^{-1} \boldsymbol{\mu}, \quad
C = \boldsymbol{\mu}^\top \Sigma^{-1} \boldsymbol{\mu}, \quad
D = AC - B^2
\]

The minimum-variance weights for a target return \( r \) are:

\[
\mathbf{w}(r) = \Sigma^{-1} \left[ \frac{C - Br}{D} \mathbf{1} + \frac{Ar - B}{D} \boldsymbol{\mu} \right]
\]

This formula allows short-selling (weights can be negative).

---

## 4. Long-Only Frontier (Practical Portfolios)

Real-world portfolios are typically **long-only** (no shorts, no leverage).  
Analytical solutions don’t apply under inequality constraints, so we use **Monte Carlo simulation**:

1. Generate many random long-only weight vectors \( \mathbf{w} \ge 0 \) with \( \sum w_i = 1 \)
2. Compute \( \mu_p \) and \( \sigma_p \) for each
3. Keep the **Pareto-efficient** ones (those with no higher return for lower volatility)

This forms the **practical efficient frontier**.

---

## 5. Sharpe Ratio and the Tangent Portfolio

The **Sharpe Ratio** measures risk-adjusted return:

\[
S = \frac{E[R_p] - R_f}{\sigma_p}
\]

where \( R_f \) is the risk-free rate (e.g., yield on Treasury bills).

The **max-Sharpe portfolio** (tangent portfolio) is the point on the frontier that maximizes \( S \):
- It provides the best trade-off between risk and reward.
- In theory, a rational investor should hold some combination of this portfolio and the risk-free asset.

---

## 6. Rolling 10-Year Analysis

To assess stability over time, we use **rolling windows** (e.g., 10 years = 120 months):

For each end-of-month \( t \):

1. Take the past 120 months of returns
2. Compute mean returns \( \mu_t \) and covariance \( \Sigma_t \)
3. Build a long-only efficient frontier
4. Extract either:
   - the **max-Sharpe** portfolio, or  
   - the **portfolio closest to a target return** (e.g., 8%)
5. Record weights, expected return, volatility, and Sharpe ratio.

This produces a **time series of “optimal” allocations** and shows how the optimizer’s recommendation drifts through market regimes.

---

## 7. Interpreting Rolling Results

| Observation | Interpretation |
|--------------|----------------|
| Weights are stable | The model is robust — a static allocation works |
| Weights fluctuate wildly | Inputs are noisy — optimization is fragile |
| Sharpe ratio changes | Regime shifts in markets (e.g., inflation, rate cycles) |

The rolling data helps build a **policy allocation** that’s *robust* rather than “perfect”.

---

## 8. Building a Policy Allocation

From the rolling data, compute the **median weight** of each asset across all 10-year windows.

Let \( \tilde{w}_i \) denote the median weight of asset \( i \).  
Normalize so they sum to 1:

\[
w_i^{(\text{policy})} = \frac{\tilde{w}_i}{\sum_j \tilde{w}_j}
\]

This forms your **strategic policy portfolio** — the long-term average mix implied by decades of historical optimal portfolios.

---

## 9. Defining Allocation Bands

To prevent over-trading, define **bands** around each policy weight:

\[
\text{Band}_i = \left[ w_i^{(\text{policy})} - \delta, \; w_i^{(\text{policy})} + \delta \right]
\]

where \( \delta \) (the **band width**) is typically **±5 percentage points**.

Example:

| Asset | Policy | Band |
|--------|---------|------|
| US Stock | 40% | 35–45% |
| Intl Stock | 20% | 15–25% |
| US Bond | 30% | 25–35% |
| Intl Bond | 10% | 5–15% |

---

## 10. Rebalancing Strategy

### **Calendar-Based**
- Rebalance periodically (e.g., annually or semi-annually)
- Simple but can cause unnecessary trades

### **Threshold-Based**
- Rebalance only if a weight exits its band
- Most cost-effective and empirically robust

### **Hybrid**
- Check monthly, act only if outside bands

---

## 11. Decision Procedure

1. **Compute current allocation** \( \mathbf{w}_{\text{current}} \)
2. **Compare** to policy bands:
   \[
   \text{if } w_i < (w_i^{(\text{policy})} - \delta) \text{ or } w_i > (w_i^{(\text{policy})} + \delta) \Rightarrow \text{Rebalance asset } i
   \]
3. **Rebalance** minimally — bring only out-of-band assets back to the mid-point of their bands.

This yields **low turnover** and **consistent risk exposure**.

---

## 12. Interpreting Volatility

The volatility number (\( \sigma_p \)) is the annualized standard deviation of returns.  
If expected return \( \mu_p = 8\% \) and \( \sigma_p = 7\% \):

- About 68% of years fall within \( 8\% \pm 7\% \) → range **1% to 15%**
- About 95% within \( 8\% \pm 14\% \) → range **–6% to 22%**

It measures *uncertainty*, not just downside risk.

---

## 13. Risk-Free Rate

Used in Sharpe ratio computations, \( R_f \) typically represents the yield on U.S. Treasury bills (e.g., 4%).  
It defines the baseline return for a riskless asset.

Typical values:

| Source | Example (2025) | Rate |
|--------|----------------|------|
| 3-month T-bill | Short-term cash proxy | 4.5% |
| 10-year Treasury | Long-term benchmark | 4.0% |
| Academic default | Fixed constant | 2–4% |

---

## 14. Handling Median Normalization

When medians per asset don’t sum to 100%, normalize:

\[
w_i^{(\text{normalized})} = \frac{\tilde{w}_i}{\sum_j \tilde{w}_j}
\]

This preserves full investment and yields consistent bands and policy weights.

---

## 15. Summary: From Math to Action

| Step | Concept | Output |
|------|----------|---------|
| 1 | Collect price history | Monthly returns |
| 2 | Compute \( \mu, \Sigma \) | Expected returns & covariances |
| 3 | Build efficient frontiers | Trade-off between risk & return |
| 4 | Perform rolling analysis | Historical sensitivity |
| 5 | Extract median allocation | Long-term policy weights |
| 6 | Define bands ±δ | Drift tolerance |
| 7 | Compare to current allocation | Rebalance signals |
| 8 | Maintain discipline | Stable, low-turnover portfolio |

---

## 16. Implementation Notes

- Use **monthly total-return indices** for accuracy.
- Minimum 10 years of history per window improves stability.
- Longer lookbacks smooth noise but adapt slower to regime shifts.
- Rebalancing frequency and band width can be tuned for transaction costs.

---

## 17. Example Interpretation

If over 30 years the rolling analysis suggests:

| Asset | Median | 25–75% Range |
|--------|---------|--------------|
| US Stock | 41% | 35–48% |
| Intl Stock | 18% | 14–23% |
| US Bond | 31% | 27–36% |
| Intl Bond | 10% | 6–13% |

Then a policy allocation of **40/20/30/10** with **±5% bands** is a realistic long-term mix.

If your current allocation deviates (e.g., 50/10/30/10), you’re overweight U.S. stocks and underweight international equities → *rebalance gradually toward policy*.

---

## 18. Key Principles

1. **Optimization is fragile**; rely on stable inputs and medians.  
2. **Diversification, not precision**, drives real-world success.  
3. **Rebalancing discipline** matters more than exact weights.  
4. **Use the model as a compass, not a GPS.**

---

## 19. Recommended Parameters

| Parameter | Symbol | Typical Value | Notes |
|------------|---------|----------------|-------|
| Risk-free rate | \( R_f \) | 0.03–0.04 | For Sharpe ratio |
| Band width | \( \delta \) | ±5 pp | Rebalancing tolerance |
| Lookback window | — | 10 years | Rolling analysis |
| Target return | \( r^* \) | 0.06–0.08 | Annualized |
| Portfolios per sim | — | 4,000–20,000 | For MC frontiers |

---

## 20. References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.  
- Sharpe, W.F. (1966). *Mutual Fund Performance*. Journal of Business.  

---

