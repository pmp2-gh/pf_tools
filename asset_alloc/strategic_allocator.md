This document explains the theory and practice behind the portfolio script. It covers:

- Modern Portfolio Theory (MPT)
- Long-only, Monte Carlo efficient frontiers
- Rolling 10-year optimization
- Policy portfolio from rolling medians
- Quantile-based rebalancing bands
- Investor profiles (so it works for 25-year-olds and people 10 years from retirement)
- Simple stress tests for “what if 2008/2022 happens again”

---

## 1. Setup and Asset Universe

We assume a small, sensible universe (adjustable in the script):

- U.S. stocks
- (optional) international stocks
- U.S. bonds
- (optional) international/synthetic bonds

The script pulls prices (e.g. via `yfinance`), builds **monthly** returns, and annualizes when constructing the frontier.

---

## 2. Modern Portfolio Theory (Markowitz 1952)

Let there be $n$ assets with expected returns $\boldsymbol{\mu} \in \mathbb{R}^n$ and covariance matrix $\Sigma \in \mathbb{R}^{n \times n}$, and portfolio weights $\mathbf{w} \in \mathbb{R}^n$ such that

$$
\sum_{i=1}^n w_i = 1.
$$

### 2.1 Expected Portfolio Return

$$
E[R_p] = \mu_p = \mathbf{w}^\top \boldsymbol{\mu}.
$$

### 2.2 Portfolio Variance / Volatility

$$
\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}, \qquad
\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}.
$$

This is the basic MPT core the script uses after it computes monthly returns.

---

## 3. Analytical Efficient Frontier (Unconstrained)

If we allow any weights (even negative, i.e. shorting), the Markowitz problem has a closed form.

Define

$$
A = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}, \quad
B = \mathbf{1}^\top \Sigma^{-1} \boldsymbol{\mu}, \quad
C = \boldsymbol{\mu}^\top \Sigma^{-1} \boldsymbol{\mu}, \quad
D = AC - B^2.
$$

For a target return $r$, the minimum-variance weights are

$$
\mathbf{w}(r) = \Sigma^{-1}
\left(
\frac{C - Br}{D} \mathbf{1}
+
\frac{Ar - B}{D} \boldsymbol{\mu}
\right).
$$

This is what the script calls the “unconstrained frontier” for plotting — useful for intuition, but not what we actually implement for live portfolios.

---

## 4. Long-Only Frontier (Practical Case)

Real portfolios are usually

$$
w_i \ge 0 \quad \text{for all } i, \qquad \sum_i w_i = 1.
$$

This makes the problem nonlinear, so the script does the practical thing:

1. Generate many random long-only portfolios.
2. Annualize their returns and volatilities:
   $$
   \mu_p = \mathbf{w}^\top \boldsymbol{\mu}, \quad
   \sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}.
   $$
3. Sort by volatility and keep only those that are not dominated (Pareto-efficient).

This is the “long-only frontier” the script compares to the analytical one.

---

## 5. Sharpe Ratio

To compare portfolios with different risk levels, we use the Sharpe ratio

$$
S = \frac{E[R_p] - R_f}{\sigma_p},
$$

where $R_f$ is the (annual) risk-free rate.

The script’s `ROLLING_MODE = "max_sharpe"` simply picks, for that month’s frontier, the portfolio with the highest $S$.

---

## 6. Rolling 10-Year Optimization

Markets change, so the script does this every month:

1. Take the last 120 months of returns (a 10-year window).
2. Estimate annualized mean and covariance:
   $$
   \boldsymbol{\mu}_t = 12 \cdot \overline{\mathbf{r}}_{t-119:t},
   \qquad
   \Sigma_t = 12 \cdot \mathrm{Cov}(\mathbf{r}_{t-119:t}).
   $$
3. Build a long-only frontier from those data.
4. Pick **one** portfolio from that frontier using a rule:
   - target return
   - target volatility
   - max Sharpe
5. Store the weights for date $t$.

Do this for every month → you get a time series of “optimal” weights, one per month.

---

## 7. Shrinkage (to Avoid Overfitting)

Historical returns are noisy, so the script blends the sample estimates with forward-looking assumptions:

Mean shrinkage:

$$
\hat{\boldsymbol{\mu}}_t
= (1 - \alpha)\, \boldsymbol{\mu}_{t,\text{sample}}
+ \alpha \, \boldsymbol{\mu}_{\text{long-run}}.
$$

Covariance shrinkage:

$$
\hat{\Sigma}_t
= (1 - \beta)\, \Sigma_{t,\text{sample}}
+ \beta \, \mathrm{diag}(\Sigma_{t,\text{sample}}).
$$

Typical settings in the script are $\alpha \approx 0.5$, $\beta \approx 0.1$ — i.e. “trust history, but not blindly.”

---

## 8. From Rolling Weights to a Policy Portfolio

Suppose you end up with $T$ months of rolling allocations:

$$
\mathbf{w}^{(1)}, \mathbf{w}^{(2)}, \dots, \mathbf{w}^{(T)}.
$$

For each asset $i$, take the **median** over time:

$$
\tilde{w}_i = \mathrm{median} \big( w_i^{(1)}, w_i^{(2)}, \dots, w_i^{(T)} \big).
$$

Then normalize:

$$
w_i^{(\text{policy})}
= \frac{\tilde{w}_i}{\sum_{j} \tilde{w}_j}.
$$

That gives you a **stable, data-backed, long-term mix** — not “whatever the optimizer liked last month.”

---

## 9. Quantile-Based Rebalancing Bands

To avoid over-trading, we use the actual distribution of rolling weights.

For asset $i$:

- Let $q_{\ell,i}$ be the low quantile (e.g. 0.30).
- Let $q_{h,i}$ be the high quantile (e.g. 0.70).

Define the band as

$$
\mathrm{Band}_i =
\big[ \max(q_{\ell,i}, w_i^{(\text{policy})} - \delta),
      \min(q_{h,i}, w_i^{(\text{policy})} + \delta) \big],
$$

where $\delta$ is a fixed cap (e.g. 5 percentage points).

If your **current** weight is outside that band → rebalance. If not → do nothing.

---

## 10. Investor Profiles

The script now has a layer that runs **after** optimization and simply says:

- if you’re young → it’s OK to let the optimizer choose low bonds
- if you’re 10 years from retirement → you must have at least, say, 35% in bonds
- if you’re retired → you must have at least, say, 60% in bonds

Formally, for a given month’s optimized weights $\mathbf{w}$ and a chosen profile with minimum safety $s_{\min}$:

1. Let $\mathcal{S}$ be the set of “safety” assets (bonds).  
2. Compute current safety share:
   $$
   s = \sum_{k \in \mathcal{S}} w_k.
   $$
3. If $s \ge s_{\min}$, keep $\mathbf{w}$.
4. If $s < s_{\min}$, scale down all non-safety assets proportionally and set safety to $s_{\min}$, then renormalize to 1.

This is how we **add** close-to-retirement usefulness without **removing** usefulness for everyone else.

---

## 11. Profiles in Practice

You can think of the built-in profiles like this:

- accumulator: $s_{\min} = 0$
- balanced: $s_{\min} \approx 0.20$
- pre\_retirement: $s_{\min} \approx 0.35$
- in\_retirement: $s_{\min} \approx 0.60$

So if the rolling optimizer says “95% stocks!” but your profile is `pre_retirement`, the script will clamp it to at least 35% bonds and scale the rest.

---

## 12. Stress Tests (for Black-Swan-ish Events)

Mean–variance doesn’t see tails, so the script can apply simple one-period shocks like:

- stocks: $-40\%$, bonds: $0\%$
- stocks: $-35\%$, bonds: $-10\%$
- stocks: $-15\%$, bonds: $-5\%$

Given policy weights $w_{\text{stock}}$ and $w_{\text{bond}}$, the shocked portfolio loss is

$$
\Delta P \approx
w_{\text{stock}} \cdot s_{\text{shock}}
+ w_{\text{bond}} \cdot b_{\text{shock}}.
$$

If you’re close to retirement and this number is worse than, say, $-25\%$, you raise $s_{\min}$ in your profile and run again.

---

## 13. Picking the Mode

You have three legit choices in the script:

1. **Max Sharpe** (`ROLLING_MODE = "max_sharpe"`)
   $$
   \text{pick } \mathbf{w} \text{ such that } S(\mathbf{w}) \text{ is maximal.}
   $$
   Best for younger / growthy users.

2. **Target Vol** (`ROLLING_MODE = "target_vol"`)
   $$
   \text{pick } \mathbf{w} \text{ such that }
   \sigma(\mathbf{w}) \approx \sigma^\*.
   $$
   Best for people 10 years from retirement — you set $\sigma^\*$ (e.g. $0.08$) to match “how much drop can I sleep through?”

3. **Target Return** (`ROLLING_MODE = "target_return"`)
   $$
   \text{pick } \mathbf{w} \text{ such that }
   \mu(\mathbf{w}) \approx r^\*.
   $$
   Useful if you know you need, say, 6–7% to make the plan work.

All 3 still go through the profile clamp afterward.

---

## 14. Age-Based Cheat Sheet

- 20s–30s → profile = accumulator, mode = max Sharpe
- late 30s–40s → profile = balanced, mode = max Sharpe or target vol
- 50s–mid 60s → profile = pre\_retirement, mode = target vol (e.g. 8%)
- retired → profile = in\_retirement, mode = target vol (e.g. 6%), maybe even tighter bands

**Keep** the cool rolling/MPT stuff, **add** a layer so near-retirees don’t get wrecked.

---

## 15. Volatility Intuition

If a portfolio has volatility $\sigma_p$, a decent first-order way to talk about “bad years” is

$$
\text{bad year} \approx 2 \sigma_p.
$$

So if $\sigma_p = 0.08$ (8%), plan for a possible $-16\%$ year.  
If that’s too much for a near-retiree → lower the target vol or raise the safety floor.

---

## 17. Recap

1. MPT gives us $(\mu, \sigma)$ math.  
2. Long-only frontier makes it realistic.  
3. Rolling windows make it regime-aware.  
4. Medians make it stable.  
5. Quantile bands make it trade sensibly.  
6. Profiles make it useful to *everyone*, not just young accumulators.  
7. Stress tests keep us honest about tail risk.

