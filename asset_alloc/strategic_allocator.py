#!/usr/bin/env python3
"""
asset_alloc_v4.py

Portfolio research script (multi-profile, headless, with stress tests):

Keeps everything from v3:
- download prices (yfinance)
- build monthly returns
- (optional) construct synthetic international bond
- annualize mean/cov with shrinkage
- unconstrained analytic frontier
- long-only frontier (Monte Carlo, vectorized, with optional per-asset min/max)
- rolling 10-year allocations with choice of target rule:
    * target_return
    * target_vol
    * max_sharpe
- derive policy bands from empirical quantiles BUT capped around policy median
  so bands don't become uselessly wide
- compare to current allocation
- simple forward-apply backtest
- save plots to PNG (headless-friendly)

ADDED in v4:
- "Investor profiles" layer (accumulator / balanced / pre_retirement) that
  can enforce a minimum safety allocation AFTER optimization, without breaking
  younger users' flows.
- time-to-retirement style clamp: profile can define min safety and we
  re-normalize.
- stress-test printout for black-swan-ish scenarios (stocks -40, stocks -40/bonds -10, etc.)
  so you can see how the *current* policy would behave in a bad regime.
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import yfinance as yf

# =========================
# ===== USER INPUTS =======
# =========================

START_DATE = "1995-01-01"
END_DATE = dt.date.today().strftime("%Y-%m-%d")

# asset universe (trim here if you only want US stock/bond)
TICKERS = {
    "us_stock": "^GSPC",
    # "intl_stock": "EFA",
    "us_bond": "AGG",
}

RUN_FRONTIERS = True
RUN_ROLLING = True
RUN_REBALANCE_CHECK = True
RUN_BACKTEST = True  # apply rolling weights to subsequent month
RUN_STRESS_TESTS = True

# rolling allocator mode: "target_return", "target_vol", "max_sharpe"
ROLLING_MODE = "target_vol"

# legacy target annual return (if mode == "target_return")
TARGET_RETURN = 0.08

# target volatility (if mode == "target_vol")
TARGET_VOL = 0.10  # 10% annualized

# band width for capping around policy median (percentage points)
FIXED_BAND_WIDTH = 5.0

# empirical quantile bands (we'll cap these)
USE_QUANTILE_BANDS = True
LOW_Q = 0.30
HIGH_Q = 0.70

# your CURRENT allocation (in %)
CURRENT_ALLOCATION_PCT = {
    "us_stock_pct": 50.0,
    "us_bond_pct": 0.0,
}

# risk-free for Sharpe
RISK_FREE_RATE = 0.04

# random portfolios per run (for long-only + rolling)
N_PORTFOLIOS = 4000

# 10-year window in months
WINDOW_MONTHS = 120

# shrinkage settings
MEAN_SHRINK_ALPHA = 0.5
COV_SHRINK_ALPHA = 0.1

# long-run forward-looking annual returns to blend with historical
LONG_RUN_RETURNS = {
    "us_stock": 0.058,
    "intl_stock": 0.062,
    "us_bond": 0.040,
    "intl_bond": 0.035,
}

# optional per-asset min/max weights for Monte Carlo (keeps frontier from hugging corners)
USE_WEIGHT_BOUNDS = True
MIN_WEIGHT = 0.02   # 2% floor
MAX_WEIGHT = 0.90   # 90% cap

# =========================
# ===== NEW: PROFILES =====
# =========================
"""
Profiles are additive — they don't remove utility for younger users.
Pick one via INVESTOR_PROFILE. Each profile can enforce a minimum
"safety" allocation (bonds/cash-like) AFTER the optimizer picks its
weights.

"years_to_retirement" is informational here; you could make it dynamic later.
"""
INVESTOR_PROFILES = {
    "accumulator": {
        "desc": "young / high risk tolerance",
        "min_safety_pct": 0.00,    # 0% bonds minimum
    },
    "balanced": {
        "desc": "mid-career / moderate risk tolerance",
        "min_safety_pct": 0.20,    # at least 20% bonds
    },
    "pre_retirement": {
        "desc": "10-ish years to retirement, wants sequence-risk protection",
        "min_safety_pct": 0.40,    # at least 40% bonds
    },
    "in_retirement": {
    "desc": "capital preservation, withdrawals",
    "min_safety_pct": 0.60,        # at least 60% bonds
    }
}

INVESTOR_PROFILE = "pre_retirement"  # change here; does NOT affect other logic

# =========================
# ===== STRESS SCENES =====
# =========================
"""
Simple regime shocks. We keep it at portfolio level; this is enough to
see how a "too stocky" policy looks under 2008-ish or 2022-ish conditions.
"""
STRESS_SCENARIOS = [
    {
        "name": "Equity crash, bonds flat",
        "us_stock_shock": -0.40,
        "us_bond_shock": 0.00,
    },
    {
        "name": "Equity crash, bonds down like 2022",
        "us_stock_shock": -0.35,
        "us_bond_shock": -0.10,
    },
    {
        "name": "Mild stagflation",
        "us_stock_shock": -0.15,
        "us_bond_shock": -0.05,
    },
]

# =========================
# ===== HELPER FUNCS ======
# =========================

def fetch_prices(tickers: dict, start: str, end: str) -> pd.DataFrame:
    series = []
    for name, ticker in tickers.items():
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )

        if df is None or df.empty:
            print(f"[WARN] No data for {ticker}, skipping.")
            continue

        if not isinstance(df.columns, pd.MultiIndex):
            if "Adj Close" in df.columns:
                s = df["Adj Close"].rename(name)
            elif "Close" in df.columns:
                s = df["Close"].rename(name)
            else:
                print(f"[WARN] {ticker} missing Close/Adj Close, skipping.")
                continue
        else:
            adj_candidates = [col for col in df.columns if col[0] == "Adj Close"]
            close_candidates = [col for col in df.columns if col[0] == "Close"]
            if adj_candidates:
                s = df[adj_candidates[0]].rename(name)
            elif close_candidates:
                s = df[close_candidates[0]].rename(name)
            else:
                print(f"[WARN] {ticker} multiindex missing Close, skipping.")
                continue

        series.append(s)

    if not series:
        raise ValueError("No price series could be downloaded.")
    return pd.concat(series, axis=1).ffill().dropna(how="all")


def build_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    monthly_prices = prices.resample("ME").last()
    returns = monthly_prices.pct_change().dropna()
    return returns


def add_synthetic_intl_bond(returns: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    base = returns["us_bond"]
    noise = np.random.normal(0, 0.002, size=len(returns))
    drift = 0.0001
    intl_bond = base * 0.8 + noise + drift
    returns["intl_bond"] = intl_bond
    return returns[["us_stock", "intl_stock", "us_bond", "intl_bond"]]


def shrink_mean(sample_mu: pd.Series, asset_order: list[str]) -> pd.Series:
    long_run = pd.Series({a: LONG_RUN_RETURNS[a] for a in asset_order if a in LONG_RUN_RETURNS})
    # for assets not in LONG_RUN_RETURNS, just keep sample
    for a in asset_order:
        if a not in long_run:
            long_run[a] = sample_mu[a]
    return (1 - MEAN_SHRINK_ALPHA) * sample_mu + MEAN_SHRINK_ALPHA * long_run


def shrink_cov(sample_cov: pd.DataFrame) -> pd.DataFrame:
    diag = np.diag(np.diag(sample_cov.values))
    shrunk = (1 - COV_SHRINK_ALPHA) * sample_cov.values + COV_SHRINK_ALPHA * diag
    return pd.DataFrame(shrunk, index=sample_cov.index, columns=sample_cov.columns)


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 60) -> pd.DataFrame:
    n = len(mu)
    ones = np.ones(n)
    cov_inv = np.linalg.inv(cov.values)
    A = ones @ cov_inv @ ones
    B = ones @ cov_inv @ mu.values
    C = mu.values @ cov_inv @ mu.values
    D = A * C - B * B

    target_rets = np.linspace(mu.min(), mu.max(), n_points)
    rows = []
    for r in target_rets:
        lam = (C - B * r) / D
        gam = (A * r - B) / D
        w = cov_inv @ (lam * ones + gam * mu.values)
        vol = np.sqrt(w @ cov.values @ w)
        rows.append((vol, r, w))
    df = pd.DataFrame(rows, columns=["vol", "ret", "weights"])
    weights_df = pd.DataFrame(df["weights"].tolist(), columns=mu.index)
    return pd.concat([df[["vol", "ret"]], weights_df], axis=1)


def long_only_frontier_vec(returns: pd.DataFrame,
                           n_portfolios: int = 20000,
                           rf: float = 0.0) -> pd.DataFrame:
    mu = returns.mean() * 12
    cov = returns.cov() * 12
    cov_vals = cov.values
    assets = returns.columns
    n_assets = len(assets)

    w = np.random.rand(n_portfolios, n_assets)
    w /= w.sum(axis=1, keepdims=True)

    if USE_WEIGHT_BOUNDS:
        w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
        w /= w.sum(axis=1, keepdims=True)

    port_rets = w @ mu.values
    port_vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov_vals, w))

    df = pd.DataFrame({
        "vol": port_vols,
        "ret": port_rets,
    })
    df["weights"] = list(w)

    df = df.sort_values("vol").reset_index(drop=True)
    efficient = []
    best_ret = -1
    for _, row in df.iterrows():
        if row["ret"] > best_ret:
            efficient.append(row)
            best_ret = row["ret"]
    eff_df = pd.DataFrame(efficient)
    eff_df["sharpe"] = (eff_df["ret"] - rf) / eff_df["vol"]
    return eff_df.reset_index(drop=True)


def pick_portfolio_from_eff(eff_df: pd.DataFrame,
                            assets: list[str],
                            mode: str,
                            target_return: float,
                            target_vol: float,
                            rf: float) -> dict:
    if mode == "target_return":
        idx = (eff_df["ret"] - target_return).abs().idxmin()
    elif mode == "target_vol":
        idx = (eff_df["vol"] - target_vol).abs().idxmin()
    elif mode == "max_sharpe":
        idx = eff_df["sharpe"].idxmax()
    else:
        raise ValueError(f"Unknown mode {mode}")

    row = eff_df.loc[idx]
    w = row["weights"]
    pct_alloc = {f"{a}_pct": wt * 100 for a, wt in zip(assets, w)}
    return {
        "ret": row["ret"],
        "vol": row["vol"],
        **pct_alloc,
    }


# ===== NEW: PROFILE CONSTRAINT APPLIER =====

def apply_profile_constraints(allocation: dict,
                              assets: list[str],
                              profile_name: str) -> dict:
    """
    After the optimizer picks weights, we can enforce a minimum "safety" slice
    if the profile asks for it. This keeps script useful for young people
    (their profile min=0) while making it safe for pre-retirement.
    """
    profile = INVESTOR_PROFILES.get(profile_name, INVESTOR_PROFILES["accumulator"])
    min_safety = profile["min_safety_pct"] * 100  # convert to same units as *_pct

    # define what counts as safety
    safety_assets = [a for a in assets if "bond" in a]

    current_safety = sum(allocation.get(f"{a}_pct", 0.0) for a in safety_assets)
    if current_safety >= min_safety or not safety_assets:
        return allocation  # nothing to do

    # we need to boost safety to min_safety, take proportionally from non-safety
    deficit = min_safety - current_safety

    # total non-safety
    risk_assets = [a for a in assets if a not in safety_assets]
    total_risk = sum(allocation.get(f"{a}_pct", 0.0) for a in risk_assets)

    if total_risk <= 0:
        # degenerate, just set safety to min and renorm
        for a in safety_assets:
            allocation[f"{a}_pct"] = min_safety / len(safety_assets)
        # renorm all
        s = sum(v for k, v in allocation.items() if k.endswith("_pct"))
        for k in allocation:
            if k.endswith("_pct"):
                allocation[k] = allocation[k] / s * 100
        return allocation

    # reduce risk assets proportionally
    factor = (total_risk - deficit) / total_risk
    for a in risk_assets:
        key = f"{a}_pct"
        allocation[key] = allocation[key] * factor

    # spread deficit across safety assets
    add_each = deficit / len(safety_assets)
    for a in safety_assets:
        key = f"{a}_pct"
        allocation[key] = allocation.get(key, 0.0) + add_each

    # final renorm to 100
    s = sum(v for k, v in allocation.items() if k.endswith("_pct"))
    for k in allocation:
        if k.endswith("_pct"):
            allocation[k] = allocation[k] / s * 100

    return allocation


def rolling_10y_allocations(returns: pd.DataFrame,
                            mode: str,
                            target_return: float,
                            target_vol: float,
                            n_portfolios: int,
                            rf: float,
                            profile_name: str) -> pd.DataFrame:
    assets = list(returns.columns)
    records = []

    for end_idx in range(WINDOW_MONTHS, len(returns)):
        sample = returns.iloc[end_idx - WINDOW_MONTHS:end_idx]

        # (we keep shrink calls to show place where you'd plug them in)
        _ = shrink_mean(sample.mean() * 12, assets)
        _ = shrink_cov(sample.cov() * 12)

        eff_df = long_only_frontier_vec(sample, n_portfolios=n_portfolios, rf=rf)

        chosen = pick_portfolio_from_eff(
            eff_df,
            assets=assets,
            mode=mode,
            target_return=target_return,
            target_vol=target_vol,
            rf=rf,
        )

        # NEW: enforce profile constraints here
        chosen = apply_profile_constraints(chosen, assets, profile_name)

        records.append({
            "date": returns.index[end_idx],
            **chosen,
        })

    return pd.DataFrame(records).set_index("date")


def make_policy_from_rolling(rolling_alloc_pct: pd.DataFrame,
                             band_width: float = 5.0,
                             use_quantiles: bool = True,
                             low_q: float = 0.30,
                             high_q: float = 0.70) -> tuple[dict, dict]:
    alloc_cols = [c for c in rolling_alloc_pct.columns if c.endswith("_pct")]
    policy_raw = rolling_alloc_pct[alloc_cols].median()
    policy = (policy_raw / policy_raw.sum()) * 100

    bands = {}
    for col in alloc_cols:
        if use_quantiles:
            lo_q = rolling_alloc_pct[col].quantile(low_q)
            hi_q = rolling_alloc_pct[col].quantile(high_q)
            lo = max(policy[col] - band_width, lo_q)
            hi = min(policy[col] + band_width, hi_q)
        else:
            lo, hi = policy[col] - band_width, policy[col] + band_width
        bands[col] = (lo, hi)

    return policy.to_dict(), bands


def check_rebalance(current_alloc: dict, bands: dict):
    print("\n=== Rebalancing check ===")
    for col, (lo, hi) in bands.items():
        current = current_alloc.get(col, 0.0)
        if current < lo or current > hi:
            status = "OUT OF BAND → consider rebalance"
        else:
            status = "within band"
        print(f"{col:15s} current={current:6.2f}%  target=({lo:6.2f}%, {hi:6.2f}%)  {status}")


def apply_rolling_weights_forward(returns: pd.DataFrame,
                                  rolling_alloc_pct: pd.DataFrame) -> pd.DataFrame:
    alloc = rolling_alloc_pct.copy()
    alloc = alloc.shift(1).dropna()
    aligned_rets = returns.reindex(alloc.index)
    port_rets = []
    for dt_idx in alloc.index:
        w = np.array([alloc.loc[dt_idx, f"{a}_pct"] for a in returns.columns]) / 100.0
        r = aligned_rets.loc[dt_idx].values @ w
        port_rets.append(r)
    port = pd.Series(port_rets, index=alloc.index, name="portfolio_ret")
    return port


def save_plot(fig, prefix: str):
    ts = int(dt.datetime.now().timestamp())
    fname = f"{prefix}_{ts}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved plot to {fname}")


# ===== NEW: STRESS TEST PRINTER =====

def run_stress_tests(policy: dict):
    print("\n=== Stress tests (one-period shocks) ===")
    # extract %s
    us_stock = policy.get("us_stock_pct", 0.0) / 100.0
    us_bond = policy.get("us_bond_pct", 0.0) / 100.0
    # tolerate missing assets
    for scen in STRESS_SCENARIOS:
        s_drop = scen["us_stock_shock"]
        b_drop = scen["us_bond_shock"]
        port_drop = us_stock * s_drop + us_bond * b_drop
        print(f"{scen['name']:35s}: portfolio change ≈ {port_drop*100:6.2f}%")


# =========================
# ========= MAIN ==========
# =========================

def main():
    print(f"[INFO] Using investor profile: {INVESTOR_PROFILE} "
          f"({INVESTOR_PROFILES[INVESTOR_PROFILE]['desc']})")

    print("[INFO] Downloading prices...")
    prices = fetch_prices(TICKERS, START_DATE, END_DATE)
    print("[INFO] Building monthly returns...")
    returns = build_monthly_returns(prices)
    # print("[INFO] Adding synthetic international bond...")
    # returns = add_synthetic_intl_bond(returns)

    if RUN_FRONTIERS:
        print("[INFO] Computing unconstrained frontier...")
        mu = returns.mean() * 12
        cov = returns.cov() * 12
        frontier = efficient_frontier(mu, cov, n_points=60)

        print("[INFO] Computing long-only frontier (vectorized)...")
        lo_frontier = long_only_frontier_vec(returns, n_portfolios=N_PORTFOLIOS, rf=RISK_FREE_RATE)

        fig = plt.figure(figsize=(7, 5))
        plt.plot(frontier["vol"], frontier["ret"], "b-", label="Unconstrained")
        plt.scatter(lo_frontier["vol"], lo_frontier["ret"], s=10, c="orange", label="Long-only")
        plt.xlabel("Volatility (σ)")
        plt.ylabel("Return (μ)")
        plt.title("Efficient Frontiers")
        plt.legend()
        plt.grid(True)
        save_plot(fig, "frontiers")

    if RUN_ROLLING or RUN_REBALANCE_CHECK or RUN_BACKTEST:
        print("[INFO] Running rolling 10-year allocation analysis...")
        rolling_alloc_pct = rolling_10y_allocations(
            returns,
            mode=ROLLING_MODE,
            target_return=TARGET_RETURN,
            target_vol=TARGET_VOL,
            n_portfolios=N_PORTFOLIOS,
            rf=RISK_FREE_RATE,
            profile_name=INVESTOR_PROFILE,
        )

        if RUN_ROLLING:
            alloc_cols = [c for c in rolling_alloc_pct.columns if c.endswith("_pct")]
            fig = plt.figure(figsize=(10, 5))
            rolling_alloc_pct[alloc_cols].plot(ax=plt.gca())
            plt.title(f"Rolling 10y allocations ({ROLLING_MODE}) + profile={INVESTOR_PROFILE}")
            plt.ylabel("Allocation (%)")
            plt.ylim(0, 100)
            plt.grid(True)
            save_plot(fig, "rolling_allocs")

        policy, bands = make_policy_from_rolling(
            rolling_alloc_pct,
            band_width=FIXED_BAND_WIDTH,
            use_quantiles=USE_QUANTILE_BANDS,
            low_q=LOW_Q,
            high_q=HIGH_Q,
        )

        print("\n=== Policy allocation (normalized from medians, profile-adjusted) ===")
        for k, v in policy.items():
            print(f"{k:15s}: {v:6.2f}%")

        if RUN_REBALANCE_CHECK:
            check_rebalance(CURRENT_ALLOCATION_PCT, bands)

        if RUN_STRESS_TESTS:
            run_stress_tests(policy)

        if RUN_BACKTEST:
            print("[INFO] Running simple forward-apply backtest...")
            port = apply_rolling_weights_forward(returns, rolling_alloc_pct)
            growth = (1 + port).cumprod()
            fig = plt.figure(figsize=(8, 4))
            plt.plot(growth.index, growth.values, label="Rolling policy portfolio")
            plt.title("Rolling policy portfolio growth (simple backtest)")
            plt.grid(True)
            plt.legend()
            save_plot(fig, "backtest_growth")


if __name__ == "__main__":
    main()
