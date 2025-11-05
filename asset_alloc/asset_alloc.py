#!/usr/bin/env python3
"""
Portfolio research script:
- download prices
- build monthly returns
- construct synthetic international bond
- mean/cov + unconstrained frontier
- long-only frontier (MC)
- rolling 10-year target-return allocations
- derive policy bands (normalized)
- compare to current allocation
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# =========================
# ===== USER INPUTS =======
# =========================

START_DATE = "1995-01-01"
END_DATE = dt.date.today().strftime("%Y-%m-%d")

# tickers: map friendly name -> yahoo ticker
TICKERS = {
    "us_stock": "^GSPC",   # or "^SP500TR" if your yfinance returns it cleanly
    "intl_stock": "EFA",
    "us_bond": "AGG",
}

# choose what to run
RUN_FRONTIERS = True
RUN_ROLLING = True
RUN_REBALANCE_CHECK = True

# target annual return for target-based portfolios (e.g. 0.08 = 8%)
TARGET_RETURN = 0.08

# band width for rebalancing decision (in percentage points)
BAND_WIDTH = 5.0

# your CURRENT allocation (in %). adjust to your real portfolio.
CURRENT_ALLOCATION_PCT = {
    "us_stock_pct": 45.0,
    "intl_stock_pct": 15.0,
    "us_bond_pct": 30.0,
    "intl_bond_pct": 10.0,
}

# risk-free for Sharpe
# 3-month US treasury bill yield or 10-year treasury yield are common choices
RISK_FREE_RATE = 0.04

# random portfolios per run (for long-only + rolling)
N_PORTFOLIOS = 4000

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
            auto_adjust=False,   # make explicit to silence warning
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
    intl_bond = (
        returns["us_bond"] * 0.8
        + np.random.normal(0, 0.002, size=len(returns))
    )
    returns["intl_bond"] = intl_bond
    return returns[["us_stock", "intl_stock", "us_bond", "intl_bond"]]


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 60) -> pd.DataFrame:
    """Unconstrained Markowitz frontier (can short)."""
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
    return pd.DataFrame(rows, columns=["vol", "ret", "weights"])


def long_only_frontier(returns: pd.DataFrame,
                       n_portfolios: int = 20000) -> pd.DataFrame:
    """Monte Carlo long-only frontier."""
    mu = returns.mean() * 12
    cov = returns.cov() * 12
    assets = returns.columns
    ports = []
    for _ in range(n_portfolios):
        w = np.random.rand(len(assets))
        w /= w.sum()
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(w @ cov.values @ w)
        ports.append((port_vol, port_ret, w))
    df = pd.DataFrame(ports, columns=["vol", "ret", "weights"]).sort_values("vol").reset_index(drop=True)
    # pareto filter
    efficient = []
    best_ret = -1
    for _, row in df.iterrows():
        if row["ret"] > best_ret:
            efficient.append(row)
            best_ret = row["ret"]
    return pd.DataFrame(efficient)


def rolling_10y_target_allocations(returns: pd.DataFrame,
                                   target_return: float,
                                   n_portfolios: int = 4000) -> pd.DataFrame:
    """For each month, look back 10 years, find long-only portfolio nearest to target return."""
    window_months = 120
    assets = returns.columns
    records = []

    for end_idx in range(window_months, len(returns)):
        sample = returns.iloc[end_idx - window_months:end_idx]
        mu_w = sample.mean() * 12
        cov_w = sample.cov() * 12

        # simulate
        ports = []
        for _ in range(n_portfolios):
            w = np.random.rand(len(assets))
            w /= w.sum()
            port_ret = np.dot(w, mu_w)
            port_vol = np.sqrt(w @ cov_w.values @ w)
            ports.append((port_vol, port_ret, w))

        df = pd.DataFrame(ports, columns=["vol", "ret", "weights"]).sort_values("vol")
        # pareto filter
        efficient = []
        best_ret = -1
        for _, row in df.iterrows():
            if row["ret"] > best_ret:
                efficient.append(row)
                best_ret = row["ret"]
        eff_df = pd.DataFrame(efficient)

        # pick closest to target
        idx = (eff_df["ret"] - target_return).abs().idxmin()
        chosen = eff_df.loc[idx]
        pct_alloc = {f"{a}_pct": wt * 100 for a, wt in zip(assets, chosen["weights"])}

        records.append({
            "date": returns.index[end_idx],
            "ret": chosen["ret"],
            "vol": chosen["vol"],
            **pct_alloc
        })

    return pd.DataFrame(records).set_index("date")


def make_policy_from_rolling(rolling_alloc_pct: pd.DataFrame,
                             band_width: float = 5.0) -> tuple[dict, dict]:
    alloc_cols = [c for c in rolling_alloc_pct.columns if c.endswith("_pct")]
    policy_raw = rolling_alloc_pct[alloc_cols].median()
    # normalize to sum to 100
    policy = (policy_raw / policy_raw.sum()) * 100
    # build bands
    bands = {
        col: (policy[col] - band_width, policy[col] + band_width)
        for col in alloc_cols
    }
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


# =========================
# ========= MAIN ==========
# =========================

def main():
    # 1. fetch & prep
    print("[INFO] Downloading prices...")
    prices = fetch_prices(TICKERS, START_DATE, END_DATE)
    print("[INFO] Building monthly returns...")
    returns = build_monthly_returns(prices)
    print("[INFO] Adding synthetic international bond...")
    returns = add_synthetic_intl_bond(returns)

    if RUN_FRONTIERS:
        print("[INFO] Computing unconstrained frontier...")
        mu = returns.mean() * 12
        cov = returns.cov() * 12
        frontier = efficient_frontier(mu, cov, n_points=60)
        print("[INFO] Computing long-only frontier...")
        lo_frontier = long_only_frontier(returns, n_portfolios=N_PORTFOLIOS)

        # add Sharpe
        frontier["sharpe"] = (frontier["ret"] - RISK_FREE_RATE) / frontier["vol"]
        lo_frontier["sharpe"] = (lo_frontier["ret"] - RISK_FREE_RATE) / lo_frontier["vol"]

        # plot both
        plt.figure(figsize=(7,5))
        plt.plot(frontier["vol"], frontier["ret"], "b-", label="Unconstrained")
        plt.scatter(lo_frontier["vol"], lo_frontier["ret"], s=10, c="orange", label="Long-only")
        plt.xlabel("Volatility (σ)")
        plt.ylabel("Return (μ)")
        plt.title("Efficient Frontiers")
        plt.legend()
        plt.grid(True)
        plt.show()

    if RUN_ROLLING or RUN_REBALANCE_CHECK:
        print("[INFO] Running rolling 10-year target-return analysis...")
        rolling_alloc_pct = rolling_10y_target_allocations(
            returns,
            target_return=TARGET_RETURN,
            n_portfolios=N_PORTFOLIOS
        )

        # optional plot of drift
        if RUN_ROLLING:
            alloc_cols = [c for c in rolling_alloc_pct.columns if c.endswith("_pct")]
            rolling_alloc_pct[alloc_cols].plot(figsize=(10,5))
            plt.title(f"Rolling 10y allocations for target ≈ {TARGET_RETURN:.0%}")
            plt.ylabel("Allocation (%)")
            plt.ylim(0, 100)
            plt.grid(True)
            plt.show()

        # build policy and bands
        policy, bands = make_policy_from_rolling(rolling_alloc_pct, band_width=BAND_WIDTH)

        print("\n=== Policy allocation (normalized from medians) ===")
        for k, v in policy.items():
            print(f"{k:15s}: {v:6.2f}%")

        if RUN_REBALANCE_CHECK:
            check_rebalance(CURRENT_ALLOCATION_PCT, bands)


if __name__ == "__main__":
    main()
