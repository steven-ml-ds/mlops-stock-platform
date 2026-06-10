"""EDA gate before feature engineering is finalized.

Answers four questions and writes evidence to charts/:
1. Data quality  — calendar alignment, NaNs, suspicious jumps
2. Target        — distribution, class balance, autocorrelation structure
3. Signal        — information coefficient (IC) scan of candidate features
4. Stationarity  — price levels vs returns, justifying ratio-only features

Run: uv run python scripts/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from platform_core.config import PROJECT_ROOT, TICKERS
from platform_core.data import load_cached

CHARTS = PROJECT_ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

report: list[str] = []


def say(line: str = "") -> None:
    print(line)
    report.append(line)


# ---------------------------------------------------------------- load
frames = {t: load_cached(t) for t in TICKERS}
spy = load_cached("SPY")
vix = load_cached("^VIX")
tnx = load_cached("^TNX")
irx = load_cached("^IRX")

# ---------------------------------------------------- 1. data quality
say("=" * 70)
say("1. DATA QUALITY")
say("=" * 70)
base_cal = set(spy.index)
for name, df in {**frames, "^VIX": vix, "^TNX": tnx, "^IRX": irx}.items():
    cal = set(df.index)
    extra, missing = sorted(cal - base_cal), sorted(base_cal - cal)
    nans = int(df.isna().sum().sum())
    bad_px = int((df[["open", "high", "low", "close"]] <= 0).sum().sum())
    say(
        f"{name:8s} rows={len(df):5d}  NaN={nans:3d}  px<=0={bad_px}  "
        f"vs SPY: +{len(extra)} extra / -{len(missing)} missing days"
    )
    if extra[:3]:
        say(f"         extra days e.g. {[d.date() for d in extra[:3]]}")
    if missing[:3]:
        say(f"         missing days e.g. {[d.date() for d in missing[:3]]}")

say()
say("Largest |1d return| per ticker (split artifacts would show as ~±50%):")
for name, df in frames.items():
    ret = df["close"].pct_change()
    worst = ret.abs().nlargest(3)
    pretty = ", ".join(f"{d.date()}:{ret[d]:+.1%}" for d in worst.index)
    say(f"{name:8s} {pretty}")

# ---------------------------------------------------------- 2. target
say()
say("=" * 70)
say("2. TARGET: next-day log return")
say("=" * 70)
rets = {t: np.log(df["close"]).diff().dropna() for t, df in frames.items()}
fig, axes = plt.subplots(1, len(TICKERS), figsize=(15, 4))
for ax, (name, r) in zip(axes, rets.items()):
    up = (r > 0).mean()
    say(
        f"{name:8s} mean={r.mean():+.5f} std={r.std():.4f} "
        f"skew={stats.skew(r):+.2f} kurt={stats.kurtosis(r):.1f} (normal=0)  "
        f"up-days={up:.1%}"
    )
    ax.hist(r, bins=80, density=True, alpha=0.7)
    x = np.linspace(r.min(), r.max(), 200)
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), "r--", lw=1)
    ax.set_title(f"{name} daily log returns")
fig.tight_layout()
fig.savefig(CHARTS / "01_return_distribution.png", dpi=110)

say()
say("Return autocorrelation (signal for lag features) & |return| autocorr (vol clustering):")
fig, axes = plt.subplots(2, len(TICKERS), figsize=(15, 7))
for i, (name, r) in enumerate(rets.items()):
    ac_r = [r.autocorr(lag) for lag in range(1, 11)]
    ac_abs = [r.abs().autocorr(lag) for lag in range(1, 22)]
    n = len(r)
    ci = 1.96 / np.sqrt(n)
    sig = [f"lag{l}:{a:+.3f}" for l, a in enumerate(ac_r, 1) if abs(a) > ci]
    say(f"{name:8s} ret autocorr beyond 95% band: {sig or 'none'}")
    say(f"{name:8s} |ret| autocorr lag1/5/21: "
        f"{ac_abs[0]:+.3f}/{ac_abs[4]:+.3f}/{ac_abs[20]:+.3f}  (vol clustering)")
    axes[0, i].bar(range(1, 11), ac_r)
    axes[0, i].axhline(ci, color="r", ls="--", lw=0.8)
    axes[0, i].axhline(-ci, color="r", ls="--", lw=0.8)
    axes[0, i].set_title(f"{name} ret ACF")
    axes[1, i].bar(range(1, 22), ac_abs, color="darkorange")
    axes[1, i].axhline(ci, color="r", ls="--", lw=0.8)
    axes[1, i].set_title(f"{name} |ret| ACF")
fig.tight_layout()
fig.savefig(CHARTS / "02_autocorrelation.png", dpi=110)

# ------------------------------------------------------ 3. signal scan
say()
say("=" * 70)
say("3. SIGNAL EXISTENCE: candidate-feature IC scan (pooled Spearman vs fwd ret)")
say("=" * 70)


def candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for lag in (1, 2, 3, 5, 10, 21):
        f[f"ret_lag{lag}"] = r.shift(lag - 1)  # ret over day t-lag+1..t window end
    for w in (5, 21, 63):
        f[f"mom_{w}"] = np.log(c / c.shift(w))
        f[f"vol_{w}"] = r.rolling(w).std()
    f["gap"] = np.log(o / c.shift(1))
    f["intraday"] = np.log(c / o)
    f["px_ma20"] = c / c.rolling(20).mean() - 1
    f["ma5_ma20"] = c.rolling(5).mean() / c.rolling(20).mean() - 1
    f["dist_hi252"] = c / c.rolling(252).max() - 1
    f["parkinson_21"] = np.sqrt(
        (np.log(h / l) ** 2).rolling(21).mean() / (4 * np.log(2))
    )
    f["vol_ratio"] = f["vol_5"] / f["vol_63"]
    f["volu_z20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    f["range_pct"] = (h - l) / c
    f["close_pos"] = ((c - l) / (h - l)).where(h > l)
    f["dow"] = df.index.dayofweek
    return f


spy_ret = np.log(spy["close"]).diff()
vix_chg = vix["close"].diff()
curve = (tnx["close"] - irx["close"])

ic_rows = []
pooled_X, pooled_y = [], []
for name, df in frames.items():
    f = candidate_features(df)
    f["spy_ret_lag1"] = spy_ret.reindex(f.index)
    f["vix_level"] = vix["close"].reindex(f.index)
    f["vix_chg"] = vix_chg.reindex(f.index)
    f["tnx_chg"] = tnx["close"].diff().reindex(f.index)
    f["curve_slope"] = curve.reindex(f.index)
    y = np.log(df["close"]).diff().shift(-1)  # next-day return
    pooled_X.append(f)
    pooled_y.append(y)

X = pd.concat(pooled_X)
y = pd.concat(pooled_y)
mask = y.notna()
for col in X.columns:
    valid = mask & X[col].notna()
    ic, p = stats.spearmanr(X.loc[valid, col], y[valid])
    ic_rows.append((col, ic, p, int(valid.sum())))

ic_df = pd.DataFrame(ic_rows, columns=["feature", "ic", "p", "n"]).sort_values(
    "ic", key=abs, ascending=False
)
say(f"{'feature':15s} {'IC':>8s} {'p-value':>9s}   verdict")
for _, row in ic_df.iterrows():
    verdict = "**" if row.p < 0.01 else ("*" if row.p < 0.05 else "")
    say(f"{row.feature:15s} {row.ic:+8.4f} {row.p:9.4f}   {verdict}")

fig, ax = plt.subplots(figsize=(9, 8))
colors = ["tab:green" if p < 0.05 else "tab:gray" for p in ic_df.p]
ax.barh(ic_df.feature[::-1], ic_df.ic[::-1], color=colors[::-1])
ax.set_title("Pooled Spearman IC vs next-day return (green: p<0.05)")
ax.axvline(0, color="k", lw=0.8)
fig.tight_layout()
fig.savefig(CHARTS / "03_ic_scan.png", dpi=110)

# --------------------------------------------------- 4. collinearity
say()
say("=" * 70)
say("4. COLLINEARITY: |corr| > 0.85 pairs (candidates to drop)")
say("=" * 70)
corr = X.corr()
seen = set()
for a in corr.columns:
    for b in corr.columns:
        if a < b and abs(corr.loc[a, b]) > 0.85 and (a, b) not in seen:
            seen.add((a, b))
            say(f"{a:15s} ~ {b:15s}  corr={corr.loc[a, b]:+.2f}")
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
fig.colorbar(im)
ax.set_title("Candidate feature correlation")
fig.tight_layout()
fig.savefig(CHARTS / "04_feature_corr.png", dpi=110)

# --------------------------------------------------- 5. stationarity
say()
say("=" * 70)
say("5. STATIONARITY: price level vs returns (rolling mean drift)")
say("=" * 70)
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for name, df in frames.items():
    axes[0].plot(df.index, df["close"] / df["close"].iloc[0], label=name)
    axes[1].plot(rets[name].rolling(63).mean().index,
                 rets[name].rolling(63).mean(), label=name)
axes[0].set_title("Price level (normalized) — non-stationary, unusable as raw feature")
axes[1].set_title("63d rolling mean of log returns — mean-reverting around 0")
axes[1].axhline(0, color="k", lw=0.8)
for ax in axes:
    ax.legend()
fig.tight_layout()
fig.savefig(CHARTS / "05_stationarity.png", dpi=110)
for name, df in frames.items():
    px = df["close"]
    half1, half2 = px[: len(px) // 2], px[len(px) // 2 :]
    r1, r2 = rets[name][: len(rets[name]) // 2], rets[name][len(rets[name]) // 2 :]
    say(
        f"{name:8s} price mean 1st/2nd half: {half1.mean():8.1f} / {half2.mean():8.1f}"
        f"   ret mean: {r1.mean():+.5f} / {r2.mean():+.5f}"
    )

(CHARTS / "eda_summary.txt").write_text("\n".join(report))
say()
say(f"charts + summary written to {CHARTS}/")
sys.exit(0)
