#!/usr/bin/env python3
"""
Shared paths, metrics and plotting helpers for the classification analysis.

Everything is implemented on numpy/scipy so the analysis runs without
statsmodels, which is not installed in the target environment. Nothing here
writes outside analysis/.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]

# Source and destination are overridable so the same analysis code can produce
# both the superseded DenseNet121 results and the locked EfficientNet-B0 ones,
# without duplicating any statistics. Defaults reproduce the original run
# exactly, so scripts invoked with no environment set behave as before.
#
#   SPL_PRED_DIR   directory of per-case probability files
#   SPL_ARCH       architecture whose files to read
#   SPL_RESULTS    directory for generated tables
#   SPL_FIGURES    directory for generated figures
#
# See docs/REPRODUCE.md for which artifact each combination produces.
PRED_DIR = Path(os.environ.get(
    "SPL_PRED_DIR", REPO / "binary_classification" / "predictions_tight"))
ARCH = os.environ.get("SPL_ARCH", "densenet121")
OUT = REPO / "analysis"
RESULTS = Path(os.environ.get("SPL_RESULTS", OUT / "results"))
FIGURES = Path(os.environ.get("SPL_FIGURES", OUT / "figures"))
REPORT = OUT / "report"

BOOT_SEED = 20260904
N_BOOT = 2000

SPLITS = ["train", "internal_val", "external_test1", "external_test2"]
EXPECTED_N = {"train": 600, "internal_val": 257, "external_test1": 108, "external_test2": 94}
EXPECTED_POSNEG = {
    "internal_val": (155, 102),
    "external_test1": (62, 46),
    "external_test2": (67, 27),
    "train": None,  # no pre-specified counts; reported as observed
}

ARCHS = [
    "densenet121", "densenet201",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
    "inception_v3", "resnet18", "resnet50", "resnet101", "vgg19",
]

VARIANTS = ["gt", "model"]


def ensure_dirs() -> None:
    for d in (RESULTS, FIGURES, REPORT):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def pred_path(arch: str, split: str, variant: str) -> Path:
    suffix = "" if variant == "gt" else f"_{variant}"
    return PRED_DIR / arch / f"{split}{suffix}_seed67_probs.txt"


def load_probs(arch: str, split: str, variant: str) -> pd.DataFrame:
    """Load a per-case probability file.

    The producing script may append a trailing '# missing entries' block after a
    blank line, so parsing stops at the first non-conforming row.
    """
    p = pred_path(arch, split, variant)
    rows = []
    with open(p) as fh:
        header = fh.readline().strip()
        assert header == "case_id,label,prob_malignant,prob_benign", f"unexpected header in {p}: {header}"
        for line in fh:
            line = line.strip()
            if not line:
                break
            if line.startswith("#"):
                break
            parts = line.split(",")
            if len(parts) != 4:
                break
            rows.append((parts[0], int(parts[1]), float(parts[2]), float(parts[3])))
    df = pd.DataFrame(rows, columns=["case_id", "label", "prob_malignant", "prob_benign"])
    return df


def get_y_p(arch: str, split: str, variant: str):
    df = load_probs(arch, split, variant)
    return df["label"].to_numpy(int), df["prob_malignant"].to_numpy(float)


# ----------------------------------------------------------------------------
# Wilson score interval
# ----------------------------------------------------------------------------

def wilson_ci(p_hat: float, n: int, alpha: float = 0.05):
    """Wilson score interval for a proportion given the point estimate and n.

    p_hat is taken as a continuous value rather than x/n, so an interval can be
    computed for a proportion paired with an arbitrary denominator.
    """
    if n <= 0 or not np.isfinite(p_hat):
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_from_counts(x: int, n: int, alpha: float = 0.05):
    return wilson_ci(x / n if n else np.nan, n, alpha)


# ----------------------------------------------------------------------------
# Point metrics
# ----------------------------------------------------------------------------

def confusion(y: np.ndarray, p: np.ndarray, thr: float):
    pred = (p >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def _safe_div(a, b):
    return a / b if b > 0 else np.nan


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    tp, fp, tn, fn = confusion(y, p, thr)
    n = tp + fp + tn + fn
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n": n,
        "n_pos": tp + fn,
        "n_neg": tn + fp,
        "n_pred_pos": tp + fp,
        "n_pred_neg": tn + fn,
        "accuracy": _safe_div(tp + tn, n),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "youden_j": _safe_div(tp, tp + fn) + _safe_div(tn, tn + fp) - 1,
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
    }


def rank_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "auprc": np.nan}
    return {"auc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}


# Denominator that each threshold metric is a proportion of.
METRIC_DENOM_KEY = {
    "accuracy": "n",
    "sensitivity": "n_pos",
    "specificity": "n_neg",
    "ppv": "n_pred_pos",
    "npv": "n_pred_neg",
}


# ----------------------------------------------------------------------------
# Threshold selection policies (fit on a tuning cohort only)
# ----------------------------------------------------------------------------

def _candidate_thresholds(p: np.ndarray, prefer: float = 0.5) -> np.ndarray:
    probs = np.unique(p)
    return np.concatenate(([probs.min() - 1e-6], probs, [probs.max() + 1e-6, prefer]))


def threshold_max_accuracy(y: np.ndarray, p: np.ndarray, prefer: float = 0.5) -> float:
    """Re-implementation of binary_classification/infer_probs_tight.py::_best_threshold_from_rows."""
    best_thr, best_acc = prefer, -np.inf
    for thr in _candidate_thresholds(p, prefer):
        acc = ((p >= thr).astype(int) == y).mean()
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) < 1e-12 and abs(thr - prefer) < abs(best_thr - prefer)):
            best_acc, best_thr = acc, thr
    return float(best_thr)


def threshold_youden(y: np.ndarray, p: np.ndarray, prefer: float = 0.5) -> float:
    """Threshold maximising Youden's J = sensitivity + specificity - 1."""
    best_thr, best_j = prefer, -np.inf
    for thr in _candidate_thresholds(p, prefer):
        m = threshold_metrics(y, p, thr)
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best_j + 1e-12 or (abs(j - best_j) < 1e-12 and abs(thr - prefer) < abs(best_thr - prefer)):
            best_j, best_thr = j, thr
    return float(best_thr)


# ----------------------------------------------------------------------------
# Stratified case-level bootstrap
# ----------------------------------------------------------------------------

def stratified_boot_indices(y: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Yield n_boot index arrays resampling cases with replacement within class."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=pos.size, replace=True),
            rng.choice(neg, size=neg.size, replace=True),
        ])
        yield idx


def bootstrap_ci(y, p, fn, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = 0.05):
    """Percentile bootstrap CI for a scalar statistic fn(y, p)."""
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    for idx in stratified_boot_indices(y, n_boot, seed):
        v = fn(y[idx], p[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan, 0)
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi), len(vals))


# ----------------------------------------------------------------------------
# DeLong
# ----------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def delong_components(y: np.ndarray, preds: np.ndarray):
    """Fast DeLong (Sun & Xu 2014).

    preds: shape (k, n) matrix of k predictors on the same n cases.
    Returns (aucs, S) where S is the k x k covariance matrix of the AUCs.
    """
    y = np.asarray(y).astype(int)
    preds = np.atleast_2d(np.asarray(preds, dtype=float))
    order = np.argsort(-y)  # positives first
    y = y[order]
    preds = preds[:, order]
    m = int(np.sum(y == 1))
    n = int(np.sum(y == 0))
    k = preds.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n           # V10, one per positive case
    v10 = 1.0 - (tz[:, m:] - ty) / m     # V01, one per negative case
    sx = np.cov(v01)
    sy = np.cov(v10)
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    S = sx / m + sy / n
    return aucs, S


def delong_auc_ci(y, p, alpha: float = 0.05):
    """DeLong 95% CI for a single AUC, computed on the logit scale then
    back-transformed so the interval stays inside [0, 1]."""
    aucs, S = delong_components(np.asarray(y), np.asarray(p)[None, :])
    auc = float(aucs[0])
    var = float(S[0, 0])
    se = float(np.sqrt(max(var, 0.0)))
    z = stats.norm.ppf(1 - alpha / 2)
    # Normal-approximation interval (the classical DeLong interval).
    lo_n, hi_n = auc - z * se, auc + z * se
    # Logit-transformed interval (recommended for small samples / high AUC).
    if 0 < auc < 1 and se > 0:
        eta = np.log(auc / (1 - auc))
        se_eta = se / (auc * (1 - auc))
        lo_l = 1 / (1 + np.exp(-(eta - z * se_eta)))
        hi_l = 1 / (1 + np.exp(-(eta + z * se_eta)))
    else:
        lo_l, hi_l = np.nan, np.nan
    return {
        "auc": auc, "se": se,
        "delong_lo": max(0.0, lo_n), "delong_hi": min(1.0, hi_n),
        "delong_logit_lo": lo_l, "delong_logit_hi": hi_l,
    }


def delong_test(y, p1, p2):
    """DeLong test for two correlated ROC AUCs on the same cases."""
    aucs, S = delong_components(np.asarray(y), np.vstack([np.asarray(p1), np.asarray(p2)]))
    diff = float(aucs[0] - aucs[1])
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    if se == 0:
        z = np.nan if diff != 0 else 0.0
        pval = np.nan if diff != 0 else 1.0
    else:
        z = diff / se
        pval = float(2 * stats.norm.sf(abs(z)))
    ci_z = stats.norm.ppf(0.975)
    return {
        "auc1": float(aucs[0]), "auc2": float(aucs[1]),
        "auc_diff": diff, "se_diff": se, "z": float(z), "p_value": pval,
        "diff_lo": diff - ci_z * se, "diff_hi": diff + ci_z * se,
    }


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------

EPS = 1e-6


def logit(p, eps: float = EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def _newton_logistic(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None,
                     max_iter: int = 200, tol: float = 1e-10):
    """Plain Newton-Raphson / IRLS fit of a logistic model.

    linear predictor = X @ beta + offset
    Returns beta. Raises no exception on separation; it simply stops at max_iter.
    """
    n, k = X.shape
    if offset is None:
        offset = np.zeros(n)
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ beta + offset
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        grad = X.T @ (y - mu)
        H = (X * w[:, None]).T @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def calibration_intercept(y, p):
    """Calibration-in-the-large: intercept of y ~ 1 + offset(logit(p)) (slope fixed at 1)."""
    y = np.asarray(y, dtype=float)
    lp = logit(p)
    X = np.ones((len(y), 1))
    beta = _newton_logistic(X, y, offset=lp)
    return float(beta[0])


def calibration_slope(y, p):
    """Slope b from the standard fit y ~ a + b * logit(p)."""
    y = np.asarray(y, dtype=float)
    lp = logit(p)
    X = np.column_stack([np.ones(len(y)), lp])
    beta = _newton_logistic(X, y)
    return float(beta[1])


def calibration_slope_and_intercept_joint(y, p):
    y = np.asarray(y, dtype=float)
    lp = logit(p)
    X = np.column_stack([np.ones(len(y)), lp])
    beta = _newton_logistic(X, y)
    return float(beta[0]), float(beta[1])


def calibration_curve_quantile(y, p, n_bins: int = 10):
    """Quantile-binned calibration curve. Returns a DataFrame with counts."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    qs = np.unique(qs)
    idx = np.digitize(p, qs[1:-1], right=True)
    rows = []
    for b in range(len(qs) - 1):
        sel = idx == b
        k = int(sel.sum())
        if k == 0:
            continue
        n_ev = int(y[sel].sum())
        obs = n_ev / k
        lo, hi = wilson_from_counts(n_ev, k)
        rows.append({
            "bin": b + 1,
            "bin_lo": float(qs[b]), "bin_hi": float(qs[b + 1]),
            "n": k, "n_events": n_ev,
            "mean_pred": float(p[sel].mean()),
            "obs_freq": float(obs),
            "obs_lo_wilson": lo, "obs_hi_wilson": hi,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Decision curve analysis
# ----------------------------------------------------------------------------

def net_benefit_model(y, p, pt: float):
    """NB_model(pt) = TP/n - (FP/n) * (pt / (1 - pt)), classifying p >= pt as positive."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    pred = p >= pt
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    if pt >= 1.0:
        return np.nan
    return tp / n - (fp / n) * (pt / (1 - pt))


def net_benefit_treat_all(y, pt: float):
    """NB_all(pt) = prev - (1 - prev) * (pt / (1 - pt)); decreasing in pt, zero at pt = prev."""
    y = np.asarray(y)
    prev = float(np.mean(y == 1))
    if pt >= 1.0:
        return np.nan
    return prev - (1 - prev) * (pt / (1 - pt))


# ----------------------------------------------------------------------------
# Plot style: colour-blind safe, sans-serif, vector-friendly
# ----------------------------------------------------------------------------

# Okabe-Ito colour-blind-safe palette
CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "skyblue": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#666666",
}


def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.8,
    })
    return plt


def save_fig(fig, stem: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png")
    print(f"  wrote figures/{stem}.pdf and .png")


PRETTY_SPLIT = {
    "train": "Training cohort",
    "internal_val": "Internal validation (tuning) cohort",
    "external_test1": "External Test 1",
    "external_test2": "External Test 2",
}
