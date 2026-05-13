"""
csr_volume_entropy_null.py
==========================
Null benchmark for the volume-entropy relationship reported in
Empirical Study (Section 5.4).

Addresses Reviewer Comment 3.2: the observed correlation between
volume and salience entropy is partially induced by the arithmetic
of a bounded Likert support, independent of any psychological
process. This script characterizes the baseline expectation
    b_0(eta) = E[H | eta]
under a row-volume-preserving null with cell probabilities pinned
at the observed item marginals, then reports the directional
residual
    H_tilde_i = H_i - b_0(eta_i)
on the empirical data.

The empirical pipeline (CSR fit with ordinal_transform="expected",
gamma=0, normalized entropy of fitted salience) is mirrored exactly
on the null so the comparison is apples-to-apples.

Pipeline
--------
  1. Fit CSR on empirical extraversion data (same settings as S5).
  2. Compute empirical item marginals pi_j (proportional to mean(y_j)-1)
     and row-total moments (T_mean, T_sd) from the data.
  3. Generate a large null dataset via gen_rowvolume_null with
     item_marginals=pi and matched row-total moments.
  4. Fit CSR on the null with identical settings; record (eta_null, H_null).
  5. Bin (eta_null, H_null) to estimate b_0(eta) as bin-mean H.
  6. Case-bootstrap the null B times for pointwise 95% bands on b_0(eta).
  7. Apply b_0(.) to empirical eta_i; compute H_tilde = H - b_0(eta).
  8. Report r(eta, H), r(eta, H_tilde), Var(H_tilde)/Var(H), and the
     fraction of Var(H) absorbed by b_0.

Outputs
-------
  {prefix}_per_person.csv : empirical eta, H, b_0(eta), H_tilde
  {prefix}_smoother.csv   : b_0(eta) on a regular grid, with 95% bands
  {prefix}_summary.csv    : single-row summary of all statistics

Run
---
  python csr_volume_entropy_null.py                       # default
  python csr_volume_entropy_null.py --quick               # smoke test
  python csr_volume_entropy_null.py --N_null 50000 --n_boot 200

Requires: csr_main.py, csr_empirical_study.py, and csr_section4x_simulation.py
(or wherever gen_rowvolume_null lives) in the same directory.
Author: Jonathan Lee
"""
from __future__ import annotations

import argparse
import time
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from csr_main import fit_csr, safe_corr, compute_entropy
from csr_empirical_study import (
    EXTRAVERSION_ITEMS,
    download_big5_data,
    load_and_prepare_data,
)

# gen_rowvolume_null lives in the Section 4.2 simulation module.
# Adjust the import below if the module name on your machine differs.
try:
    from csr_section4x_simulation import gen_rowvolume_null
except ImportError:  # pragma: no cover - fallback to alternate filenames
    try:
        from csr_section42_simulation import gen_rowvolume_null
    except ImportError:
        from csr_section42_hetero_baseline import gen_rowvolume_null


# ===========================================================
# Configuration
# ===========================================================

RANDOM_SEED = 42
DEFAULT_N_NULL = 20_000
DEFAULT_N_BOOT = 100
DEFAULT_N_BINS = 80


# ===========================================================
# Empirical item marginals
# ===========================================================

def compute_empirical_pi(y: np.ndarray) -> np.ndarray:
    """Item marginals matched to gen_rowvolume_null's allocation scheme.

    Under gen_rowvolume_null with K-Likert support {1,..,K}, item j
    receives counts_ij additional units above the floor of 1, with
    E[counts_ij | T_i] proportional to pi_j. Matching empirical item
    means therefore requires pi_j proportional to (mean(y_j) - 1).
    """
    excess = y.mean(axis=0) - 1.0
    excess = np.clip(excess, 1e-12, None)
    return excess / excess.sum()


# ===========================================================
# CSR fit + normalized entropy
# ===========================================================

def fit_csr_and_metrics(
    y: np.ndarray,
    ordinal_transform: str = "expected",
    gamma: float = 0.0,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit CSR with the empirical-study settings; return (eta_hat, H_norm)."""
    res = fit_csr(
        y,
        ordinal_transform=ordinal_transform,
        gamma=gamma,
        max_iter=200,
        tol=1e-6,
        verbose=False,
        seed=seed,
    )
    J = y.shape[1]
    H_norm = compute_entropy(res.S) / np.log(J)
    return res.eta, H_norm


# ===========================================================
# Baseline curve: bin-and-mean
# ===========================================================

def baseline_curve(
    eta: np.ndarray,
    H: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    eta_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Estimate b_0(eta) as the mean H within eta bins.

    Empty bins are filled by linear interpolation between adjacent
    occupied centers so the curve is defined on the full range.
    """
    if eta_range is None:
        lo, hi = float(eta.min()), float(eta.max())
    else:
        lo, hi = eta_range
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(eta, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        mask = bin_idx == k
        n_k = int(mask.sum())
        counts[k] = n_k
        if n_k > 0:
            means[k] = float(H[mask].mean())

    valid = ~np.isnan(means)
    if valid.sum() < 2:
        raise ValueError("Too few occupied bins to estimate baseline curve.")
    means_filled = np.interp(centers, centers[valid], means[valid])

    return {
        "centers": centers,
        "means": means_filled,
        "counts": counts,
        "edges": edges,
    }


def evaluate_baseline(
    eta_query: np.ndarray,
    curve: Dict[str, np.ndarray],
) -> np.ndarray:
    """Linear interpolation of b_0(eta) at queried eta values."""
    return np.interp(eta_query, curve["centers"], curve["means"])


# ===========================================================
# Bootstrap bands
# ===========================================================

def bootstrap_bands(
    eta_null: np.ndarray,
    H_null: np.ndarray,
    eta_range: Tuple[float, float],
    n_bins: int = DEFAULT_N_BINS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Pointwise 95% bands for b_0(eta) via case resampling of the null."""
    rng = np.random.default_rng(seed)
    N = eta_null.shape[0]
    boot = np.zeros((n_boot, n_bins))
    t0 = time.time()
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        curve_b = baseline_curve(
            eta_null[idx], H_null[idx], n_bins=n_bins, eta_range=eta_range,
        )
        boot[b] = curve_b["means"]
        if verbose and ((b + 1) % 20 == 0 or b == n_boot - 1):
            elapsed = time.time() - t0
            print(f"    bootstrap {b + 1:>4d}/{n_boot}   "
                  f"elapsed = {elapsed:6.1f}s")
    return {
        "mean": boot.mean(axis=0),
        "low":  np.percentile(boot, 2.5, axis=0),
        "high": np.percentile(boot, 97.5, axis=0),
    }


# ===========================================================
# End-to-end pipeline
# ===========================================================

def run_volume_entropy_null(
    N_null: int = DEFAULT_N_NULL,
    K: int = 5,
    T_mean: Optional[float] = None,
    T_sd: Optional[float] = None,
    n_boot: int = DEFAULT_N_BOOT,
    n_bins: int = DEFAULT_N_BINS,
    ordinal_transform: str = "expected",
    country_filter: Optional[str] = "US",
    seed: int = RANDOM_SEED,
    output_prefix: str = "csr_volume_entropy_null",
) -> Dict[str, Any]:
    """End-to-end pipeline. See module docstring for details."""

    print("=" * 72)
    print("  VOLUME-ENTROPY NULL BENCHMARK  (Empirical Study, S5.4)")
    print("=" * 72)

    # ----- 1. Empirical data -----
    csv_path = download_big5_data()
    data = load_and_prepare_data(csv_path, country_filter=country_filter)
    y_emp = data[EXTRAVERSION_ITEMS].values
    N_emp, J = y_emp.shape
    print(f"\n  Empirical sample: N = {N_emp:,}, J = {J}, K = {K}")

    # ----- 2. Empirical CSR -----
    print("\n[1/4] Fitting CSR on empirical data ...")
    eta_emp, H_emp = fit_csr_and_metrics(
        y_emp, ordinal_transform=ordinal_transform, seed=seed,
    )
    r_raw, _ = safe_corr(eta_emp, H_emp)
    print(f"  Raw r(eta, H) = {r_raw:+.4f}")

    # ----- 3. Empirical pi and row-total moments -----
    pi = compute_empirical_pi(y_emp)
    sum_scores = y_emp.sum(axis=1)
    T_mean_use = float(sum_scores.mean()) if T_mean is None else T_mean
    T_sd_use   = float(sum_scores.std(ddof=1)) if T_sd is None else T_sd
    print(f"\n  Empirical item marginals pi (matched to mean(y_j) - 1):")
    pi_str = ", ".join(f"{p:.4f}" for p in pi)
    print(f"    [{pi_str}]")
    print(f"  Row totals: T_mean = {T_mean_use:.2f}, T_sd = {T_sd_use:.2f}")

    # ----- 4. Generate null and fit CSR -----
    print(f"\n[2/4] Generating null (N = {N_null:,}) and fitting CSR ...")
    t0 = time.time()
    null = gen_rowvolume_null(
        N=N_null, J=J, K=K,
        T_mean=T_mean_use, T_sd=T_sd_use,
        item_marginals=pi,
        seed=seed,
    )
    y_null = null["y"]
    eta_null, H_null = fit_csr_and_metrics(
        y_null, ordinal_transform=ordinal_transform, seed=seed,
    )
    print(f"  Null CSR fit completed in {time.time() - t0:.1f} sec")
    print(f"  Null eta range: [{eta_null.min():.3f}, {eta_null.max():.3f}]")
    print(f"  Emp  eta range: [{eta_emp.min():.3f}, {eta_emp.max():.3f}]")

    # ----- 5. Baseline curve on null -----
    print(f"\n[3/4] Estimating b_0(eta) on null (n_bins = {n_bins}) ...")
    eta_range = (
        float(min(eta_null.min(), eta_emp.min())),
        float(max(eta_null.max(), eta_emp.max())),
    )
    curve = baseline_curve(
        eta_null, H_null, n_bins=n_bins, eta_range=eta_range,
    )

    # Apply baseline to empirical eta
    b0_emp = evaluate_baseline(eta_emp, curve)
    H_tilde = H_emp - b0_emp

    # ----- 6. Bootstrap bands on the curve -----
    print(f"\n[4/4] Bootstrap bands (n_boot = {n_boot}) ...")
    bands = bootstrap_bands(
        eta_null, H_null, eta_range=eta_range,
        n_bins=n_bins, n_boot=n_boot, seed=seed + 7919,
    )

    # ----- 7. Summary -----
    r_tilde, _ = safe_corr(eta_emp, H_tilde)
    var_H = float(H_emp.var(ddof=1))
    var_H_tilde = float(H_tilde.var(ddof=1))
    absorbed = 1.0 - var_H_tilde / var_H if var_H > 0 else float("nan")

    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"  Raw r(eta, H):                   {r_raw:+.4f}")
    print(f"  Residual r(eta, H_tilde):        {r_tilde:+.4f}")
    print(f"  Var(H):                          {var_H:.6f}")
    print(f"  Var(H_tilde):                    {var_H_tilde:.6f}")
    print(f"  Fraction Var(H) absorbed by b_0: {absorbed:.3f}")

    # ----- 8. Export -----
    print("\n  Exporting:")
    pp_df = pd.DataFrame({
        "person_idx": np.arange(N_emp),
        "eta_emp":    eta_emp,
        "H_emp":      H_emp,
        "b0_eta":     b0_emp,
        "H_tilde":    H_tilde,
    })
    pp_path = f"{output_prefix}_per_person.csv"
    pp_df.to_csv(pp_path, index=False)
    print(f"    per-person:  {pp_path}")

    grid_df = pd.DataFrame({
        "eta_center": curve["centers"],
        "b0_point":   curve["means"],
        "b0_count":   curve["counts"],
        "b0_boot_mean": bands["mean"],
        "b0_low95":   bands["low"],
        "b0_high95":  bands["high"],
    })
    grid_path = f"{output_prefix}_smoother.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"    smoother:    {grid_path}")

    sum_df = pd.DataFrame([{
        "N_emp":             N_emp,
        "N_null":            N_null,
        "J":                 J,
        "K":                 K,
        "T_mean":            T_mean_use,
        "T_sd":              T_sd_use,
        "n_bins":            n_bins,
        "n_boot":            n_boot,
        "ordinal_transform": ordinal_transform,
        "r_raw":             r_raw,
        "r_tilde":           r_tilde,
        "var_H":             var_H,
        "var_H_tilde":       var_H_tilde,
        "fraction_absorbed": absorbed,
    }])
    sum_path = f"{output_prefix}_summary.csv"
    sum_df.to_csv(sum_path, index=False)
    print(f"    summary:     {sum_path}")

    return {
        "eta_emp":      eta_emp,
        "H_emp":        H_emp,
        "b0_emp":       b0_emp,
        "H_tilde":      H_tilde,
        "curve":        curve,
        "bands":        bands,
        "r_raw":        r_raw,
        "r_tilde":      r_tilde,
        "var_H":        var_H,
        "var_H_tilde":  var_H_tilde,
        "absorbed":     absorbed,
        "pi":           pi,
        "T_mean":       T_mean_use,
        "T_sd":         T_sd_use,
    }


# ===========================================================
# CLI
# ===========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Volume-entropy null benchmark (Empirical Study, S5.4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--N_null", type=int, default=DEFAULT_N_NULL,
                   help="Null sample size")
    p.add_argument("--K", type=int, default=5,
                   help="Number of Likert categories")
    p.add_argument("--T_mean", type=float, default=None,
                   help="Row total mean (default: empirical)")
    p.add_argument("--T_sd", type=float, default=None,
                   help="Row total SD (default: empirical)")
    p.add_argument("--n_boot", type=int, default=DEFAULT_N_BOOT,
                   help="Bootstrap replications for bands")
    p.add_argument("--n_bins", type=int, default=DEFAULT_N_BINS,
                   help="Bins for baseline curve")
    p.add_argument("--transform", type=str, default="expected",
                   choices=["none", "midpoint", "expected", "rank"],
                   help="CSR ordinal transformation")
    p.add_argument("--country", type=str, default="US",
                   help="Country filter ('all' for none)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help="Random seed")
    p.add_argument("--output_prefix", type=str,
                   default="csr_volume_entropy_null",
                   help="Output file prefix")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: N_null=5000, n_boot=20")
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if args.quick:
        args.N_null = 5_000
        args.n_boot = 20
        print("[quick mode] N_null = 5000, n_boot = 20")

    country = None if args.country.lower() == "all" else args.country
    run_volume_entropy_null(
        N_null=args.N_null,
        K=args.K,
        T_mean=args.T_mean,
        T_sd=args.T_sd,
        n_boot=args.n_boot,
        n_bins=args.n_bins,
        ordinal_transform=args.transform,
        country_filter=country,
        seed=args.seed,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
