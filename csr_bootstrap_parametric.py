"""
csr_bootstrap_parametric.py
==========================
Parametric row-perturbation bootstrap for per-individual CSR salience
uncertainty.

This file replaces the parametric section of csr_bootstrap_fixed.py.
The earlier version was too pessimistic because it computed residuals on the
centered expected-value ordinal scale, while CSR reconstruction eta * s is
nonnegative. That scale mismatch produced large negative residual means,
strong skew, and inflated perturbation noise.

Fixes in this version
---------------------
1. Use a positive expected-value (positive-EV) scale:
      z_ev       = E[z | y] under the fixed logistic thresholds
      z_positive = z_ev - min(z_ev) + lower_bound
   with lower_bound = 1.0 by default. This preserves ordinal spacing while
   satisfying CSR's nonnegative-domain requirement.

2. Fit CSR to z_positive directly with ordinal_transform='none'. The
   perturbation and refit are now performed on the same positive continuous
   scale, avoiding latent-scale/sign mismatch.

3. Center residuals before perturbing. The bootstrap should represent
   independent noise around the fitted reconstruction, not systematic bias.

4. Use heteroscedastic residual noise by eta_hat quintile by default. This
   addresses the reviewer question about how uncertainty scales with response
   magnitude and avoids using one implausible global sigma when residual spread
   varies strongly by volume.

5. Provide both per-person whole-vector stability and per-cell SE(s_ij), plus
   eta- and entropy-quintile summaries.

Run:
  python csr_bootstrap_parametric.py --n-boot 500
  python csr_bootstrap_parametric.py --n-boot 200 --noise-mode eta_quintile
  python csr_bootstrap_parametric.py --n-boot 200 --noise-mode residual_resample

Requires:
  csr_main.py
  csr_empirical_study.py

Author: Jonathan Lee
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from csr_main import (
    fit_csr,
    safe_corr,
    compute_entropy,
    ordinal_to_latent,
)
from csr_empirical_study import (
    EXTRAVERSION_ITEMS,
    download_big5_data,
    load_and_prepare_data,
)


# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
DEFAULT_LOWER_BOUND = 1.0


# ============================================================================
# Result containers
# ============================================================================

@dataclass
class ResidualDiagnostics:
    """Diagnostics for residuals on the positive continuous CSR scale."""
    sigma_global_raw: float
    sigma_global_centered: float
    residual_mean: float
    residual_skew: float
    heteroscedasticity_corr: float
    sigma_by_eta_quintile: np.ndarray
    eta_quintile_edges: np.ndarray


@dataclass
class ParametricBootstrapResults:
    """Container for per-individual parametric bootstrap outputs."""
    n_boot: int
    n_valid: int
    n_individuals: int
    n_items: int

    noise_mode: str
    lower_bound: float
    diagnostics: ResidualDiagnostics

    eta_hat: np.ndarray
    S_hat: np.ndarray
    entropy_norm: np.ndarray

    se_S: np.ndarray
    stability_per_person: np.ndarray
    mean_se_per_person: np.ndarray

    eta_quintile_label: np.ndarray
    entropy_quintile_label: np.ndarray
    eta_quintile_summary: pd.DataFrame
    entropy_quintile_summary: pd.DataFrame

    n_failed: int
    elapsed_seconds: float


# ============================================================================
# Core helpers
# ============================================================================

def cosine_rows(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine similarity between matched matrices A and B."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    nA = np.linalg.norm(A, axis=1) + eps
    nB = np.linalg.norm(B, axis=1) + eps
    return np.sum(A * B, axis=1) / (nA * nB)


def sample_skewness(x: np.ndarray, eps: float = 1e-12) -> float:
    """Simple moment skewness."""
    x = np.asarray(x, dtype=float).ravel()
    mu = float(x.mean())
    sd = float(x.std())
    return float(np.mean((x - mu) ** 3) / (sd ** 3 + eps))


def positive_ev_transform(
    responses: np.ndarray,
    lower_bound: float = DEFAULT_LOWER_BOUND,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Map Likert responses to a positive expected-value scale.

    Returns
    -------
    z_pos : ndarray
        Positive continuous response matrix used for CSR fitting.
    z_ev : ndarray
        Original expected-value latent scores before shifting.
    shift : float
        Shift added to z_ev: z_pos = z_ev + shift.

    Notes
    -----
    The original expected-value transform is centered and can contain negative
    values. CSR requires nonnegative input because s_ij >= 0 and eta_i > 0.
    Shifting the EV scores to a positive lower bound makes the parametric
    bootstrap coherent with the CSR reconstruction scale.
    """
    y = np.asarray(responses, dtype=float)
    K = int(np.nanmax(y))
    z_ev = ordinal_to_latent(y, K=K, method="expected")
    shift = float(lower_bound - np.min(z_ev))
    z_pos = z_ev + shift
    return z_pos, z_ev, shift


def quintile_labels(values: np.ndarray, keep_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return 1..5 quintile labels and the four quintile edges.
    Labels are NaN where keep_mask is False.
    """
    values = np.asarray(values, dtype=float)
    if keep_mask is None:
        keep_mask = np.ones(values.shape, dtype=bool)

    labels = np.full(values.shape, np.nan)
    if np.sum(keep_mask) < 5:
        return labels, np.full(4, np.nan)

    edges = np.quantile(values[keep_mask], [0.2, 0.4, 0.6, 0.8])
    labels[keep_mask] = np.digitize(values[keep_mask], edges) + 1
    return labels, edges


def summarize_by_quintile(
    labels: np.ndarray,
    stability: np.ndarray,
    mean_se: np.ndarray,
    eta_hat: np.ndarray,
    entropy_norm: np.ndarray,
) -> pd.DataFrame:
    """Create a compact quintile summary table."""
    rows = []
    for q in range(1, 6):
        mask = labels == q
        if np.sum(mask) == 0:
            rows.append({
                "quintile": q,
                "n": 0,
                "mean_eta": np.nan,
                "mean_entropy": np.nan,
                "mean_cosine": np.nan,
                "median_cosine": np.nan,
                "pct_cos_gt_90": np.nan,
                "pct_cos_gt_95": np.nan,
                "mean_SE_s_ij": np.nan,
            })
            continue

        stab = stability[mask]
        rows.append({
            "quintile": q,
            "n": int(np.sum(mask)),
            "mean_eta": float(np.mean(eta_hat[mask])),
            "mean_entropy": float(np.mean(entropy_norm[mask])),
            "mean_cosine": float(np.mean(stab)),
            "median_cosine": float(np.median(stab)),
            "pct_cos_gt_90": float(np.mean(stab > 0.90)),
            "pct_cos_gt_95": float(np.mean(stab > 0.95)),
            "mean_SE_s_ij": float(np.mean(mean_se[mask])),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Residual diagnostics and noise generation
# ============================================================================

def residual_diagnostics_positive_scale(
    z_pos: np.ndarray,
    eta_hat: np.ndarray,
    S_hat: np.ndarray,
    eta_labels: np.ndarray,
    verbose: bool = True,
) -> ResidualDiagnostics:
    """Residual diagnostics on the positive continuous CSR scale."""
    z_pred = eta_hat[:, None] * S_hat
    residuals = z_pos - z_pred
    residuals_centered = residuals - residuals.mean()

    sigma_by_q = np.zeros(5)
    for q in range(1, 6):
        mask = eta_labels == q
        if np.sum(mask) == 0:
            sigma_by_q[q - 1] = np.nan
        else:
            r_q = residuals[mask]
            sigma_by_q[q - 1] = float(np.std(r_q - r_q.mean()))

    abs_e_per_person = np.mean(np.abs(residuals_centered), axis=1)
    het_corr, _ = safe_corr(abs_e_per_person, eta_hat)

    diag = ResidualDiagnostics(
        sigma_global_raw=float(np.std(residuals)),
        sigma_global_centered=float(np.std(residuals_centered)),
        residual_mean=float(np.mean(residuals)),
        residual_skew=sample_skewness(residuals_centered),
        heteroscedasticity_corr=float(het_corr),
        sigma_by_eta_quintile=sigma_by_q,
        eta_quintile_edges=np.array([
            np.nanquantile(eta_hat, 0.2),
            np.nanquantile(eta_hat, 0.4),
            np.nanquantile(eta_hat, 0.6),
            np.nanquantile(eta_hat, 0.8),
        ]),
    )

    if verbose:
        print("  Residual diagnostics on positive-EV scale:")
        print(f"    residual mean:           {diag.residual_mean:+.4f}")
        print(f"    global SD raw:           {diag.sigma_global_raw:.4f}")
        print(f"    global SD centered:      {diag.sigma_global_centered:.4f}")
        print(f"    centered skewness:       {diag.residual_skew:+.3f}")
        print(f"    corr(mean |e|, eta):     {diag.heteroscedasticity_corr:+.3f}")
        print("    SD by eta quintile:      " + ", ".join(f"{x:.4f}" for x in diag.sigma_by_eta_quintile))

        if abs(diag.heteroscedasticity_corr) > 0.20:
            print("    [note] residual spread varies by eta; eta_quintile noise is recommended.")
        if abs(diag.residual_skew) > 0.50:
            print("    [note] residuals remain skewed; residual_resample is useful as a sensitivity check.")

    return diag


def generate_parametric_matrix(
    z_pred: np.ndarray,
    residuals_centered: np.ndarray,
    eta_labels: np.ndarray,
    diagnostics: ResidualDiagnostics,
    rng: np.random.Generator,
    noise_mode: str = "eta_quintile",
) -> np.ndarray:
    """
    Generate one perturbed positive-scale response matrix.

    noise_mode options
    ------------------
    global_gaussian:
        Add N(0, global centered SD^2) noise to all cells.
    eta_quintile:
        Add N(0, sigma_q^2) noise using eta quintile-specific residual SD.
    residual_resample:
        Nonparametric residual bootstrap within eta quintile, sampling centered
        residual cells with replacement.
    """
    N, J = z_pred.shape

    if noise_mode == "global_gaussian":
        noise = rng.normal(0.0, diagnostics.sigma_global_centered, size=(N, J))
        return z_pred + noise

    if noise_mode == "eta_quintile":
        noise = np.zeros((N, J))
        for q in range(1, 6):
            mask = eta_labels == q
            if not np.any(mask):
                continue
            sigma_q = diagnostics.sigma_by_eta_quintile[q - 1]
            if not np.isfinite(sigma_q) or sigma_q <= 0:
                sigma_q = diagnostics.sigma_global_centered
            noise[mask] = rng.normal(0.0, sigma_q, size=(np.sum(mask), J))
        return z_pred + noise

    if noise_mode == "residual_resample":
        noise = np.zeros((N, J))
        for q in range(1, 6):
            mask = eta_labels == q
            if not np.any(mask):
                continue
            # Center WITHIN quintile so the resampled noise is zero-mean
            # within the quintile. Globally-centered residuals would carry
            # a per-quintile bias under heteroscedasticity.
            r_q = residuals_centered[mask]
            pool = (r_q - r_q.mean()).ravel()
            if pool.size == 0:
                pool = (residuals_centered - residuals_centered.mean()).ravel()
            noise[mask] = rng.choice(pool, size=(np.sum(mask), J), replace=True)
        return z_pred + noise

    raise ValueError(
        "Unknown noise_mode. Use 'global_gaussian', 'eta_quintile', or 'residual_resample'."
    )


# ============================================================================
# Bootstrap loop
# ============================================================================

def run_parametric_bootstrap(
    responses: np.ndarray,
    n_boot: int = 500,
    lower_bound: float = DEFAULT_LOWER_BOUND,
    noise_mode: str = "eta_quintile",
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> ParametricBootstrapResults:
    """
    Run parametric row-perturbation bootstrap on the positive-EV scale.
    """
    rng = np.random.default_rng(seed)
    responses = np.asarray(responses, dtype=float)
    N, J = responses.shape

    if verbose:
        print("\n" + "=" * 70)
        print("CSR PARAMETRIC ROW-PERTURBATION BOOTSTRAP")
        print("=" * 70)
        print(f"  N = {N:,}, J = {J}, n_boot = {n_boot}")
        print(f"  Positive-EV lower bound: {lower_bound}")
        print(f"  Noise mode: {noise_mode}")

    # Positive expected-value transform
    z_pos, z_ev, shift = positive_ev_transform(responses, lower_bound=lower_bound)
    if verbose:
        print("\nPositive-EV transform:")
        print(f"  raw EV range:       [{z_ev.min():.4f}, {z_ev.max():.4f}]")
        print(f"  shift applied:      {shift:.4f}")
        print(f"  positive EV range:  [{z_pos.min():.4f}, {z_pos.max():.4f}]")

    # Original CSR fit on the exact continuous positive scale.
    # No ordinal transform here; the transform has already been applied.
    original_fit = fit_csr(
        z_pos,
        ordinal_transform="none",
        verbose=False,
        seed=seed,
    )
    S_hat = original_fit.S
    eta_hat = original_fit.eta
    entropy_norm = compute_entropy(S_hat) / np.log(J)

    eta_labels, eta_edges = quintile_labels(eta_hat)
    entropy_labels, _ = quintile_labels(entropy_norm)

    diagnostics = residual_diagnostics_positive_scale(
        z_pos=z_pos,
        eta_hat=eta_hat,
        S_hat=S_hat,
        eta_labels=eta_labels,
        verbose=verbose,
    )
    diagnostics.eta_quintile_edges = eta_edges

    z_pred = eta_hat[:, None] * S_hat
    residuals = z_pos - z_pred
    residuals_centered = residuals - residuals.mean()

    # Welford accumulators for SE(s_ij)
    welford_mean = np.zeros((N, J))
    welford_M2 = np.zeros((N, J))
    n_seen = 0

    cosine_sum = np.zeros(N)
    n_failed = 0
    start_time = time.time()

    for b in range(n_boot):
        try:
            z_b = generate_parametric_matrix(
                z_pred=z_pred,
                residuals_centered=residuals_centered,
                eta_labels=eta_labels,
                diagnostics=diagnostics,
                rng=rng,
                noise_mode=noise_mode,
            )

            # Enforce CSR admissible domain at the empirical lower bound of
            # the positive-EV scale. Clipping at lower_bound (rather than at
            # a tiny constant) keeps the bootstrap data domain consistent
            # with the original analysis scale.
            z_b = np.maximum(z_b, lower_bound)

            fit_b = fit_csr(
                z_b,
                ordinal_transform="none",
                verbose=False,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            S_b = fit_b.S

        except Exception as exc:
            n_failed += 1
            print(f"  [warning] parametric bootstrap replication failed: {exc}")
            continue

        n_seen += 1

        delta = S_b - welford_mean
        welford_mean += delta / n_seen
        delta2 = S_b - welford_mean
        welford_M2 += delta * delta2

        cosine_sum += cosine_rows(S_hat, S_b)

        if verbose and ((b + 1) % 25 == 0 or b == 0):
            elapsed = time.time() - start_time
            rate = (b + 1) / max(elapsed, 1e-12)
            remaining = (n_boot - b - 1) / rate
            print(
                f"  [{b + 1:4d}/{n_boot}] elapsed = {elapsed:7.1f}s, "
                f"remaining ≈ {remaining:7.1f}s, failed = {n_failed}"
            )

    elapsed = time.time() - start_time

    if n_seen == 0:
        raise RuntimeError("All parametric bootstrap replications failed.")

    if n_seen > 1:
        se_S = np.sqrt(np.maximum(welford_M2 / (n_seen - 1), 0.0))
    else:
        se_S = np.zeros_like(welford_M2)

    stability_per_person = cosine_sum / n_seen
    mean_se_per_person = se_S.mean(axis=1)

    eta_summary = summarize_by_quintile(
        labels=eta_labels,
        stability=stability_per_person,
        mean_se=mean_se_per_person,
        eta_hat=eta_hat,
        entropy_norm=entropy_norm,
    )
    entropy_summary = summarize_by_quintile(
        labels=entropy_labels,
        stability=stability_per_person,
        mean_se=mean_se_per_person,
        eta_hat=eta_hat,
        entropy_norm=entropy_norm,
    )

    return ParametricBootstrapResults(
        n_boot=n_boot,
        n_valid=n_seen,
        n_individuals=N,
        n_items=J,
        noise_mode=noise_mode,
        lower_bound=lower_bound,
        diagnostics=diagnostics,
        eta_hat=eta_hat,
        S_hat=S_hat,
        entropy_norm=entropy_norm,
        se_S=se_S,
        stability_per_person=stability_per_person,
        mean_se_per_person=mean_se_per_person,
        eta_quintile_label=eta_labels,
        entropy_quintile_label=entropy_labels,
        eta_quintile_summary=eta_summary,
        entropy_quintile_summary=entropy_summary,
        n_failed=n_failed,
        elapsed_seconds=elapsed,
    )


# ============================================================================
# Reporting and export
# ============================================================================

def print_report(results: ParametricBootstrapResults) -> None:
    """Print per-individual reliability report."""
    stab = results.stability_per_person
    se_flat = results.se_S.ravel()

    print("\n" + "=" * 70)
    print("PARAMETRIC BOOTSTRAP RESULTS")
    print("=" * 70)
    print(f"  Requested replications: {results.n_boot}")
    print(f"  Valid replications:     {results.n_valid}")
    print(f"  Failed replications:    {results.n_failed}")
    print(f"  Noise mode:             {results.noise_mode}")
    print(f"  Lower bound:            {results.lower_bound}")
    print(f"  Total time:             {results.elapsed_seconds:.1f}s")

    print("\n--- Residual noise model ---")
    d = results.diagnostics
    print(f"  Residual mean:          {d.residual_mean:+.4f}")
    print(f"  Global SD centered:     {d.sigma_global_centered:.4f}")
    print(f"  Residual skewness:      {d.residual_skew:+.3f}")
    print(f"  Heteroscedasticity r:   {d.heteroscedasticity_corr:+.3f}")
    print("  Sigma by eta quintile:  " + ", ".join(f"{x:.4f}" for x in d.sigma_by_eta_quintile))

    print("\n--- Per-person salience stability ---")
    print(f"  Mean cosine:            {stab.mean():.4f}")
    print(f"  SD across persons:      {stab.std():.4f}")
    print(f"  Median:                 {np.median(stab):.4f}")
    print(f"  10th / 90th pctile:     {np.percentile(stab, 10):.4f} / {np.percentile(stab, 90):.4f}")
    print(f"  Persons cos > 0.90:     {100 * np.mean(stab > 0.90):.1f}%")
    print(f"  Persons cos > 0.95:     {100 * np.mean(stab > 0.95):.1f}%")

    print("\n--- Per-cell SE(s_ij) ---")
    print(f"  Mean SE(s_ij):          {se_flat.mean():.4f}")
    print(f"  Median SE(s_ij):        {np.median(se_flat):.4f}")
    print(f"  90th pctile:            {np.percentile(se_flat, 90):.4f}")

    print("\n--- Stability by eta quintile ---")
    print(results.eta_quintile_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Stability by entropy quintile ---")
    print(results.entropy_quintile_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def export_results(
    results: ParametricBootstrapResults,
    output_prefix: str = "csr_bootstrap_parametric",
) -> None:
    """Export per-person, per-cell, and quintile summaries."""
    pp_df = pd.DataFrame({
        "person_idx": np.arange(results.n_individuals),
        "eta_hat": results.eta_hat,
        "entropy_norm": results.entropy_norm,
        "eta_quintile": results.eta_quintile_label,
        "entropy_quintile": results.entropy_quintile_label,
        "stability_cosine": results.stability_per_person,
        "mean_SE_s_ij": results.mean_se_per_person,
    })
    pp_path = f"{output_prefix}_per_person.csv"
    pp_df.to_csv(pp_path, index=False)

    se_df = pd.DataFrame(
        results.se_S,
        columns=[f"se_s_item_{j + 1}" for j in range(results.n_items)],
    )
    se_df.insert(0, "person_idx", np.arange(results.n_individuals))
    se_path = f"{output_prefix}_se_cells.csv"
    se_df.to_csv(se_path, index=False)

    eta_path = f"{output_prefix}_eta_quintile_summary.csv"
    entropy_path = f"{output_prefix}_entropy_quintile_summary.csv"
    results.eta_quintile_summary.to_csv(eta_path, index=False)
    results.entropy_quintile_summary.to_csv(entropy_path, index=False)

    diag_df = pd.DataFrame({
        "noise_mode": [results.noise_mode],
        "lower_bound": [results.lower_bound],
        "n_boot": [results.n_boot],
        "n_valid": [results.n_valid],
        "n_failed": [results.n_failed],
        "residual_mean": [results.diagnostics.residual_mean],
        "sigma_global_raw": [results.diagnostics.sigma_global_raw],
        "sigma_global_centered": [results.diagnostics.sigma_global_centered],
        "residual_skew": [results.diagnostics.residual_skew],
        "heteroscedasticity_corr": [results.diagnostics.heteroscedasticity_corr],
        "sigma_eta_q1": [results.diagnostics.sigma_by_eta_quintile[0]],
        "sigma_eta_q2": [results.diagnostics.sigma_by_eta_quintile[1]],
        "sigma_eta_q3": [results.diagnostics.sigma_by_eta_quintile[2]],
        "sigma_eta_q4": [results.diagnostics.sigma_by_eta_quintile[3]],
        "sigma_eta_q5": [results.diagnostics.sigma_by_eta_quintile[4]],
    })
    diag_path = f"{output_prefix}_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)

    print("\nExported parametric bootstrap results:")
    print(f"  Per-person:          {pp_path}")
    print(f"  Per-cell SE:         {se_path}")
    print(f"  Eta quintiles:       {eta_path}")
    print(f"  Entropy quintiles:   {entropy_path}")
    print(f"  Diagnostics:         {diag_path}")


# ============================================================================
# Main
# ============================================================================

def main(
    n_boot: int = 500,
    country_filter: str = "US",
    lower_bound: float = DEFAULT_LOWER_BOUND,
    noise_mode: str = "eta_quintile",
    export: bool = True,
    seed: int = RANDOM_SEED,
) -> Dict[str, object]:
    print("\n" + "=" * 70)
    print("CSR PARAMETRIC BOOTSTRAP")
    print("=" * 70)

    csv_path = download_big5_data()
    data = load_and_prepare_data(csv_path, country_filter=country_filter)
    responses = data[EXTRAVERSION_ITEMS].values

    results = run_parametric_bootstrap(
        responses=responses,
        n_boot=n_boot,
        lower_bound=lower_bound,
        noise_mode=noise_mode,
        seed=seed,
        verbose=True,
    )

    print_report(results)

    if export:
        export_results(results)

    print("\n" + "=" * 70)
    print("PARAMETRIC BOOTSTRAP COMPLETE")
    print("=" * 70)

    return {"parametric_results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSR parametric row-perturbation bootstrap on positive-EV scale",
    )
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--country", type=str, default="US")
    parser.add_argument("--lower-bound", type=float, default=DEFAULT_LOWER_BOUND)
    parser.add_argument(
        "--noise-mode",
        type=str,
        default="eta_quintile",
        choices=["global_gaussian", "eta_quintile", "residual_resample"],
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-export", action="store_true")

    args, _ = parser.parse_known_args()

    country = None if args.country.lower() == "all" else args.country

    main(
        n_boot=args.n_boot,
        country_filter=country,
        lower_bound=args.lower_bound,
        noise_mode=args.noise_mode,
        export=not args.no_export,
        seed=args.seed,
    )
