"""
csr_bootstrap_aggregate.py
==========================
Case-resampling bootstrap for population-level CSR empirical outputs.

This file contains ONLY the valid aggregate/bootstrap layer from the
empirical CSR analysis. It resamples persons with replacement, refits CSR,
and summarizes quantities that are genuinely population/sample-level:

  1. Archetype reproducibility
  2. Volume-entropy correlation percentile CI
  3. CSR-vs-uniform reconstruction MSE reduction percentile CI

It intentionally does NOT estimate per-person salience reliability. Case
resampling reuses exact response rows, and CSR salience is row-wise
deterministic given the response vector, so per-person case-bootstrap
cosines are tautological.

Run:
  python csr_bootstrap_aggregate.py --n-boot 500
  python csr_bootstrap_aggregate.py --n-boot 200 --no-export

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

from csr_main import fit_csr, safe_corr, compute_entropy
from csr_empirical_study import (
    EXTRAVERSION_ITEMS,
    download_big5_data,
    load_and_prepare_data,
    fit_csr_model,
    extract_archetypes,
    clr_transform,
)


# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
N_ARCHETYPES = 3
ARCHETYPE_RECOVERY_THRESHOLD = 0.90


# ============================================================================
# Result container
# ============================================================================

@dataclass
class AggregateBootstrapResults:
    """Container for case-resampling bootstrap outputs."""
    n_boot: int
    n_valid: int
    n_individuals: int

    archetype_cosines: np.ndarray
    archetype_full_recovery_rate: float

    vol_entropy_corr_boot: np.ndarray
    vol_entropy_ci: Tuple[float, float]

    mse_csr_boot: np.ndarray
    mse_uniform_boot: np.ndarray
    mse_reduction_ci: Tuple[float, float]

    n_failed: int
    elapsed_seconds: float


# ============================================================================
# Helpers
# ============================================================================

def cosine_pair(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two 1D vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))


def hungarian_match_archetypes(
    boot_centroids: np.ndarray,
    orig_centroids: np.ndarray,
) -> np.ndarray:
    """
    Align bootstrap archetypes to original archetypes using Hungarian matching
    on cosine similarity.

    Returns
    -------
    perm : ndarray of shape (K,)
        perm[k] is the bootstrap centroid index aligned to original archetype k.
    """
    from scipy.optimize import linear_sum_assignment

    K = orig_centroids.shape[0]
    cost = np.zeros((K, K))
    for boot_k in range(K):
        for orig_k in range(K):
            cost[boot_k, orig_k] = -cosine_pair(boot_centroids[boot_k], orig_centroids[orig_k])

    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros(K, dtype=int)
    for boot_k, orig_k in zip(row_ind, col_ind):
        perm[orig_k] = boot_k
    return perm


def uniform_reconstruction_mse(y: np.ndarray) -> float:
    """
    Reconstruction MSE under uniform salience s_j = 1/J.

    With uniform salience, the optimal reconstruction is each person's mean
    response repeated across items.
    """
    y = np.asarray(y, dtype=float)
    person_means = y.mean(axis=1, keepdims=True)
    yhat = np.repeat(person_means, y.shape[1], axis=1)
    return float(np.mean((y - yhat) ** 2))


# ============================================================================
# Single bootstrap replication
# ============================================================================

def run_one_aggregate_bootstrap(
    responses: np.ndarray,
    orig_centroids: np.ndarray,
    rng: np.random.Generator,
    ordinal_transform: str = "expected",
    n_archetypes: int = N_ARCHETYPES,
) -> Optional[Dict[str, np.ndarray | float]]:
    """Run one case-resampling bootstrap replication."""
    N = responses.shape[0]

    try:
        boot_idx = rng.integers(0, N, size=N)
        y_boot = responses[boot_idx]

        res = fit_csr(
            y_boot,
            ordinal_transform=ordinal_transform,
            verbose=False,
            seed=int(rng.integers(0, 2**31 - 1)),
        )

        S_boot = res.S
        eta_boot = res.eta

        # Entropy normalized to [0, 1]
        H_norm = compute_entropy(S_boot) / np.log(S_boot.shape[1])

        # CLR-k-means archetypes
        from sklearn.cluster import KMeans

        S_clr = clr_transform(S_boot)
        km = KMeans(
            n_clusters=n_archetypes,
            random_state=int(rng.integers(0, 2**31 - 1)),
            n_init=10,
        )
        labels_boot = km.fit_predict(S_clr)

        boot_centroids = np.zeros((n_archetypes, S_boot.shape[1]))
        for k in range(n_archetypes):
            mask = labels_boot == k
            if np.sum(mask) == 0:
                return None
            boot_centroids[k] = S_boot[mask].mean(axis=0)

        perm = hungarian_match_archetypes(boot_centroids, orig_centroids)
        aligned_centroids = boot_centroids[perm]
        archetype_cosines = np.array([
            cosine_pair(aligned_centroids[k], orig_centroids[k])
            for k in range(n_archetypes)
        ])

        vol_entropy_corr, _ = safe_corr(eta_boot, H_norm)

        y_for_mse = res.z if res.z is not None else y_boot.astype(float)
        yhat_csr = eta_boot[:, None] * S_boot
        mse_csr = float(np.mean((y_for_mse - yhat_csr) ** 2))
        mse_uniform = uniform_reconstruction_mse(y_for_mse)

        return {
            "archetype_cosines": archetype_cosines,
            "vol_entropy_corr": vol_entropy_corr,
            "mse_csr": mse_csr,
            "mse_uniform": mse_uniform,
        }

    except Exception as exc:
        print(f"  [warning] aggregate bootstrap replication failed: {exc}")
        return None


# ============================================================================
# Bootstrap loop
# ============================================================================

def run_aggregate_bootstrap(
    responses: np.ndarray,
    orig_centroids: np.ndarray,
    n_boot: int = 500,
    ordinal_transform: str = "expected",
    n_archetypes: int = N_ARCHETYPES,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> AggregateBootstrapResults:
    """Run case-resampling bootstrap for aggregate CSR quantities."""
    N, J = responses.shape
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"\n{'=' * 70}")
        print("CASE BOOTSTRAP: AGGREGATE CSR OUTPUTS")
        print(f"{'=' * 70}")
        print(f"  N = {N:,}, J = {J}, n_boot = {n_boot}")
        print(f"  Transform: {ordinal_transform}")

    archetype_cosines_all = []
    vol_entropy_corr_all = []
    mse_csr_all = []
    mse_uniform_all = []
    n_failed = 0
    start_time = time.time()

    for b in range(n_boot):
        rep = run_one_aggregate_bootstrap(
            responses=responses,
            orig_centroids=orig_centroids,
            rng=rng,
            ordinal_transform=ordinal_transform,
            n_archetypes=n_archetypes,
        )

        if rep is None:
            n_failed += 1
            continue

        archetype_cosines_all.append(rep["archetype_cosines"])
        vol_entropy_corr_all.append(rep["vol_entropy_corr"])
        mse_csr_all.append(rep["mse_csr"])
        mse_uniform_all.append(rep["mse_uniform"])

        if verbose and ((b + 1) % 50 == 0 or b == 0):
            elapsed = time.time() - start_time
            rate = (b + 1) / max(elapsed, 1e-12)
            eta_seconds = (n_boot - b - 1) / rate
            print(
                f"  [{b + 1:4d}/{n_boot}] elapsed = {elapsed:7.1f}s, "
                f"remaining ≈ {eta_seconds:7.1f}s, failed = {n_failed}"
            )

    elapsed = time.time() - start_time

    archetype_cosines = np.asarray(archetype_cosines_all)
    vol_entropy_corr_boot = np.asarray(vol_entropy_corr_all)
    mse_csr_boot = np.asarray(mse_csr_all)
    mse_uniform_boot = np.asarray(mse_uniform_all)

    if len(vol_entropy_corr_boot) == 0:
        raise RuntimeError("All aggregate bootstrap replications failed.")

    full_recovery = np.all(archetype_cosines > ARCHETYPE_RECOVERY_THRESHOLD, axis=1)
    mse_reduction = mse_uniform_boot - mse_csr_boot

    return AggregateBootstrapResults(
        n_boot=n_boot,
        n_valid=len(vol_entropy_corr_boot),
        n_individuals=N,
        archetype_cosines=archetype_cosines,
        archetype_full_recovery_rate=float(np.mean(full_recovery)),
        vol_entropy_corr_boot=vol_entropy_corr_boot,
        vol_entropy_ci=(
            float(np.percentile(vol_entropy_corr_boot, 2.5)),
            float(np.percentile(vol_entropy_corr_boot, 97.5)),
        ),
        mse_csr_boot=mse_csr_boot,
        mse_uniform_boot=mse_uniform_boot,
        mse_reduction_ci=(
            float(np.percentile(mse_reduction, 2.5)),
            float(np.percentile(mse_reduction, 97.5)),
        ),
        n_failed=n_failed,
        elapsed_seconds=elapsed,
    )


# ============================================================================
# Reporting and export
# ============================================================================

def print_report(
    results: AggregateBootstrapResults,
    orig_vol_entropy_corr: float,
    orig_mse_csr: float,
) -> None:
    """Print publication-oriented aggregate bootstrap report."""
    print(f"\n{'=' * 70}")
    print("AGGREGATE BOOTSTRAP RESULTS")
    print(f"{'=' * 70}")
    print(f"  Requested replications: {results.n_boot}")
    print(f"  Valid replications:     {results.n_valid}")
    print(f"  Failed replications:    {results.n_failed}")
    print(f"  Total time:             {results.elapsed_seconds:.1f}s")

    print("\n--- Archetype reproducibility ---")
    arch_means = results.archetype_cosines.mean(axis=0)
    arch_sds = results.archetype_cosines.std(axis=0)
    for k, (m, sd) in enumerate(zip(arch_means, arch_sds), start=1):
        print(f"  Archetype {k}: mean cosine = {m:.4f} (SD = {sd:.4f})")
    print(
        f"  All-archetype recovery (all > {ARCHETYPE_RECOVERY_THRESHOLD}): "
        f"{100 * results.archetype_full_recovery_rate:.1f}%"
    )

    print("\n--- Volume-entropy correlation ---")
    print(f"  Original estimate:       r = {orig_vol_entropy_corr:.4f}")
    print(f"  Bootstrap mean:          r = {results.vol_entropy_corr_boot.mean():.4f}")
    print(f"  Bootstrap SD:            {results.vol_entropy_corr_boot.std():.4f}")
    print(
        f"  95% percentile CI:       "
        f"[{results.vol_entropy_ci[0]:.4f}, {results.vol_entropy_ci[1]:.4f}]"
    )

    print("\n--- Reconstruction MSE: CSR vs. uniform salience ---")
    mse_reduction = results.mse_uniform_boot - results.mse_csr_boot
    mse_red_mean = float(mse_reduction.mean())
    mse_red_pct = 100 * mse_red_mean / float(results.mse_uniform_boot.mean())
    print(f"  Original CSR MSE:        {orig_mse_csr:.4f}")
    print(
        f"  Bootstrap CSR MSE:       mean = {results.mse_csr_boot.mean():.4f}, "
        f"SD = {results.mse_csr_boot.std():.4f}"
    )
    print(
        f"  Bootstrap uniform MSE:   mean = {results.mse_uniform_boot.mean():.4f}, "
        f"SD = {results.mse_uniform_boot.std():.4f}"
    )
    print(f"  Mean MSE reduction:      {mse_red_mean:.4f} ({mse_red_pct:.1f}%)")
    print(
        f"  95% percentile CI:       "
        f"[{results.mse_reduction_ci[0]:.4f}, {results.mse_reduction_ci[1]:.4f}]"
    )


def export_results(
    results: AggregateBootstrapResults,
    output_prefix: str = "csr_bootstrap_aggregate",
) -> None:
    """Export replication-level aggregate bootstrap results."""
    rep_df = pd.DataFrame({
        "replication": np.arange(1, results.n_valid + 1),
        "vol_entropy_corr": results.vol_entropy_corr_boot,
        "mse_csr": results.mse_csr_boot,
        "mse_uniform": results.mse_uniform_boot,
        "mse_reduction": results.mse_uniform_boot - results.mse_csr_boot,
    })
    for k in range(results.archetype_cosines.shape[1]):
        rep_df[f"archetype_{k + 1}_cosine"] = results.archetype_cosines[:, k]

    path = f"{output_prefix}_replications.csv"
    rep_df.to_csv(path, index=False)
    print(f"\nExported aggregate bootstrap replications: {path}")


# ============================================================================
# Main
# ============================================================================

def main(
    n_boot: int = 500,
    country_filter: str = "US",
    ordinal_transform: str = "expected",
    n_archetypes: int = N_ARCHETYPES,
    export: bool = True,
    seed: int = RANDOM_SEED,
) -> Dict[str, object]:
    print("\n" + "=" * 70)
    print("CSR AGGREGATE BOOTSTRAP")
    print("=" * 70)

    csv_path = download_big5_data()
    data = load_and_prepare_data(csv_path, country_filter=country_filter)
    responses = data[EXTRAVERSION_ITEMS].values
    sum_scores = data["sum_score"].values

    print(f"\n{'=' * 70}")
    print("ORIGINAL CSR FIT")
    print(f"{'=' * 70}")
    orig_model = fit_csr_model(
        responses,
        sum_scores,
        model_name="Original CSR",
        ordinal_transform=ordinal_transform,
        verbose=True,
    )

    _, orig_centroids, orig_arch_stats = extract_archetypes(
        orig_model.salience,
        n_archetypes=n_archetypes,
    )
    print(f"\n  Original archetypes: {orig_arch_stats['variance_explained'] * 100:.1f}% variance explained")

    orig_ve_corr, _ = safe_corr(orig_model.volume, orig_model.entropy)

    results = run_aggregate_bootstrap(
        responses=responses,
        orig_centroids=orig_centroids,
        n_boot=n_boot,
        ordinal_transform=ordinal_transform,
        n_archetypes=n_archetypes,
        seed=seed,
    )

    print_report(
        results,
        orig_vol_entropy_corr=orig_ve_corr,
        orig_mse_csr=orig_model.reconstruction_mse,
    )

    if export:
        export_results(results)

    print("\n" + "=" * 70)
    print("AGGREGATE BOOTSTRAP COMPLETE")
    print("=" * 70)

    return {
        "orig_model": orig_model,
        "orig_centroids": orig_centroids,
        "orig_vol_entropy_corr": orig_ve_corr,
        "aggregate_results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSR aggregate case-resampling bootstrap",
    )
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--country", type=str, default="US")
    parser.add_argument(
        "--transform",
        type=str,
        default="expected",
        choices=["none", "midpoint", "expected", "rank"],
    )
    parser.add_argument("--archetypes", type=int, default=N_ARCHETYPES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-export", action="store_true")

    args, _ = parser.parse_known_args()

    country = None if args.country.lower() == "all" else args.country

    main(
        n_boot=args.n_boot,
        country_filter=country,
        ordinal_transform=args.transform,
        n_archetypes=args.archetypes,
        export=not args.no_export,
        seed=args.seed,
    )
