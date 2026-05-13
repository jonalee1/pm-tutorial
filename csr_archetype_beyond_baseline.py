"""
csr_archetype_null_test.py
==========================
Test whether the empirical 3-archetype solution from Section 5 / Table 7
is reproducible by the bounded-allocation null.

For each replication of B1 (uniform pi) and B2 (empirical pi):
  1. Generate row-volume-preserving bounded null data on {1..K}
  2. Fit CSR (ordinal_transform="expected", gamma=0)  -- same as Section 5
  3. Apply CLR + K-means at K=3                       -- same pipeline
  4. Record:
       - variance explained (in original simplex space)
       - mean Aitchison centroid distance
       - silhouette score (in CLR space)
       - smallest-cluster proportion (size balance)

Compare against empirical reference values from emp_out_3.txt:
  variance explained       : 0.114
  mean Aitchison distance  : 4.82
  cluster size balance     : 35.6 / 32.9 / 31.4 (%)

Empirical archetype centroids (Table 7) are also embedded so a B2 null
centroid panel can be displayed alongside them for qualitative comparison.

Run
---
  python csr_archetype_null_test.py               # default: n_reps=100
  python csr_archetype_null_test.py --quick       # n_reps=25
  python csr_archetype_null_test.py --n_reps 50
  python csr_archetype_null_test.py --csv arch_null.csv

Requires: csr_main.py and csr_section4x_simulation.py in the same directory.
Author: Jonathan Lee
"""

from __future__ import annotations

import argparse
import csv
import time
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from csr_main import fit_csr
from csr_section4x_simulation import (
    EMPIRICAL_PI_EXTRAVERSION,
    gen_rowvolume_null,
)


RANDOM_SEED = 42   # matches csr_empirical_study.py
K_ARCH = 3


# ===========================================================
# CLR + K-means  (matches csr_empirical_study.py:extract_archetypes)
# ===========================================================

def clr_transform(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S_safe = S + eps
    log_S = np.log(S_safe)
    return log_S - log_S.mean(axis=1, keepdims=True)


def archetype_metrics(salience: np.ndarray, K: int = 3) -> Dict[str, Any]:
    """Run CLR + K-means and return diagnostic metrics."""
    S_clr = clr_transform(salience)

    km = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(S_clr)

    J = salience.shape[1]
    centroids = np.zeros((K, J))
    for k in range(K):
        mask = labels == k
        if mask.any():
            centroids[k] = salience[mask].mean(axis=0)

    within_ss = 0.0
    for k in range(K):
        mask = labels == k
        if mask.any():
            within_ss += float(np.sum((salience[mask] - centroids[k]) ** 2))
    total_ss = float(np.sum((salience - salience.mean(axis=0)) ** 2))
    var_explained = (total_ss - within_ss) / total_ss if total_ss > 0 else 0.0

    centroids_clr = clr_transform(centroids)
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dists.append(float(np.sqrt(np.sum((centroids_clr[i] - centroids_clr[j]) ** 2))))
    mean_dist = float(np.mean(dists)) if dists else float("nan")

    try:
        sil = float(silhouette_score(S_clr, labels))
    except Exception:
        sil = float("nan")

    sizes = np.bincount(labels, minlength=K) / labels.size
    return {
        "variance_explained": float(var_explained),
        "mean_centroid_distance": mean_dist,
        "silhouette": sil,
        "min_cluster_pct": float(sizes.min()),
        "max_cluster_pct": float(sizes.max()),
        "cluster_sizes_pct": sizes.tolist(),
        "centroids": centroids,
        "labels": labels,
    }


# ===========================================================
# Run null condition repeatedly
# ===========================================================

def run_null_archetypes(
    name: str,
    gen_kwargs: Dict[str, Any],
    n_reps: int,
    seed_base: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    centroids_last = None
    t0 = time.time()
    for rep in range(n_reps):
        seed = seed_base + rep
        data = gen_rowvolume_null(seed=seed, **gen_kwargs)
        res = fit_csr(
            data["y"],
            ordinal_transform="expected",
            gamma=0.0,
            max_iter=200,
            tol=1e-6,
            verbose=False,
            seed=42,
        )
        m = archetype_metrics(res.S, K=K_ARCH)
        rows.append(m)
        centroids_last = m["centroids"]
        step = max(1, n_reps // 5)
        if (rep + 1) % step == 0 or rep == n_reps - 1:
            mean_ve = float(np.mean([r["variance_explained"] for r in rows]))
            print(f"    rep {rep + 1:>4d}/{n_reps}   "
                  f"running mean var.expl. = {mean_ve:.3f}")
    elapsed = time.time() - t0

    def arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=float)

    return {
        "name": name,
        "n_reps": n_reps,
        "var_explained_mean": float(np.mean(arr("variance_explained"))),
        "var_explained_sd": float(np.std(arr("variance_explained"), ddof=1)),
        "var_explained_max": float(np.max(arr("variance_explained"))),
        "var_explained_p95": float(np.percentile(arr("variance_explained"), 95)),
        "var_explained_p99": float(np.percentile(arr("variance_explained"), 99)),
        "centroid_dist_mean": float(np.mean(arr("mean_centroid_distance"))),
        "centroid_dist_sd": float(np.std(arr("mean_centroid_distance"), ddof=1)),
        "silhouette_mean": float(np.nanmean(arr("silhouette"))),
        "silhouette_sd": float(np.nanstd(arr("silhouette"), ddof=1)),
        "min_cluster_pct_mean": float(np.mean(arr("min_cluster_pct"))),
        "max_cluster_pct_mean": float(np.mean(arr("max_cluster_pct"))),
        "centroids_last": centroids_last,
        "elapsed_sec": elapsed,
    }


# ===========================================================
# Empirical reference (from emp_out_3.txt, 2-component model)
# ===========================================================

EMPIRICAL_REF = {
    "var_explained":       0.114,
    "mean_centroid_dist":  4.8219,
    "cluster_pcts":        [0.356, 0.329, 0.314],
    "centroids": np.array([
        [0.046, 0.100, 0.188, 0.056, 0.204, 0.145, 0.156, 0.004, 0.053, 0.047],
        [0.058, 0.074, 0.152, 0.064, 0.095, 0.134, 0.069, 0.132, 0.191, 0.030],
        [0.021, 0.183, 0.133, 0.050, 0.135, 0.387, 0.001, 0.020, 0.045, 0.025],
    ]),
    "labels": ["Social Initiators", "Attention-Comfortable", "Verbal Expressors"],
    "top_items": [
        "E5=.204, E3=.188, E7=.156",
        "E9=.191, E3=.152, E6=.134",
        "E6=.387, E2=.183, E5=.135",
    ],
}


# ===========================================================
# Output
# ===========================================================

def print_summary_table(b1: Dict[str, Any], b2: Dict[str, Any]) -> None:
    emp = EMPIRICAL_REF
    print("\n" + "=" * 92)
    print(f"  ARCHETYPE COMPARISON (K-means K={K_ARCH} on CLR-transformed salience)")
    print("=" * 92)
    print(f"  {'Source':<22s}  {'Var.Expl.':>16s}  {'Centroid d':>16s}  "
          f"{'Silhouette':>16s}  {'min cluster':>12s}")
    print("  " + "-" * 90)
    for r in [b1, b2]:
        ve = f"{r['var_explained_mean']:.3f} \u00b1 {r['var_explained_sd']:.3f}"
        cd = f"{r['centroid_dist_mean']:.3f} \u00b1 {r['centroid_dist_sd']:.3f}"
        si = f"{r['silhouette_mean']:.3f} \u00b1 {r['silhouette_sd']:.3f}"
        mc = f"{r['min_cluster_pct_mean'] * 100:>5.1f}%"
        print(f"  {r['name']:<22s}  {ve:>16s}  {cd:>16s}  {si:>16s}  {mc:>12s}")
    print("  " + "-" * 90)
    ve_e = f"{emp['var_explained']:.3f}"
    cd_e = f"{emp['mean_centroid_dist']:.3f}"
    mc_e = f"{min(emp['cluster_pcts']) * 100:>5.1f}%"
    print(f"  {'Empirical (Table 7)':<22s}  {ve_e:>16s}  {cd_e:>16s}  "
          f"{'(not computed)':>16s}  {mc_e:>12s}")
    print("=" * 92)


def print_centroid_panel(b2: Dict[str, Any]) -> None:
    emp = EMPIRICAL_REF
    items_hdr = ("E1     E2     E3     E4     E5     E6     E7     E8     E9    E10")

    print("\nEmpirical archetype centroids (Table 7):")
    print(f"  Item:                {items_hdr}")
    for k, (lbl, top, c) in enumerate(zip(emp["labels"], emp["top_items"], emp["centroids"])):
        c_str = "  ".join(f"{v:5.3f}" for v in c)
        print(f"  {k+1}. {lbl:<23s}{c_str}     top: {top}")

    print("\nB2 null centroids from one replication (for visual comparison):")
    print(f"  Item:                {items_hdr}")
    for k in range(K_ARCH):
        c = b2["centroids_last"][k]
        c_str = "  ".join(f"{v:5.3f}" for v in c)
        idx = np.argsort(c)[::-1][:3]
        top = ", ".join(f"E{j+1}={c[j]:.3f}" for j in idx)
        print(f"  {k+1}. (null cluster {k+1})       {c_str}     top: {top}")


def print_effect_size(b1: Dict[str, Any], b2: Dict[str, Any]) -> None:
    emp = EMPIRICAL_REF
    emp_ve = emp["var_explained"]
    print("\nEffect-size summary:")
    print(f"  Empirical variance explained:                {emp_ve:.3f}")
    print(f"  B1 null max across {b1['n_reps']:>3d} reps:                "
          f"{b1['var_explained_max']:.3f}")
    print(f"  B1 null 95th percentile:                     {b1['var_explained_p95']:.3f}")
    print(f"  B1 null 99th percentile:                     {b1['var_explained_p99']:.3f}")
    print(f"  B2 null max across {b2['n_reps']:>3d} reps:                "
          f"{b2['var_explained_max']:.3f}")
    print(f"  B2 null 95th percentile:                     {b2['var_explained_p95']:.3f}")
    print(f"  B2 null 99th percentile:                     {b2['var_explained_p99']:.3f}")
    print(f"  Empirical / B2 mean ratio:                   "
          f"{emp_ve / b2['var_explained_mean']:.2f}x")


def write_csv(b1: Dict[str, Any], b2: Dict[str, Any], path: str) -> None:
    fields = [
        "name", "n_reps",
        "var_explained_mean", "var_explained_sd",
        "var_explained_max", "var_explained_p95", "var_explained_p99",
        "centroid_dist_mean", "centroid_dist_sd",
        "silhouette_mean", "silhouette_sd",
        "min_cluster_pct_mean", "max_cluster_pct_mean",
        "elapsed_sec",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in [b1, b2]:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n[CSV written] {path}")


# ===========================================================
# CLI
# ===========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Archetype null comparison test (Section 4.X follow-up)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--N", type=int, default=800, help="Persons per replication")
    p.add_argument("--J", type=int, default=10, help="Items")
    p.add_argument("--K", type=int, default=5, help="Likert categories")
    p.add_argument("--n_reps", type=int, default=100, help="Replications per condition")
    p.add_argument("--seed", type=int, default=20251104, help="Base random seed")
    p.add_argument("--T_mean", type=float, default=30.6)
    p.add_argument("--T_sd", type=float, default=9.0)
    p.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: n_reps=25 (overrides --n_reps)")
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_reps = 25
        print("[quick mode] n_reps = 25")

    print("=" * 72)
    print("  Archetype Null Test  (compare to empirical Section 5 / Table 7)")
    print("=" * 72)
    print(f"  N = {args.N}, J = {args.J}, K-Likert = {args.K}, "
          f"n_reps = {args.n_reps}, K-arch = {K_ARCH}")
    print(f"  Empirical reference: var.expl = {EMPIRICAL_REF['var_explained']:.3f}, "
          f"centroid d = {EMPIRICAL_REF['mean_centroid_dist']:.3f}")

    common = {"N": args.N, "J": args.J, "K": args.K,
              "T_mean": args.T_mean, "T_sd": args.T_sd}

    print("\n>>> Null B1: uniform pi (pure mechanical)")
    b1 = run_null_archetypes(
        "B1 uniform pi",
        {**common, "item_marginals": None},
        n_reps=args.n_reps, seed_base=args.seed,
    )

    print("\n>>> Null B2: empirical pi (preserves item marginals)")
    b2 = run_null_archetypes(
        "B2 empirical pi",
        {**common, "item_marginals": EMPIRICAL_PI_EXTRAVERSION},
        n_reps=args.n_reps, seed_base=args.seed + 5000,
    )

    print_summary_table(b1, b2)
    print_centroid_panel(b2)
    print_effect_size(b1, b2)

    if args.csv is not None:
        write_csv(b1, b2, args.csv)

    total = b1["elapsed_sec"] + b2["elapsed_sec"]
    print(f"\n[Total elapsed: {total:.1f} sec]")


if __name__ == "__main__":
    main()
