"""
Study: Robustness of the fixed-threshold expected-value (EV) transform.

Tests CSR recovery (eta and salience) when ordinal Likert data are generated
under conditions that depart from the symmetric, evenly-spaced threshold
assumption used by the package's default expected-value transformation.

Factorial design (3 x 3 x 3 = 27 cells):
  K (categories):           3, 5, 7
  Marginal shape:           symmetric, floor, ceiling
                            (global shift of generative thresholds)
  Threshold heterogeneity:  homogeneous, moderate, strong
                            (per-item additive perturbation of thresholds)

Estimation paths compared per cell:
  oracle  - CSR on continuous z (pre-categorization); ceiling baseline.
  direct  - CSR on integer codes y, no transformation; model-free alternative.
  ev      - CSR on EV-transformed y using fixed default thresholds; the path
            under test.

Outcomes: salience cosine similarity, eta Pearson correlation.

Run:
  python csr_ev_transform.py
  python csr_ev_transform.py --n-reps 50 --seed 7
"""

import argparse
import time
from typing import Dict, List, Tuple, Any

import numpy as np

from csr_main import (
    generate_csr_data,
    fit_csr,
    get_default_thresholds,
    eval_recovery,
)


# ===========================================================
# Factor levels
# ===========================================================

K_LEVELS = [3, 5, 7]
SHAPE_LEVELS = ["symmetric", "floor", "ceiling"]
HET_LEVELS = ["homogeneous", "moderate", "strong"]
PATHS = ["oracle", "direct", "ev"]
METRICS = ["sal_cos", "eta_pearson"]

SHAPE_SHIFT = {"symmetric": 0.0, "floor": 1.0, "ceiling": -1.0}
HET_DELTA = {"homogeneous": 0.0, "moderate": 0.3, "strong": 0.7}

SHAPE_INT = {"symmetric": 0, "floor": 1, "ceiling": 2}
HET_INT = {"homogeneous": 0, "moderate": 1, "strong": 2}


# ===========================================================
# Generative thresholds (per cell, per replication)
# ===========================================================

def build_threshold_matrix(
    K: int,
    shape: str,
    heterogeneity: str,
    z_mean: float,
    J: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build a J x (K-1) matrix of per-item generative thresholds.

    The base threshold vector is the package default for K, centered on the
    realized z mean and globally shifted by SHAPE_SHIFT[shape]. Per-item
    additive perturbations of magnitude HET_DELTA[heterogeneity] preserve
    threshold ordering within an item.
    """
    base = get_default_thresholds(K).astype(float)            # (K-1,)
    center = z_mean + SHAPE_SHIFT[shape]
    base_centered = base + center                              # (K-1,)

    delta = HET_DELTA[heterogeneity]
    if delta > 0.0:
        per_item_shift = rng.uniform(-delta, delta, size=J)
    else:
        per_item_shift = np.zeros(J)

    tau_matrix = np.tile(base_centered, (J, 1))                # (J, K-1)
    tau_matrix = tau_matrix + per_item_shift[:, None]
    return tau_matrix


def digitize_per_item(z: np.ndarray, tau_matrix: np.ndarray) -> np.ndarray:
    """Apply per-item thresholds to continuous z, returning integer codes 1..K."""
    N, J = z.shape
    y = np.zeros_like(z)
    for j in range(J):
        y[:, j] = np.digitize(z[:, j], tau_matrix[j]) + 1
    return y.astype(float)


def column_collapsed(y: np.ndarray) -> bool:
    """Detect any column that took only one value (categorization failure)."""
    return bool(np.any(np.all(y == y[0:1, :], axis=0)))


# ===========================================================
# Replication
# ===========================================================

def make_seed(base_seed: int, K: int, shape: str, het: str, rep_idx: int) -> int:
    """Deterministic seed across cell x rep."""
    return (
        base_seed * 1_000_003
        + rep_idx * 997
        + K * 31
        + SHAPE_INT[shape] * 7
        + HET_INT[het]
    )


def run_one_rep(
    K: int,
    shape: str,
    het: str,
    rep_idx: int,
    base_seed: int,
    N: int,
    J: int,
    noise_sd: float,
) -> Dict[str, Any]:
    """Run one replication. Returns dict of per-path metric values, or degenerate."""
    seed = make_seed(base_seed, K, shape, het, rep_idx)

    # 1. Generate continuous data (no Likert step)
    data = generate_csr_data(
        N=N, J=J, alpha=0.35,
        signal_scale=3.0, noise_sd=noise_sd,
        likert=None, seed=seed,
    )
    z = data["z"]
    S_true = data["S_true"]
    eta_true = data["eta_true"]

    # 2. Build true generative thresholds and digitize
    rng = np.random.default_rng(seed + 7)
    z_mean = float(np.mean(z))
    tau_matrix = build_threshold_matrix(K, shape, het, z_mean, J, rng)
    y_int = digitize_per_item(z, tau_matrix)

    if column_collapsed(y_int):
        return {"degenerate": True}

    # 3. Three estimation paths
    out: Dict[str, Any] = {"degenerate": False}

    # Oracle: continuous z
    res_oracle = fit_csr(
        z, ordinal_transform="none",
        sigma2=noise_sd ** 2, verbose=False, seed=seed,
    )
    m_oracle = eval_recovery(
        res_oracle.S, res_oracle.eta, S_true, eta_true,
        exclude_flatliners=res_oracle.flatliners,
    )

    # Direct: integer codes, no transform
    res_direct = fit_csr(
        y_int, ordinal_transform="none",
        K=K,
        verbose=False, seed=seed,
    )
    m_direct = eval_recovery(
        res_direct.S, res_direct.eta, S_true, eta_true,
        exclude_flatliners=res_direct.flatliners,
    )

    # EV: integer codes, fixed default thresholds
    res_ev = fit_csr(
        y_int, ordinal_transform="expected",
        K=K,
        verbose=False, seed=seed,
    )
    m_ev = eval_recovery(
        res_ev.S, res_ev.eta, S_true, eta_true,
        exclude_flatliners=res_ev.flatliners,
    )

    out["oracle"] = {
        "sal_cos": m_oracle["salience_cosine"],
        "eta_pearson": m_oracle["eta_pearson"],
    }
    out["direct"] = {
        "sal_cos": m_direct["salience_cosine"],
        "eta_pearson": m_direct["eta_pearson"],
    }
    out["ev"] = {
        "sal_cos": m_ev["salience_cosine"],
        "eta_pearson": m_ev["eta_pearson"],
    }
    return out


# ===========================================================
# Aggregation and reporting
# ===========================================================

def aggregate_cells(
    cell_results: Dict[Tuple[int, str, str], Dict[str, Dict[str, List[float]]]],
) -> List[Dict[str, Any]]:
    """Convert per-rep lists into per-cell mean/sd rows."""
    rows: List[Dict[str, Any]] = []
    for K in K_LEVELS:
        for shape in SHAPE_LEVELS:
            for het in HET_LEVELS:
                cell = cell_results[(K, shape, het)]
                row: Dict[str, Any] = {"K": K, "shape": shape, "heterogeneity": het}
                for p in PATHS:
                    for m in METRICS:
                        vals = cell[p][m]
                        row[f"{p}_{m}_mean"] = (
                            float(np.mean(vals)) if vals else float("nan")
                        )
                        row[f"{p}_{m}_sd"] = (
                            float(np.std(vals)) if vals else float("nan")
                        )
                row["n_valid"] = (
                    len(cell[PATHS[0]][METRICS[0]]) if cell[PATHS[0]][METRICS[0]] else 0
                )
                rows.append(row)
    return rows


def print_summary_tables(rows: List[Dict[str, Any]]) -> None:
    """Print one block per K with the 3x3 (shape x heterogeneity) grid."""
    for K in K_LEVELS:
        print(f"\n--- K = {K} categories ---")
        header = (
            f"{'shape':<11} {'heterog':<13} | "
            f"{'oracle s_cos':>13} {'direct s_cos':>13} {'EV s_cos':>10} | "
            f"{'oracle η_r':>11} {'direct η_r':>11} {'EV η_r':>8}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            if r["K"] != K:
                continue
            print(
                f"{r['shape']:<11} {r['heterogeneity']:<13} | "
                f"{r['oracle_sal_cos_mean']:>13.3f} "
                f"{r['direct_sal_cos_mean']:>13.3f} "
                f"{r['ev_sal_cos_mean']:>10.3f} | "
                f"{r['oracle_eta_pearson_mean']:>11.3f} "
                f"{r['direct_eta_pearson_mean']:>11.3f} "
                f"{r['ev_eta_pearson_mean']:>8.3f}"
            )


def print_ev_minus_direct(rows: List[Dict[str, Any]]) -> None:
    """Per-cell delta of EV vs direct: positive means EV helps."""
    print("\n\n=== EV minus Direct (positive = EV helps over raw integers) ===")
    for K in K_LEVELS:
        print(f"\n--- K = {K} categories ---")
        header = (
            f"{'shape':<11} {'heterog':<13} | "
            f"{'Δ s_cos':>10} {'Δ η_r':>10}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            if r["K"] != K:
                continue
            d_sal = r["ev_sal_cos_mean"] - r["direct_sal_cos_mean"]
            d_eta = r["ev_eta_pearson_mean"] - r["direct_eta_pearson_mean"]
            print(
                f"{r['shape']:<11} {r['heterogeneity']:<13} | "
                f"{d_sal:>+10.3f} {d_eta:>+10.3f}"
            )


# ===========================================================
# Main study
# ===========================================================

def study_ev_robustness(
    n_reps: int = 30,
    base_seed: int = 42,
    N: int = 800,
    J: int = 5,
    noise_sd: float = 0.15,
) -> Dict[str, Any]:
    """Run the full 3 x 3 x 3 robustness study."""
    print("=" * 72)
    print("Study: EV transform robustness under threshold misspecification")
    print("=" * 72)
    total_cells = len(K_LEVELS) * len(SHAPE_LEVELS) * len(HET_LEVELS)
    print(f"N={N}, J={J}, noise_sd={noise_sd}, n_reps={n_reps}, base_seed={base_seed}")
    print(f"Cells: {total_cells}\n")

    cell_results: Dict[Tuple[int, str, str], Dict[str, Dict[str, List[float]]]] = {}
    n_degenerate = 0
    cell_idx = 0
    overall_start = time.time()

    for K in K_LEVELS:
        for shape in SHAPE_LEVELS:
            for het in HET_LEVELS:
                cell_idx += 1
                cell_start = time.time()

                cell_results[(K, shape, het)] = {
                    p: {m: [] for m in METRICS} for p in PATHS
                }
                cell_degen = 0
                for rep in range(n_reps):
                    out = run_one_rep(
                        K, shape, het, rep, base_seed,
                        N=N, J=J, noise_sd=noise_sd,
                    )
                    if out.get("degenerate", False):
                        n_degenerate += 1
                        cell_degen += 1
                        continue
                    for p in PATHS:
                        for m in METRICS:
                            cell_results[(K, shape, het)][p][m].append(out[p][m])

                cell_elapsed = time.time() - cell_start
                avg_per_cell = (time.time() - overall_start) / cell_idx
                remaining = avg_per_cell * (total_cells - cell_idx)

                degen_str = f"  [{cell_degen} degen]" if cell_degen else ""
                eta_str = (
                    f"  ETA {int(remaining // 60):d}m {int(remaining % 60):02d}s"
                    if cell_idx < total_cells else ""
                )
                print(
                    f"  [{cell_idx:2d}/{total_cells}] "
                    f"K={K} shape={shape:<9} het={het:<12} "
                    f"{cell_elapsed:5.1f}s{degen_str}{eta_str}"
                )

    total_elapsed = time.time() - overall_start
    print(f"\n  Total: {int(total_elapsed // 60):d}m {int(total_elapsed % 60):02d}s")

    rows = aggregate_cells(cell_results)
    print_summary_tables(rows)
    print_ev_minus_direct(rows)

    if n_degenerate > 0:
        print(
            f"\n[Note] {n_degenerate} replication(s) skipped due to category collapse "
            f"(extreme floor/ceiling or strong heterogeneity)."
        )

    return {
        "study": "ev_robustness",
        "summary": rows,
        "n_degenerate": n_degenerate,
        "config": {
            "n_reps": n_reps, "base_seed": base_seed,
            "N": N, "J": J, "noise_sd": noise_sd,
        },
    }


# ===========================================================
# CLI
# ===========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-reps", type=int, default=30,
                   help="Replications per cell (default: 30)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed (default: 42)")
    p.add_argument("--N", type=int, default=800,
                   help="Sample size (default: 800)")
    p.add_argument("--J", type=int, default=5,
                   help="Number of items (default: 5)")
    p.add_argument("--noise-sd", type=float, default=0.15,
                   help="Generative noise SD (default: 0.15)")
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    study_ev_robustness(
        n_reps=args.n_reps,
        base_seed=args.seed,
        N=args.N,
        J=args.J,
        noise_sd=args.noise_sd,
    )


if __name__ == "__main__":
    main()
