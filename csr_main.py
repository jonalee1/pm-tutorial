"""
csr_main.py
===========
Continuous Salience-Response Model

Core model:
    z_ij = eta_i * s_ij + eps    (latent continuous)
    y_ij = k  iff  τ_{k-1} < z_ij ≤ τ_k   (ordinal observation)

Salience vector s_i lives on the simplex: s_ij >= 0, sum_j s_ij = 1

CSR identifies:
  - Salience (s_i): Within-person relative allocation (SHAPE)
  - Volume (η_i): Overall response intensity (LEVEL)

For ordinal data, transforms y → latent z, then runs continuous CSR.

Run:
  python csr_main.py

Author: Jonathan Lee
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from scipy.integrate import quad


# ===========================================================
# Utilities
# ===========================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
    """Pearson correlation with safe fallbacks."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan"), False
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan"), False
    return float(np.corrcoef(x, y)[0, 1]), True


def safe_rank_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
    """Spearman rank correlation."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan"), False
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan"), False

    def rankdata(arr):
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1)
        return ranks

    corr, valid = safe_corr(rankdata(x), rankdata(y))
    return corr, valid


def cosine_similarity_rows(S: np.ndarray, T: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine similarity between two (N, J) matrices."""
    Sn = np.linalg.norm(S, axis=1, keepdims=True) + eps
    Tn = np.linalg.norm(T, axis=1, keepdims=True) + eps
    return np.sum(S * T, axis=1) / (Sn[:, 0] * Tn[:, 0])


def softmax_stable(x: np.ndarray, axis: int = -1, tau: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    z = x / max(tau, 1e-12)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def detect_flatliners(y: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Return boolean mask of individuals with constant responses."""
    return np.ptp(y, axis=1) < tol


def compute_sparsity(S: np.ndarray, tol: float = 1e-6) -> float:
    """Mean proportion of near-zero salience entries."""
    near_zero = np.abs(S) < tol
    return float(np.mean(near_zero))


def compute_entropy(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy of each salience vector (row)."""
    S_safe = np.clip(S, eps, 1.0)
    return -np.sum(S_safe * np.log(S_safe), axis=1)


def compute_snr(y_signal: np.ndarray, noise_sd: float) -> Dict[str, float]:
    """Compute signal-to-noise ratio diagnostics."""
    mean_signal = float(np.mean(np.abs(y_signal)))
    snr = mean_signal / noise_sd if noise_sd > 0 else float("inf")
    return {"mean_signal": mean_signal, "noise_sd": noise_sd, "snr": snr}


def reference_salience(
    y: np.ndarray,
    mode: str = "minmax",
    tau: float = 1.0,
    eps: float = 1e-12
) -> np.ndarray:
    """Construct reference salience from responses."""
    y = np.asarray(y, dtype=float)
    N, J = y.shape

    if mode == "softmax":
        return softmax_stable(y, axis=1, tau=tau)

    if mode == "deviation":
        means = np.mean(y, axis=1, keepdims=True)
        dev = np.abs(y - means)
        s = np.sum(dev, axis=1, keepdims=True)
        r = np.where(s > eps, dev / s, np.full((N, J), 1.0 / J))
        return r

    # Default: minmax
    mn = np.min(y, axis=1, keepdims=True)
    mx = np.max(y, axis=1, keepdims=True)
    denom = np.maximum(mx - mn, eps)
    z = (y - mn) / denom
    s = np.sum(z, axis=1, keepdims=True)
    r = np.where(s > eps, z / s, np.full((N, J), 1.0 / J))
    return r


# ===========================================================
# Compositional helpers
# ===========================================================

def clr_transform(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Centered log-ratio (CLR) transformation for compositional data.

    Maps a row-stochastic matrix S (each row on the simplex) to
    unconstrained Euclidean coordinates suitable for standard
    distance-based methods (e.g., k-means clustering, PCA).

    Parameters
    ----------
    S : (N, J) array of compositional rows (rows sum to 1, all entries >= 0)
    eps : small constant added before log to avoid log(0)

    Returns
    -------
    (N, J) array; each row sums to 0.

    References
    ----------
    Aitchison (1986). The Statistical Analysis of Compositional Data.
    """
    S_safe = S + eps
    log_S = np.log(S_safe)
    geometric_mean = log_S.mean(axis=1, keepdims=True)
    return log_S - geometric_mean


def inverse_clr(S_clr: np.ndarray) -> np.ndarray:
    """
    Inverse CLR: map back from CLR coordinates to the simplex.

    Parameters
    ----------
    S_clr : (N, J) array of CLR coordinates

    Returns
    -------
    (N, J) array; each row sums to 1.
    """
    S_exp = np.exp(S_clr)
    return S_exp / S_exp.sum(axis=1, keepdims=True)


# ===========================================================
# Ordinal → Latent Continuous Transformation
# ===========================================================

def get_default_thresholds(K: int = 5) -> np.ndarray:
    """Default symmetric thresholds for K categories."""
    if K == 5:
        return np.array([-1.5, -0.5, 0.5, 1.5])
    elif K == 7:
        return np.array([-2.0, -1.2, -0.4, 0.4, 1.2, 2.0])
    elif K == 4:
        return np.array([-1.0, 0.0, 1.0])
    elif K == 3:
        return np.array([-0.5, 0.5])
    else:
        half_width = 0.5 * (K - 1) * 0.5
        return np.linspace(-half_width, half_width, K - 1)


def _truncated_logistic_mean(a: float, b: float, mu: float = 0.0) -> float:
    """Compute E[z | a < z ≤ b] for logistic distribution."""
    if np.isinf(a) and a < 0:
        a = mu - 20.0
    if np.isinf(b) and b > 0:
        b = mu + 20.0

    def logistic_pdf(x):
        t = np.exp(-(x - mu))
        return t / (1 + t) ** 2

    def integrand_num(x):
        return x * logistic_pdf(x)

    F_a = expit(a - mu)
    F_b = expit(b - mu)
    denom = F_b - F_a

    if denom < 1e-12:
        return (a + b) / 2.0

    num, _ = quad(integrand_num, a, b, limit=50)
    return num / denom


def _compute_category_expected_values(tau: np.ndarray, mu: float = 0.0) -> np.ndarray:
    """Precompute E[z | y=k] for all categories k = 1, ..., K."""
    K = len(tau) + 1
    expected = np.zeros(K)
    tau_ext = np.concatenate([[-np.inf], tau, [np.inf]])

    for k in range(K):
        a = tau_ext[k]
        b = tau_ext[k + 1]
        expected[k] = _truncated_logistic_mean(a, b, mu)

    return expected


def ordinal_to_latent(
    y: np.ndarray,
    tau: Optional[np.ndarray] = None,
    K: Optional[int] = None,
    method: str = "midpoint",
    epsilon: float = 0.1,
) -> np.ndarray:
    """
    Transform ordinal responses to latent continuous scale.

    Methods:
      "raw":       z = y (no transformation; admissible for CSR if y >= 1).
      "midpoint":  z = (tau_{k-1} + tau_k) / 2.
      "expected":  z = E[z | tau_{k-1} < z <= tau_k] under logistic latent.
                   NOTE: produces negative values for low categories under
                   symmetric thresholds; not admissible for CSR salience
                   estimation without further shifting.
      "positive_expected": same as "expected", then shifted so the lowest
                   category maps to `epsilon`. Admissible.
      "rank":      z = Phi^{-1}((rank - 0.5) / J) within person.

    Parameters
    ----------
    K : int, optional
        Number of Likert categories. When supplied, takes precedence over
        any inference from `y`. If both `K` and `tau` are None, K is
        inferred from `int(np.max(y))`; a warning is emitted whenever
        the data span looks suspect (smallest observed value > 1, or
        fewer unique values than the inferred K).
    tau : array-like, optional
        Threshold vector with length K-1. If passed, takes precedence and
        K is set to len(tau) + 1.
    epsilon : float
        Floor for the "positive_expected" transform. Only used when
        method == "positive_expected".
    """
    y = np.asarray(y, dtype=float)
    N, J = y.shape

    if method == "raw":
        return y.copy()

    if method == "rank":
        z = np.zeros_like(y)
        for i in range(N):
            order = np.argsort(y[i, :])
            ranks = np.zeros(J)
            ranks[order] = np.arange(1, J + 1)
            for j in range(J):
                tied = y[i, :] == y[i, j]
                if np.sum(tied) > 1:
                    ranks[tied] = np.mean(ranks[tied])
            z[i, :] = norm.ppf((ranks - 0.5) / J)
        return z

    # Resolve K and tau in a single, predictable order:
    #   1) tau passed -> K := len(tau) + 1 (ignore K argument)
    #   2) K passed   -> tau := get_default_thresholds(K)
    #   3) neither    -> infer K from data, warn if data span is suspect
    if tau is not None:
        tau = np.asarray(tau, dtype=float)
        K = len(tau) + 1
    elif K is not None:
        tau = get_default_thresholds(int(K))
    else:
        K_inferred = int(np.max(y))
        observed_min = int(np.min(y))
        n_unique = int(len(np.unique(y)))
        if observed_min > 1 or n_unique < K_inferred:
            import warnings
            warnings.warn(
                f"ordinal_to_latent: K not specified and inferred K={K_inferred} "
                f"from data with min={observed_min}, unique categories={n_unique}. "
                f"Pass K explicitly if the scale has unobserved low/high categories.",
                stacklevel=2,
            )
        K = K_inferred
        tau = get_default_thresholds(K)

    if method in ("expected", "positive_expected"):
        expected_values = _compute_category_expected_values(tau, mu=0.0)
        if method == "positive_expected":
            shift = -float(np.min(expected_values)) + float(epsilon)
            expected_values = expected_values + shift
        z = np.zeros_like(y)
        for k in range(1, K + 1):
            mask = (y == k)
            z[mask] = expected_values[k - 1]
        return z

    # Default: midpoint
    tau_lo = tau[0] - 1.0
    tau_hi = tau[-1] + 1.0
    tau_ext = np.concatenate([[tau_lo], tau, [tau_hi]])

    midpoints = np.zeros(K)
    for k in range(K):
        midpoints[k] = (tau_ext[k] + tau_ext[k + 1]) / 2.0

    z = np.zeros_like(y)
    for k in range(1, K + 1):
        mask = (y == k)
        z[mask] = midpoints[k - 1]

    return z


def is_likert(y: np.ndarray, max_categories: int = 10) -> bool:
    """Detect if data appears to be Likert/ordinal."""
    y_flat = y.ravel()

    if not np.allclose(y_flat, np.round(y_flat)):
        return False

    y_min, y_max = y_flat.min(), y_flat.max()
    if y_min < 1 or y_max > max_categories:
        return False

    if len(np.unique(y_flat)) < 2:
        return False

    return True


# ===========================================================
# CSR core: closed-form updates
# ===========================================================

def simplex_waterfilling(
    d: np.ndarray,
    c: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 200
) -> np.ndarray:
    """
    Solve simplex-constrained QP via KKT water-filling.

    minimize_s  sum_j (1/2 c_j s_j^2 - d_j s_j)
    s.t. s_j >= 0, sum_j s_j = 1
    """
    d = np.asarray(d, dtype=float)
    c = np.asarray(c, dtype=float)
    J = d.size
    c = np.maximum(c, 1e-12)

    def sum_s(nu: float) -> float:
        s = np.maximum(0.0, (d - nu) / c)
        return float(np.sum(s))

    nu_lo = float(np.min(d - c)) - 1.0
    nu_hi = float(np.max(d)) + 1.0

    for _ in range(50):
        if sum_s(nu_lo) >= 1.0:
            break
        nu_lo -= (nu_hi - nu_lo) * 2.0

    for _ in range(50):
        if sum_s(nu_hi) <= 1.0:
            break
        nu_hi += (nu_hi - nu_lo) * 2.0

    if sum_s(nu_lo) < 1.0 or sum_s(nu_hi) > 1.0:
        return np.full(J, 1.0 / J)

    for _ in range(max_iter):
        nu_mid = 0.5 * (nu_lo + nu_hi)
        s_mid = sum_s(nu_mid)
        if abs(s_mid - 1.0) <= tol:
            break
        if s_mid > 1.0:
            nu_lo = nu_mid
        else:
            nu_hi = nu_mid

    nu = 0.5 * (nu_lo + nu_hi)
    s = np.maximum(0.0, (d - nu) / c)

    s_sum = np.sum(s)
    if s_sum <= 1e-12:
        return np.full(J, 1.0 / J)
    return s / s_sum


def update_salience_row(
    y_i: np.ndarray,
    eta_i: float,
    r_i: Optional[np.ndarray],
    gamma: float,
    sigma2: float,
) -> np.ndarray:
    """Closed-form salience update via KKT water-filling."""
    y_i = np.asarray(y_i, dtype=float)
    sigma2 = max(sigma2, 1e-12)
    J = y_i.size

    c = (eta_i ** 2) / sigma2 + gamma * np.ones(J)
    d = (eta_i * y_i) / sigma2

    if r_i is not None and gamma > 0:
        d = d + gamma * np.asarray(r_i, dtype=float)

    return simplex_waterfilling(d=d, c=c)


def update_eta(y: np.ndarray, S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Closed-form eta update."""
    num = np.sum(S * y, axis=1)
    den = np.sum(S * S, axis=1) + eps
    eta = num / den
    return np.maximum(eta, eps)


def estimate_sigma2(y: np.ndarray, S: np.ndarray, eta: np.ndarray) -> float:
    """Estimate residual variance from current reconstruction."""
    yhat = eta[:, None] * S
    return float(np.mean((y - yhat) ** 2))


def compute_mae(y: np.ndarray, S: np.ndarray, eta: np.ndarray) -> float:
    """Compute mean absolute error."""
    yhat = eta[:, None] * S
    return float(np.mean(np.abs(y - yhat)))


def csr_loss(
    y: np.ndarray,
    S: np.ndarray,
    eta: np.ndarray,
    r: Optional[np.ndarray],
    gamma: float,
    sigma2: float,
) -> float:
    """Compute objective value for convergence monitoring."""
    yhat = eta[:, None] * S
    recon = 0.5 * np.mean((y - yhat) ** 2) / max(sigma2, 1e-12)
    reg_s = 0.0
    if r is not None and gamma > 0:
        reg_s = 0.5 * gamma * float(np.mean((S - r) ** 2))
    return recon + reg_s


# ===========================================================
# CSR fitting
# ===========================================================

@dataclass
class CSRResult:
    """Container for CSR estimation results."""
    S: np.ndarray
    eta: np.ndarray
    r: Optional[np.ndarray]
    tau: Optional[np.ndarray]
    z: Optional[np.ndarray]
    history: Dict[str, Any]
    flatliners: np.ndarray
    reconstruction_mse: float
    reconstruction_mae: float
    sparsity: float
    entropy: np.ndarray
    sigma2: float
    ordinal_transform: str


def fit_csr(
    y: np.ndarray,
    *,
    ordinal_transform: str = "auto",
    tau: Optional[np.ndarray] = None,
    K: Optional[int] = None,
    gamma: float = 0.0,
    sigma2: Optional[float] = None,
    estimate_sigma: bool = False,
    ref_mode: str = "minmax",
    ref_tau: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    flatliner_tol: float = 1e-8,
    verbose: bool = True,
    seed: int = 42,
) -> CSRResult:
    """
    Fit CSR by alternating closed-form updates.

    For ordinal/Likert data, transforms to latent continuous scale
    before fitting, then runs standard continuous CSR.

    Ordinal transform options:
      "auto": Detect Likert and use "midpoint" if found
      "none": No transformation (treat as continuous)
      "raw":  No transformation (alias of "none" for explicit admissibility)
      "midpoint": Threshold interval midpoints
      "expected": E[z|y] from truncated logistic (NOT admissible: yields
                  negative values; use only for sensitivity comparisons)
      "positive_expected": Expected, shifted to nonnegative (admissible)
      "rank": Within-person rank → normal scores

    K, tau resolution
    -----------------
    When ordinal_transform requires thresholds, the (K, tau) pair is
    resolved as follows:
      - If `tau` is passed: K := len(tau) + 1 (overrides any K argument).
      - Else if `K` is passed: tau := get_default_thresholds(K).
      - Else: K is inferred from int(np.max(y)); a warning is emitted
        if the data span looks suspect (min(y) > 1 or fewer unique
        categories than the inferred K). To avoid silent failures on
        sparse data, prefer passing K explicitly.
    """
    set_seed(seed)
    y_orig = np.asarray(y, dtype=float)
    N, J = y_orig.shape

    # Ordinal → Latent Transform
    z = None
    actual_transform = "none"

    if ordinal_transform == "auto":
        if is_likert(y_orig):
            ordinal_transform = "midpoint"
            if verbose:
                K_detected = int(np.max(y_orig))
                print(f"[CSR] Detected Likert-{K_detected} data, applying '{ordinal_transform}' transform")
        else:
            ordinal_transform = "none"

    # "raw" is treated as a no-op transform identical to "none" — it's an
    # explicit-admissibility alias that downstream callers can pass to
    # signal "I deliberately want CSR fit on the original scale".
    if ordinal_transform in ("none", "raw"):
        y = y_orig
        actual_transform = ordinal_transform
    else:
        # Resolve K and tau honoring user inputs.
        if tau is not None:
            tau_arr = np.asarray(tau, dtype=float)
            K_actual = len(tau_arr) + 1
        elif K is not None:
            K_actual = int(K)
            tau_arr = get_default_thresholds(K_actual)
        else:
            K_actual = int(np.max(y_orig))
            observed_min = int(np.min(y_orig))
            n_unique = int(len(np.unique(y_orig)))
            if observed_min > 1 or n_unique < K_actual:
                import warnings
                warnings.warn(
                    f"fit_csr: K not specified and inferred K={K_actual} from "
                    f"data with min={observed_min}, unique categories={n_unique}. "
                    f"Pass K explicitly if the scale has unobserved categories.",
                    stacklevel=2,
                )
            tau_arr = get_default_thresholds(K_actual)

        z = ordinal_to_latent(y_orig, tau=tau_arr, K=K_actual,
                              method=ordinal_transform)
        y = z
        tau = tau_arr  # so downstream CSRResult records the resolved tau
        actual_transform = ordinal_transform

        if verbose:
            print(f"[CSR] Transformed y → z using method='{ordinal_transform}' "
                  f"(K={K_actual})")
            print(f"      z range: [{z.min():.3f}, {z.max():.3f}], mean={z.mean():.3f}")

    # Detect flat-liners
    flatliners = detect_flatliners(y, tol=flatliner_tol)
    n_flat = int(np.sum(flatliners))
    if verbose and n_flat > 0:
        print(f"[CSR] Detected {n_flat} flat-liner(s); assigning uniform salience.")

    # Reference salience
    r = None
    if gamma > 0:
        r = reference_salience(y, mode=ref_mode, tau=ref_tau)
        if verbose:
            print(f"[CSR] Reference salience computed (mode={ref_mode})")

    # Initialize
    eta = np.maximum(np.sum(np.abs(y), axis=1), 1e-6)
    if r is not None:
        S = r.copy()
    else:
        S = np.full((N, J), 1.0 / J, dtype=float)

    sigma2_use = sigma2 if sigma2 is not None else 1.0

    history: Dict[str, Any] = {"loss": [], "delta": [], "iters": 0}

    if verbose:
        print(f"\n[CSR] Fitting N={N}, J={J}")
        print(f"      gamma={gamma}, sigma2={sigma2_use:.4g}")
        print("-" * 60)

    prev_loss = None
    for t in range(1, max_iter + 1):
        # S-update
        for i in range(N):
            if flatliners[i]:
                S[i, :] = 1.0 / J
            else:
                S[i, :] = update_salience_row(
                    y_i=y[i, :],
                    eta_i=float(eta[i]),
                    r_i=r[i, :] if r is not None else None,
                    gamma=float(gamma),
                    sigma2=float(sigma2_use),
                )

        # η-update
        eta = update_eta(y, S)

        # Optional sigma2 estimation
        if estimate_sigma and t >= 5:
            sigma2_use = max(estimate_sigma2(y, S, eta), 1e-6)

        # Monitor convergence
        cur_loss = csr_loss(y, S, eta, r, gamma, sigma2_use)
        history["loss"].append(cur_loss)

        if prev_loss is None:
            delta = float("inf")
        else:
            delta = abs(prev_loss - cur_loss) / (abs(prev_loss) + 1e-12)
        history["delta"].append(delta)
        history["iters"] = t

        if verbose and (t == 1 or t % 10 == 0 or t == max_iter or delta < tol):
            print(f"[iter {t:03d}] loss={cur_loss:.6e}  rel_delta={delta:.3e}")

        if prev_loss is not None and delta < tol:
            if verbose:
                print(f"[converged] at iteration {t}")
            break
        prev_loss = cur_loss

    # Final diagnostics
    reconstruction_mse = estimate_sigma2(y, S, eta)
    reconstruction_mae = compute_mae(y, S, eta)
    sparsity = compute_sparsity(S)
    entropy = compute_entropy(S)

    return CSRResult(
        S=S, eta=eta, r=r,
        tau=tau if actual_transform not in ("none", "raw") else None,
        z=z,
        history=history, flatliners=flatliners,
        reconstruction_mse=reconstruction_mse,
        reconstruction_mae=reconstruction_mae,
        sparsity=sparsity, entropy=entropy,
        sigma2=sigma2_use,
        ordinal_transform=actual_transform,
    )


# ===========================================================
# Data generation
# ===========================================================

def sample_simplex_dirichlet(N: int, J: int, alpha: float = 0.5) -> np.ndarray:
    """Sample N points from Dirichlet(alpha, ..., alpha) on J-simplex."""
    return np.random.dirichlet(alpha=np.full(J, alpha), size=N)


def likert_from_latent(z: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Convert latent continuous z to Likert categories via thresholds."""
    z = np.asarray(z, dtype=float)
    tau = np.asarray(tau, dtype=float)
    y = np.digitize(z, tau) + 1
    return y.astype(float)


def generate_csr_data(
    N: int = 800,
    J: int = 5,
    *,
    alpha: float = 0.35,
    eta_loc: float = 0.0,
    eta_scale: float = 0.4,
    signal_scale: float = 3.0,
    noise_sd: float = 0.15,
    n_flatliners: int = 0,
    likert: Optional[int] = None,
    likert_tau: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate synthetic CSR data."""
    set_seed(seed)

    S_true = sample_simplex_dirichlet(N, J, alpha=alpha)
    eta_true = np.exp(np.random.normal(loc=eta_loc, scale=eta_scale, size=N))
    eta_true = eta_true / np.mean(eta_true) * signal_scale

    y_signal = eta_true[:, None] * S_true

    snr_info = compute_snr(y_signal, noise_sd)
    z = y_signal + np.random.normal(0.0, noise_sd, size=(N, J))

    # Flat-liners
    flatliner_mask = np.zeros(N, dtype=bool)
    if n_flatliners > 0:
        flat_idx = np.random.choice(N, size=min(n_flatliners, N), replace=False)
        flatliner_mask[flat_idx] = True
        for i in flat_idx:
            z[i, :] = np.mean(z[i, :])
            S_true[i, :] = 1.0 / J

    # Likert transformation
    if likert is not None and likert >= 3:
        if likert_tau is None:
            likert_tau = get_default_thresholds(likert)
        y = likert_from_latent(z, likert_tau)
    else:
        y = z

    return {
        "y": y,
        "z": z,
        "y_signal": y_signal,
        "S_true": S_true,
        "eta_true": eta_true,
        "flatliners": flatliner_mask,
        "noise_sd": noise_sd,
        "snr_info": snr_info,
        "likert_tau": likert_tau if likert else None,
    }


# ===========================================================
# Evaluation
# ===========================================================

def eval_recovery(
    S_hat: np.ndarray,
    eta_hat: np.ndarray,
    S_true: np.ndarray,
    eta_true: np.ndarray,
    z_hat: Optional[np.ndarray] = None,
    z_true: Optional[np.ndarray] = None,
    exclude_flatliners: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Recovery diagnostics."""
    if exclude_flatliners is not None:
        mask = ~exclude_flatliners
        S_hat_eval, S_true_eval = S_hat[mask], S_true[mask]
        eta_hat_eval, eta_true_eval = eta_hat[mask], eta_true[mask]
        if z_hat is not None and z_true is not None:
            z_hat_eval, z_true_eval = z_hat[mask], z_true[mask]
        else:
            z_hat_eval, z_true_eval = None, None
    else:
        S_hat_eval, S_true_eval = S_hat, S_true
        eta_hat_eval, eta_true_eval = eta_hat, eta_true
        z_hat_eval, z_true_eval = z_hat, z_true

    sal_cos = float(np.mean(cosine_similarity_rows(S_hat_eval, S_true_eval)))
    eta_pearson, eta_pearson_valid = safe_corr(eta_hat_eval, eta_true_eval)
    eta_spearman, eta_spearman_valid = safe_rank_corr(eta_hat_eval, eta_true_eval)

    result = {
        "salience_cosine": sal_cos,
        "eta_pearson": eta_pearson,
        "eta_pearson_valid": eta_pearson_valid,
        "eta_spearman": eta_spearman,
        "eta_spearman_valid": eta_spearman_valid,
    }

    if z_hat_eval is not None and z_true_eval is not None:
        z_corr, z_valid = safe_corr(z_hat_eval.ravel(), z_true_eval.ravel())
        result["z_corr"] = z_corr
        result["z_corr_valid"] = z_valid

    return result


def format_metric(name: str, value: float, valid: bool = True) -> str:
    """Format a metric for display."""
    if not valid or np.isnan(value):
        return f"  {name:>24s}: N/A"
    return f"  {name:>24s}: {value:.4f}"


# ===========================================================
# Demo
# ===========================================================

def demo_run(args: argparse.Namespace) -> None:
    """Simple demo: generate data, fit CSR, report recovery."""
    print("\n" + "=" * 60)
    print("CSR Demo Run")
    print("=" * 60)

    # Generate synthetic data
    data = generate_csr_data(
        N=args.N,
        J=args.J,
        alpha=args.alpha,
        signal_scale=args.signal_scale,
        noise_sd=args.noise_sd,
        likert=args.likert if args.likert > 0 else None,
        seed=args.seed,
    )

    snr = data["snr_info"]
    print(f"\nData: N={args.N}, J={args.J}, alpha={args.alpha}")
    print(f"SNR: mean|signal|={snr['mean_signal']:.3f}, noise_sd={snr['noise_sd']:.3f}, SNR={snr['snr']:.2f}")

    if args.likert > 0:
        print(f"Likert: K={args.likert}")

    # Fit CSR
    res = fit_csr(
        data["y"],
        ordinal_transform="auto",
        sigma2=args.noise_sd ** 2,
        gamma=args.gamma,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=True,
        seed=args.seed,
    )

    # Evaluate recovery
    metrics = eval_recovery(
        res.S, res.eta,
        data["S_true"], data["eta_true"],
        z_hat=res.z, z_true=data["z"],
        exclude_flatliners=res.flatliners
    )

    print(f"\n--- Recovery Metrics ---")
    print(format_metric("salience_cosine", metrics["salience_cosine"]))
    print(format_metric("eta_corr (Pearson)", metrics["eta_pearson"], metrics["eta_pearson_valid"]))
    print(format_metric("eta_corr (Spearman)", metrics["eta_spearman"], metrics["eta_spearman_valid"]))
    print(format_metric("reconstruction_mse", res.reconstruction_mse))
    print(format_metric("reconstruction_mae", res.reconstruction_mae))
    print(format_metric("sparsity", res.sparsity))
    print(format_metric("mean_entropy", float(np.mean(res.entropy))))

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


# ===========================================================
# CLI
# ===========================================================

def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(
        description="CSR: Continuous Salience-Response Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data generation
    p.add_argument("--N", type=int, default=500, help="Number of individuals")
    p.add_argument("--J", type=int, default=5, help="Number of items")
    p.add_argument("--alpha", type=float, default=0.35, help="Dirichlet concentration")
    p.add_argument("--signal_scale", type=float, default=3.0, help="Mean eta scale")
    p.add_argument("--noise_sd", type=float, default=0.15, help="Noise standard deviation")
    p.add_argument("--likert", type=int, default=5, help="Likert categories (0=continuous)")

    # CSR hyperparameters
    p.add_argument("--gamma", type=float, default=0.0, help="Reference regularization")

    # Optimization
    p.add_argument("--max_iter", type=int, default=200, help="Max iterations")
    p.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_known_args()


def main() -> None:
    args, _ = parse_args()
    #demo_run(args)

if __name__ == "__main__":
    main()
