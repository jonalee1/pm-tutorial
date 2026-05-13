"""
csr_empirical_study.py
======================
CSR Empirical Study: Extraversion Measurement Heterogeneity

Complete CSR analysis protocol:
  1. Fit 2-component model (y = η·s)
  2. Archetype analysis (K-means on salience vectors)
  3. Regression analysis
  4. Export results

Requires: csr_main.py in same directory
Author: Jonathan Lee
"""

import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from csr_main import (
    fit_csr, CSRResult, set_seed,
    compute_entropy, safe_corr, safe_rank_corr
)


# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
set_seed(RANDOM_SEED)

EXTRAVERSION_ITEMS = [f'E{i}' for i in range(1, 11)]
REVERSE_ITEMS = ['E2', 'E4', 'E6', 'E8', 'E10']  # For scoring only

# Item content descriptions (for interpretation, not analysis)
ITEM_DESCRIPTIONS = {
    'E1' : 'Life of the party',
    'E2' : 'Talk a lot',
    'E3' : 'Comfortable around people',
    'E4' : 'Keep in background',
    'E5' : 'Start conversations',
    'E6' : 'Have a lot to say',
    'E7' : 'Talk to many at parties',
    'E8' : 'Draw attention',
    'E9' : 'Center of attention',
    'E10': 'Quiet around strangers'
}


# ============================================================================
# Data Preparation
# ============================================================================

def download_big5_data(data_dir: str = ".") -> str:
    """Download OpenPsychometrics Big Five dataset if not present."""
    csv_path = os.path.join(data_dir, "BIG5_data.csv")

    if os.path.exists(csv_path):
        print(f"Dataset already exists: {csv_path}")
        return csv_path

    print("Attempting to download OpenPsychometrics Big Five Dataset...")
    url = "http://openpsychometrics.org/_rawdata/BIG5.zip"
    zip_path = os.path.join(data_dir, "BIG5.zip")

    try:
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        extracted_path = os.path.join(data_dir, "BIG5", "data.csv")
        if os.path.exists(extracted_path):
            os.rename(extracted_path, csv_path)

        if os.path.exists(zip_path):
            os.remove(zip_path)

        print(f"Dataset downloaded to: {csv_path}")
        return csv_path

    except Exception as e:
        raise RuntimeError(
            f"Download failed: {e}\n"
            "Please download manually from: http://openpsychometrics.org/_rawdata/BIG5.zip\n"
            "Extract and place data.csv as BIG5_data.csv in the working directory."
        )


def load_and_prepare_data(
    csv_path: str,
    country_filter: Optional[str] = "US",
    min_age: int = 18,
    max_age: int = 80
) -> pd.DataFrame:
    """Load and prepare Big Five data for CSR analysis."""

    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    data = pd.read_csv(csv_path, delimiter='\t')
    print(f"Original dataset: N = {len(data):,}")

    if country_filter:
        data = data[data['country'] == country_filter].copy()
        print(f"After {country_filter} filter: N = {len(data):,}")

    data = data[(data['age'] >= min_age) & (data['age'] <= max_age)].copy()
    print(f"After age filter ({min_age}-{max_age}): N = {len(data):,}")

    data = data[data['gender'].isin([1, 2])].copy()
    data['gender_label'] = data['gender'].map({1: 'Male', 2: 'Female'})
    print(f"After gender filter: N = {len(data):,}")

    e_valid = True
    for item in EXTRAVERSION_ITEMS:
        e_valid &= (data[item] >= 1) & (data[item] <= 5)
    data = data[e_valid].copy()
    print(f"After response validation: N = {len(data):,}")

    print("\nReverse scoring items:", REVERSE_ITEMS)
    for item in REVERSE_ITEMS:
        data[item] = 6 - data[item]

    data['age_group'] = pd.cut(
        data['age'],
        bins=[0, 30, 50, 100],
        labels=['18-30', '31-50', '51+']
    )

    data['sum_score'] = data[EXTRAVERSION_ITEMS].sum(axis=1)
    data['mean_score'] = data[EXTRAVERSION_ITEMS].mean(axis=1)

    responses = data[EXTRAVERSION_ITEMS].values
    person_var = np.var(responses, axis=1)
    flatliners = person_var < 0.1
    n_flat = np.sum(flatliners)
    print(f"\nFlat-liners (var < 0.1): {n_flat} ({100 * n_flat / len(data):.1f}%)")

    data = data[~flatliners].copy()
    print(f"Final analytic sample: N = {len(data):,}")
    print(f"  Age: M = {data['age'].mean():.1f}, SD = {data['age'].std():.1f}")
    print(f"  Gender: {data['gender_label'].value_counts().to_dict()}")
    print(f"  Extraversion sum: M = {data['sum_score'].mean():.1f}, SD = {data['sum_score'].std():.1f}")

    return data


# ============================================================================
# CSR Model Results Container
# ============================================================================

@dataclass
class CSRModelResults:
    """Container for CSR analysis results."""
    model_name: str

    # Core outputs
    volume: np.ndarray
    salience: np.ndarray

    # Configuration
    gamma: float
    ordinal_transform: str

    # Derived indices
    entropy: np.ndarray
    concentration: np.ndarray

    # Validation metrics
    volume_sum_corr: float
    diversity_index: float
    reconstruction_mse: float
    sparsity: float

    # Sample info
    n_participants: int
    n_items: int


# ============================================================================
# Core Analysis Functions
# ============================================================================

def compute_diversity_index(salience: np.ndarray) -> float:
    """Compute salience diversity index."""
    J = salience.shape[1]
    observed_var = np.sum(np.var(salience, axis=0))
    max_var = J * (1.0 / J) * (1 - 1.0 / J)
    return observed_var / max_var


def compute_concentration(salience: np.ndarray) -> np.ndarray:
    """Compute concentration (max salience) per person."""
    return np.max(salience, axis=1)


def fit_csr_model(
    responses: np.ndarray,
    sum_scores: np.ndarray,
    model_name: str = "CSR Model",
    gamma: float = 0.0,
    ordinal_transform: str = "expected",
    ref_mode: str = "minmax",
    verbose: bool = True
) -> CSRModelResults:
    """Fit CSR model and return comprehensive results."""
    N, J = responses.shape

    if verbose:
        print(f"\nFitting: {model_name}")
        print(f"  Model: y = η·s")
        print(f"  γ = {gamma}, transform = {ordinal_transform}")

    result = fit_csr(
        responses,
        ordinal_transform=ordinal_transform,
        gamma=gamma,
        ref_mode=ref_mode,
        max_iter=200,
        tol=1e-6,
        verbose=False,
        seed=RANDOM_SEED
    )

    volume = result.eta
    salience = result.S

    entropy_raw = compute_entropy(salience)
    max_entropy = np.log(J)
    entropy_normalized = entropy_raw / max_entropy

    concentration = compute_concentration(salience)

    vol_sum_corr, _ = safe_corr(volume, sum_scores)
    di = compute_diversity_index(salience)

    if verbose:
        print(f"  MSE = {result.reconstruction_mse:.4f}, DI = {di:.3f}, η-sum r = {vol_sum_corr:.4f}")

    return CSRModelResults(
        model_name=model_name,
        volume=volume,
        salience=salience,
        gamma=gamma,
        ordinal_transform=ordinal_transform,
        entropy=entropy_normalized,
        concentration=concentration,
        volume_sum_corr=vol_sum_corr,
        diversity_index=di,
        reconstruction_mse=result.reconstruction_mse,
        sparsity=result.sparsity,
        n_participants=N,
        n_items=J
    )


# ============================================================================
# Archetype Analysis with CLR transformation
# ============================================================================

def clr_transform(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Centered Log-Ratio (CLR) transformation for compositional data."""
    S_safe = S + eps
    log_S = np.log(S_safe)
    geometric_mean = log_S.mean(axis=1, keepdims=True)
    return log_S - geometric_mean


def inverse_clr(S_clr: np.ndarray) -> np.ndarray:
    """Inverse CLR transformation: map back to simplex."""
    S_exp = np.exp(S_clr)
    return S_exp / S_exp.sum(axis=1, keepdims=True)


def extract_archetypes(
    salience: np.ndarray,
    n_archetypes: int = 3,
    method: str = 'clr_kmeans'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Extract salience archetypes via clustering in CLR space."""
    from sklearn.cluster import KMeans

    if method == 'clr_kmeans':
        S_transformed = clr_transform(salience)
    else:
        S_transformed = salience

    km = KMeans(n_clusters=n_archetypes, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(S_transformed)

    centroids = np.zeros((n_archetypes, salience.shape[1]))
    for k in range(n_archetypes):
        mask = labels == k
        if np.sum(mask) > 0:
            centroids[k] = salience[mask].mean(axis=0)

    cluster_sizes = np.bincount(labels, minlength=n_archetypes)

    within_ss = 0
    for k in range(n_archetypes):
        mask = labels == k
        if np.sum(mask) > 0:
            within_ss += np.sum((salience[mask] - centroids[k]) ** 2)

    total_ss = np.sum((salience - salience.mean(axis=0)) ** 2)
    between_ss = total_ss - within_ss
    variance_explained = between_ss / total_ss if total_ss > 0 else 0

    centroids_clr = clr_transform(centroids)
    centroid_distances = []
    for i in range(n_archetypes):
        for j in range(i + 1, n_archetypes):
            dist = np.sqrt(np.sum((centroids_clr[i] - centroids_clr[j]) ** 2))
            centroid_distances.append(dist)

    stats = {
        'n_archetypes': n_archetypes,
        'cluster_sizes': cluster_sizes,
        'variance_explained': variance_explained,
        'mean_centroid_distance': np.mean(centroid_distances) if centroid_distances else 0,
        'inertia': km.inertia_,
        'method': method
    }

    return labels, centroids, stats


def characterize_archetype(centroid: np.ndarray) -> str:
    """
    Characterize archetype based on which items have highest salience.
    Uses content-based interpretation aligned with manuscript Section 5.
    
    Archetypes (from Table 7):
      - Social Initiators: E5, E3, E7 (social contact initiation)
      - Attention-Comfortable: E9, E8, E3 (comfort with visibility)
      - Verbal Expressors: E6, E2, E5 (verbal fluency)
    
    E3 appears in multiple archetypes, so we prioritize distinctive items.
    """
    sorted_idx = np.argsort(centroid)[::-1]
    top_items = [EXTRAVERSION_ITEMS[idx] for idx in sorted_idx[:3]]
    top_set = set(top_items[:3])

    # Distinctive items for each archetype (E3 is shared, so excluded)
    attention_distinctive = {'E8', 'E9'}  # Only in Attention-Comfortable
    verbal_distinctive = {'E2', 'E6'}      # Only in Verbal Expressors
    social_distinctive = {'E5', 'E7'}      # Primarily Social Initiators (E5 shared with Verbal)

    # Check for distinctive attention items first (E8, E9 are unique)
    if top_set & attention_distinctive:
        return "Attention-Comfortable"
    # Check for verbal items (E2, E6)
    elif top_set & verbal_distinctive:
        return "Verbal Expressors"
    # Check for social initiative items (E5, E7)
    elif top_set & social_distinctive:
        return "Social Initiators"
    else:
        return "Mixed Profile"


def analyze_archetypes(model: CSRModelResults, n_archetypes: int = 3) -> Dict:
    """Full archetype analysis for extraversion model."""

    labels, centroids, stats = extract_archetypes(
        model.salience, n_archetypes=n_archetypes
    )

    archetype_details = []

    for k in range(n_archetypes):
        mask = labels == k
        n_k = np.sum(mask)

        mean_volume = np.mean(model.volume[mask])
        mean_entropy = np.mean(model.entropy[mask])

        centroid = centroids[k]

        sorted_idx = np.argsort(centroid)[::-1]
        top_items = [(EXTRAVERSION_ITEMS[idx], centroid[idx]) for idx in sorted_idx[:3]]

        character = characterize_archetype(centroid)

        archetype_details.append({
            'archetype': k + 1,
            'n': n_k,
            'pct': 100 * n_k / len(labels),
            'centroid': centroid,
            'top_items': top_items,
            'character': character,
            'mean_volume': mean_volume,
            'mean_entropy': mean_entropy
        })

    archetype_details = sorted(archetype_details, key=lambda x: -x['n'])
    for i, detail in enumerate(archetype_details):
        detail['archetype'] = i + 1

    return {
        'labels': labels,
        'centroids': centroids,
        'stats': stats,
        'details': archetype_details
    }


# ============================================================================
# Regression Analysis
# ============================================================================

def regression_analysis(data: pd.DataFrame, model: CSRModelResults) -> Dict:
    """Regression analysis predicting salience characteristics."""

    trait_level = data['sum_score'].values
    trait_centered = trait_level - np.mean(trait_level)

    age = data['age'].values
    age_centered = age - np.mean(age)

    gender_female = (data['gender'] == 2).astype(float)

    X = np.column_stack([
        np.ones(len(data)),
        trait_centered,
        age_centered,
        gender_female
    ])

    predictor_names = ['Intercept', 'Trait Level', 'Age', 'Female']
    results = {}

    # Model: Predicting Entropy from demographics and trait level
    y_entropy = model.entropy
    beta_entropy = np.linalg.lstsq(X, y_entropy, rcond=None)[0]
    y_hat_entropy = X @ beta_entropy
    ss_res_entropy = np.sum((y_entropy - y_hat_entropy) ** 2)
    ss_tot_entropy = np.sum((y_entropy - np.mean(y_entropy)) ** 2)
    r2_entropy = 1 - ss_res_entropy / ss_tot_entropy

    n, p = len(y_entropy), X.shape[1]
    mse_entropy = ss_res_entropy / (n - p)
    var_beta_entropy = mse_entropy * np.linalg.inv(X.T @ X).diagonal()
    se_entropy = np.sqrt(var_beta_entropy)
    t_entropy = beta_entropy / se_entropy

    results['entropy'] = {
        'beta': beta_entropy, 'se': se_entropy, 't': t_entropy,
        'r2': r2_entropy, 'predictor_names': predictor_names
    }

    # Volume-Entropy correlation (key finding)
    vol_entropy_corr, _ = safe_corr(model.volume, model.entropy)
    results['vol_entropy_corr'] = vol_entropy_corr

    return results


# ============================================================================
# Output Functions
# ============================================================================

def print_model_summary(model: CSRModelResults) -> None:
    """Print summary for a single model."""

    print(f"\n{'=' * 60}")
    print(f"MODEL: {model.model_name}")
    print(f"{'=' * 60}")

    print(f"\nModel specification: y = η·s")
    print(f"Sample: N = {model.n_participants:,}, J = {model.n_items}")
    print(f"Configuration: γ = {model.gamma}, transform = {model.ordinal_transform}")

    print(f"\nFit Statistics:")
    print(f"  Reconstruction MSE: {model.reconstruction_mse:.4f}")
    print(f"  Sparsity: {model.sparsity:.1%}")
    print(f"  Volume-Sum Score r: {model.volume_sum_corr:.4f}")

    print(f"\nHeterogeneity:")
    print(f"  Diversity Index: {model.diversity_index:.3f}")
    print(f"  Mean Entropy: {np.mean(model.entropy):.3f} (SD = {np.std(model.entropy):.3f})")
    print(f"  Mean Concentration: {np.mean(model.concentration):.3f} (SD = {np.std(model.concentration):.3f})")

    print(f"\nMean Salience per Item:")
    mean_sal = np.mean(model.salience, axis=0)
    for j, item in enumerate(EXTRAVERSION_ITEMS):
        desc = ITEM_DESCRIPTIONS[item]
        print(f"  {item}: {mean_sal[j]:.4f}  # {desc}")


def print_archetype_analysis(archetype_results: Dict) -> None:
    """Print archetype analysis results."""

    print(f"\n{'=' * 70}")
    print("ARCHETYPE ANALYSIS: CLR-Transformed K-Means Clustering")
    print(f"{'=' * 70}")

    stats = archetype_results['stats']
    print(f"\nClustering Statistics:")
    print(f"  Method: {stats.get('method', 'clr_kmeans')} (Aitchison-appropriate)")
    print(f"  Number of archetypes: {stats['n_archetypes']}")
    print(f"  Variance explained: {stats['variance_explained'] * 100:.1f}%")
    print(f"  Mean centroid distance (Aitchison): {stats['mean_centroid_distance']:.4f}")

    print(f"\nArchetype Details:")
    for detail in archetype_results['details']:
        print(f"\n  Archetype {detail['archetype']}: {detail['character']} (n={detail['n']}, {detail['pct']:.1f}%)")
        print(f"    Top 3 items: ", end="")
        top_str = ", ".join([f"{item}={sal:.3f}" for item, sal in detail['top_items']])
        print(top_str)
        print(f"    Mean volume: {detail['mean_volume']:.2f}")
        print(f"    Mean entropy: {detail['mean_entropy']:.3f}")

        centroid_str = ", ".join([f"{s:.3f}" for s in detail['centroid']])
        print(f"    Centroid: [{centroid_str}]")


def print_regression_results(reg_results: Dict) -> None:
    """Print regression analysis results."""

    print(f"\n{'=' * 70}")
    print("REGRESSION: Predicting Entropy from Trait Level")
    print(f"{'=' * 70}")

    print(f"\nVolume-Entropy Correlation: r = {reg_results['vol_entropy_corr']:.4f}")

    print(f"\n--- Predicting Entropy ---")
    print(f"{'Predictor':<15} {'β':>10} {'SE':>10} {'t':>10}")
    print(f"{'-' * 45}")

    for i, name in enumerate(reg_results['entropy']['predictor_names']):
        b = reg_results['entropy']['beta'][i]
        se = reg_results['entropy']['se'][i]
        t = reg_results['entropy']['t'][i]
        print(f"{name:<15} {b:>10.4f} {se:>10.4f} {t:>10.2f}")

    print(f"{'-' * 45}")
    print(f"R² = {reg_results['entropy']['r2']:.3f}")


def print_publication_summary(
    model: CSRModelResults,
    reg_results: Dict,
    archetype_results: Dict,
    data: pd.DataFrame
) -> None:
    """Print publication-ready summary."""

    print(f"\n{'=' * 70}")
    print("PUBLICATION SUMMARY")
    print(f"{'=' * 70}")

    n = len(data)

    print(f"""
SAMPLE
------
N = {n:,} US adults (ages 18-80)
Age: M = {data['age'].mean():.1f}, SD = {data['age'].std():.1f}
Gender: {(data['gender'] == 2).sum():,} female ({100 * (data['gender'] == 2).mean():.1f}%)
Extraversion (10-item sum): M = {data['sum_score'].mean():.2f}, SD = {data['sum_score'].std():.2f}

CSR ANALYSIS
------------
Model: y = η·s (2-component)
Transform: {model.ordinal_transform}
γ = {model.gamma}

KEY FINDINGS
------------
1. HETEROGENEITY
   Diversity Index = {model.diversity_index:.3f}
   Mean Entropy = {np.mean(model.entropy):.3f} (SD = {np.std(model.entropy):.3f})
   → {'Substantial' if model.diversity_index > 0.1 else 'Modest'} individual differences in extraversion expression

2. ARCHETYPES ({archetype_results['stats']['n_archetypes']} clusters, {archetype_results['stats']['variance_explained'] * 100:.1f}% variance explained)""")

    for detail in archetype_results['details']:
        print(f"   Archetype {detail['archetype']}: {detail['character']} ({detail['pct']:.1f}%) - "
              f"Top: {detail['top_items'][0][0]}={detail['top_items'][0][1]:.3f}")

    print(f"""
3. VOLUME-ENTROPY RELATIONSHIP
   Volume-Entropy r = {reg_results['vol_entropy_corr']:.3f}
   Entropy R² = {reg_results['entropy']['r2']:.3f}
   → Higher trait levels associated with more uniform salience
""")


# ============================================================================
# Export Functions
# ============================================================================

def export_results(
    data: pd.DataFrame,
    model: CSRModelResults,
    archetype_results: Dict,
    output_prefix: str = "csr_extraversion"
) -> None:
    """Export results to CSV files."""

    print(f"\n{'=' * 60}")
    print("EXPORTING RESULTS")
    print(f"{'=' * 60}")

    results_df = data[['age', 'gender', 'gender_label', 'age_group', 'sum_score']].copy()

    results_df['volume'] = model.volume
    results_df['entropy'] = model.entropy
    results_df['concentration'] = model.concentration

    for j, item in enumerate(EXTRAVERSION_ITEMS):
        results_df[f'sal_{item}'] = model.salience[:, j]

    results_df['archetype'] = archetype_results['labels'] + 1

    results_path = f"{output_prefix}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  Results saved to: {results_path}")

    centroid_df = pd.DataFrame(
        archetype_results['centroids'],
        columns=EXTRAVERSION_ITEMS
    )
    centroid_df.insert(0, 'archetype', range(1, len(archetype_results['centroids']) + 1))
    centroid_path = f"{output_prefix}_archetypes.csv"
    centroid_df.to_csv(centroid_path, index=False)
    print(f"  Archetype centroids saved to: {centroid_path}")


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def run_csr_analysis(
    country_filter: str = "US",
    ordinal_transform: str = "expected",
    gamma: float = 0.0,
    n_archetypes: int = 3,
    export: bool = True
) -> Dict:
    """
    Run complete CSR analysis.

    Protocol:
    1. Fit CSR model (y = η·s)
    2. Archetype analysis (K-means clustering)
    3. Regression analysis
    4. Export results
    """

    print("\n" + "=" * 70)
    print("CSR EMPIRICAL STUDY: EXTRAVERSION MEASUREMENT HETEROGENEITY")
    print("=" * 70)
    print("\nAnalysis Protocol:")
    print("  Step 1: Fit CSR model (y = η·s)")
    print("  Step 2: Archetype analysis (K-means on 10-dim salience)")
    print("  Step 3: Regression analysis")

    # Data Preparation
    csv_path = download_big5_data()
    data = load_and_prepare_data(csv_path, country_filter=country_filter)

    responses = data[EXTRAVERSION_ITEMS].values
    sum_scores = data['sum_score'].values

    # Step 1: Fit CSR Model
    print(f"\n{'=' * 60}")
    print("STEP 1: CSR MODEL (y = η·s)")
    print(f"{'=' * 60}")

    model = fit_csr_model(
        responses, sum_scores,
        model_name="CSR Model",
        gamma=gamma,
        ordinal_transform=ordinal_transform
    )
    print_model_summary(model)

    # Step 2: Archetype Analysis
    print(f"\n{'=' * 60}")
    print("STEP 2: ARCHETYPE ANALYSIS")
    print(f"{'=' * 60}")

    archetype_results = analyze_archetypes(model, n_archetypes=n_archetypes)
    print_archetype_analysis(archetype_results)

    # Step 3: Regression Analysis
    print(f"\n{'=' * 60}")
    print("STEP 3: REGRESSION ANALYSIS")
    print(f"{'=' * 60}")

    reg_results = regression_analysis(data, model)
    print_regression_results(reg_results)

    # Publication Summary
    print_publication_summary(model, reg_results, archetype_results, data)

    # Export
    if export:
        export_results(data, model, archetype_results)

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")

    return {
        'data': data,
        'model': model,
        'archetype_results': archetype_results,
        'reg_results': reg_results
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CSR Empirical Study: Complete Analysis Protocol"
    )
    parser.add_argument(
        "--country", type=str, default="US",
        help="Country filter (default: US, use 'all' for no filter)"
    )
    parser.add_argument(
        "--transform", type=str, default="expected",
        choices=["none", "midpoint", "expected", "rank"],
        help="Ordinal transformation method (default: expected)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.0,
        help="Reference salience regularization (default: 0.0)"
    )
    parser.add_argument(
        "--archetypes", type=int, default=3,
        help="Number of archetypes (default: 3)"
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip exporting results to CSV"
    )

    args, _ = parser.parse_known_args()

    country = None if args.country.lower() == 'all' else args.country

    results = run_csr_analysis(
        country_filter=country,
        ordinal_transform=args.transform,
        gamma=args.gamma,
        n_archetypes=args.archetypes,
        export=not args.no_export
    )
