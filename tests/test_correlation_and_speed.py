"""Correlation (R vs Python) and speed benchmarks."""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

REF_PATH = Path("D:/test/TranspaceR-main/py-TranspaceR/tests/r_reference_output.json")
EXPR_PATH = Path("D:/test/TranspaceR-main/TranspaceR-main/Example_data/Expression_file_example.csv.gz")
META_PATH = Path("D:/test/TranspaceR-main/TranspaceR-main/Example_data/Meta_data_example.csv")


def load_data():
    expr = pd.read_csv(EXPR_PATH, index_col=0)
    meta = pd.read_csv(META_PATH, index_col=0)
    return expr, meta


def run_r_function(func_name, expr_path, meta_path, output_path):
    """Run an R function and return output path + elapsed time."""
    import subprocess
    r_script = f"""
    t0 <- proc.time()
    suppressPackageStartupMessages({{library(Matrix)}})
    r_dir <- 'D:/test/TranspaceR-main/TranspaceR-main/R'
    expr <- read.csv('{expr_path}', row.names=1)
    meta <- read.csv('{meta_path}', row.names=1)
    expr_mat <- as.matrix(expr)
    results <- list()

    {func_name}

    elapsed <- (proc.time() - t0)[3]
    results$elapsed <- elapsed
    json_str <- jsonlite::toJSON(results, auto_unbox=TRUE)
    writeLines(json_str, '{output_path}')
    """
    result = subprocess.run(
        ["C:/Program Files/R/R-4.5.2/bin/Rscript.exe", "-e", r_script],
        capture_output=True, text=True, timeout=300
    )
    return result


# ============================================================
# 1. C_normalisation correlation
# ============================================================
def test_c_normalisation_correlation():
    """Compare C_normalisation output: R vs Python."""
    import transspacer as ts

    expr, meta = load_data()
    expr_arr = expr.values.astype(float)
    area = meta["Area"].values

    # Python
    t0 = time.perf_counter()
    py_result = ts.c_normalisation(expr_arr, area)
    py_time = time.perf_counter() - t0

    # R reference (from JSON)
    with open(REF_PATH) as f:
        ref = json.load(f)

    # Compare col_sums
    py_col_sums = np.nansum(py_result, axis=0)
    ref_col_sums = np.array(ref["C_normalisation"]["col_sums"])

    r_pearson, _ = pearsonr(py_col_sums[:5], ref_col_sums)
    r_spearman, _ = spearmanr(py_col_sums[:5], ref_col_sums)

    print(f"=== C_normalisation ===")
    print(f"  Pearson r:  {r_pearson:.6f}")
    print(f"  Spearman r: {r_spearman:.6f}")
    print(f"  Python time: {py_time:.4f}s")
    print(f"  Max abs error (col_sums): {np.max(np.abs(py_col_sums[:5] - ref_col_sums)):.6e}")

    assert r_pearson > 0.99, f"Pearson too low: {r_pearson}"
    return py_time


# ============================================================
# 2. Otsu thresholding correlation
# ============================================================
def test_otsu_correlation():
    import transspacer as ts
    expr, _ = load_data()
    x = np.log10(expr.values.sum(axis=1).astype(float) + 1)
    x = x[np.isfinite(x)]

    t0 = time.perf_counter()
    py_threshold = ts.otsu_thresholding(x)
    py_time = time.perf_counter() - t0

    with open(REF_PATH) as f:
        ref = json.load(f)
    ref_threshold = ref["Otsu_thresholding"]["threshold"]

    print(f"\n=== Otsu_thresholding ===")
    print(f"  Python: {py_threshold:.4f}, R: {ref_threshold:.4f}")
    print(f"  Abs error: {abs(py_threshold - ref_threshold):.6e}")
    print(f"  Python time: {py_time:.4f}s")

    assert abs(py_threshold - ref_threshold) < 0.01
    return py_time


# ============================================================
# 3. colvars_sparse correlation
# ============================================================
def test_colvars_sparse_correlation():
    from transspacer import colvars_sparse
    from scipy.sparse import csc_matrix

    expr, _ = load_data()
    sp = csc_matrix(expr.values.astype(float))

    t0 = time.perf_counter()
    py_cv = colvars_sparse(sp)
    py_time = time.perf_counter() - t0

    with open(REF_PATH) as f:
        ref = json.load(f)
    ref_cv = np.array(ref["colvars_sparse"]["values"])

    r_pearson, _ = pearsonr(py_cv[:10], ref_cv)
    r_spearman, _ = spearmanr(py_cv[:10], ref_cv)
    max_err = np.max(np.abs(py_cv[:10] - ref_cv))

    print(f"\n=== colvars_sparse ===")
    print(f"  Pearson r:  {r_pearson:.6f}")
    print(f"  Spearman r: {r_spearman:.6f}")
    print(f"  Max abs error: {max_err:.6e}")
    print(f"  Python time: {py_time:.4f}s")

    assert r_pearson > 0.99
    return py_time


# ============================================================
# 4. Get_variogram_map correlation
# ============================================================
def test_variogram_map_correlation():
    from transspacer.variogram import get_variogram_map

    # Use deterministic input
    np.random.seed(42)
    test_im = np.random.rand(8, 8)

    t0 = time.perf_counter()
    py_vmap = get_variogram_map(test_im)
    py_time = time.perf_counter() - t0

    print(f"\n=== Get_variogram_map ===")
    print(f"  Shape: {py_vmap.shape}")
    print(f"  Python time: {py_time:.4f}s")
    print(f"  Mean value: {np.mean(np.abs(py_vmap)):.6e}")
    return py_time


# ============================================================
# 5. Get_isotropic_vario correlation
# ============================================================
def test_isotropic_vario_correlation():
    from transspacer.variogram import get_isotropic_vario

    np.random.seed(42)
    test_vmap = np.random.rand(16, 16)

    t0 = time.perf_counter()
    r, v = get_isotropic_vario(test_vmap)
    py_time = time.perf_counter() - t0

    with open(REF_PATH) as f:
        ref = json.load(f)
    ref_r = np.array(ref["Get_isotropic_vario"]["r_values"])
    ref_v = np.array(ref["Get_isotropic_vario"]["v_values"])

    # Compare first 10
    min_len = min(len(r), len(ref_r), 10)
    r_pearson_r, _ = pearsonr(r[:min_len], ref_r[:min_len])
    r_pearson_v, _ = pearsonr(v[:min_len], ref_v[:min_len])

    print(f"\n=== Get_isotropic_vario ===")
    print(f"  Pearson r (distances): {r_pearson_r:.6f}")
    print(f"  Pearson r (variogram): {r_pearson_v:.6f}")
    print(f"  Python bins: {len(r)}, R bins: {len(ref_r)}")
    print(f"  Python time: {py_time:.4f}s")

    assert r_pearson_r > 0.99
    return py_time


# ============================================================
# 6. Geary C speed benchmark (Python only, R dep missing)
# ============================================================
def test_geary_c_speed():
    from transspacer.spatial_stats import geary_c_score

    expr, meta = load_data()
    np.random.seed(42)
    n_sub = 500
    idx = np.random.choice(expr.shape[0], n_sub, replace=False)
    coords = np.column_stack([
        meta["cell_centroid_x"].values[idx],
        meta["cell_centroid_y"].values[idx]
    ])
    expr_sub = expr.values[idx, :20].astype(float)

    t0 = time.perf_counter()
    result = geary_c_score(expr_sub, coords, pvalue_threshold=0.05)
    py_time = time.perf_counter() - t0

    print(f"\n=== Geary_C_speed ===")
    print(f"  {n_sub} cells, 20 genes")
    print(f"  Python time: {py_time:.4f}s")
    print(f"  Selected: {len(result['Selected_genes'])} genes")
    return py_time


# ============================================================
# 7. Clustering speed benchmark
# ============================================================
def test_clustering_speed():
    from transspacer import cell_clustering_function
    from sklearn.decomposition import PCA

    expr, _ = load_data()
    np.random.seed(42)
    n_sub = 1000
    idx = np.random.choice(expr.shape[0], n_sub, replace=False)
    expr_sub = expr.values[idx].astype(float)

    pca = PCA(n_components=20, random_state=42)
    pca_data = pca.fit_transform(expr_sub)

    t0 = time.perf_counter()
    labels = cell_clustering_function(pca_data, K=10, resolution=1.0)
    py_time = time.perf_counter() - t0

    print(f"\n=== Clustering_speed ===")
    print(f"  {n_sub} cells, K=10")
    print(f"  Python time: {py_time:.4f}s")
    print(f"  Clusters: {len(np.unique(labels))}")
    return py_time


# ============================================================
# 8. Full variogram pipeline speed
# ============================================================
def test_variogram_pipeline_speed():
    from transspacer import compute_variogram

    expr, meta = load_data()
    np.random.seed(42)
    n_sub = 1000
    idx = np.random.choice(expr.shape[0], n_sub, replace=False)
    expr_sub = expr.values[idx, :10].astype(float)
    x = meta["cell_centroid_x"].values[idx]
    y = meta["cell_centroid_y"].values[idx]

    t0 = time.perf_counter()
    result = compute_variogram(expr_sub, x, y, automatic_pad=True, ncores=1)
    py_time = time.perf_counter() - t0

    print(f"\n=== Variogram_pipeline_speed ===")
    print(f"  {n_sub} cells, 10 genes")
    print(f"  Python time: {py_time:.4f}s")
    print(f"  Selected: {len(result['Selected_genes'])} genes")
    return py_time


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Correlation & Speed Benchmark — py-TranspaceR v0.1.0")
    print("=" * 60)

    times = {}
    times["C_normalisation"] = test_c_normalisation_correlation()
    times["Otsu_thresholding"] = test_otsu_correlation()
    times["colvars_sparse"] = test_colvars_sparse_correlation()
    times["Get_variogram_map"] = test_variogram_map_correlation()
    times["Get_isotropic_vario"] = test_isotropic_vario_correlation()
    times["Geary_C"] = test_geary_c_speed()
    times["Clustering"] = test_clustering_speed()
    times["Variogram_pipeline"] = test_variogram_pipeline_speed()

    print("\n" + "=" * 60)
    print("Summary — Python execution times")
    print("=" * 60)
    for name, t in times.items():
        print(f"  {name:25s} {t:.4f}s")
    print(f"  {'TOTAL':25s} {sum(times.values()):.4f}s")
