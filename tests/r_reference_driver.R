#!/usr/bin/env Rscript
# R reference driver for TranspaceR parity testing
# Usage: Rscript r_reference_driver.R <expression_csv> <meta_csv> <output_json>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript r_reference_driver.R <expression_csv> <meta_csv> <output_json>\n")
  quit(status = 1)
}

expr_path <- args[1]
meta_path <- args[2]
output_path <- args[3]

suppressPackageStartupMessages({
  library(Matrix)
  library(stats)
})

# Source individual R functions (skip those requiring unavailable packages)
r_dir <- "D:/test/TranspaceR-main/TranspaceR-main/R"

# Source base-R-only functions first
base_r_files <- c(
  "fftshift.R", "ifftshift.R", "pad_definitor.R",
  "C_normalisation.R", "Otsu_thresholding.R",
  "colvars_sparse.R", "aggregate_sparse.R",
  "calculate_log2fc.R", "color_convertion.R", "string.to.colors.R",
  "Compute_radius.R", "Curate_data.R", "Fraction_multiple_samples.R",
  "Select_genes.R"
)

for (f in base_r_files) {
  fpath <- file.path(r_dir, f)
  if (file.exists(fpath)) {
    tryCatch(source(fpath, local = TRUE), error = function(e) {
      cat("Warning: could not source", f, ":", e$message, "\n")
    })
  }
}

# Load data
cat("Loading data...\n")
expr_df <- read.csv(expr_path, row.names = 1)
meta_df <- read.csv(meta_path, row.names = 1)

cat("Expression:", nrow(expr_df), "cells x", ncol(expr_df), "genes\n")
cat("Meta:", nrow(meta_df), "rows x", ncol(meta_df), "cols\n")

results <- list()

# === C_normalisation ===
cat("Testing C_normalisation...\n")
tryCatch({
  area <- meta_df$Area
  expr_mat <- as.matrix(expr_df)
  normed <- C_normalisation(expr_mat, area)
  # Save a small sample for comparison
  results$C_normalisation <- list(
    shape = dim(normed),
    sample_values = normed[1:min(5, nrow(normed)), 1:min(5, ncol(normed))],
    col_sums = head(colSums(normed, na.rm = TRUE), 5)
  )
}, error = function(e) {
  cat("C_normalisation error:", e$message, "\n")
})

# === Otsu_thresholding ===
cat("Testing Otsu_thresholding...\n")
tryCatch({
  x_vals <- log10(rowSums(expr_mat) + 1)
  x_vals <- x_vals[is.finite(x_vals)]
  threshold <- Otsu_thresholding(x_vals)
  results$Otsu_thresholding <- list(threshold = threshold)
}, error = function(e) {
  cat("Otsu_thresholding error:", e$message, "\n")
})

# === pad_definitor ===
cat("Testing pad_definitor...\n")
tryCatch({
  n_pad <- pad_definitor(meta_df)
  results$pad_definitor <- list(n_pad = n_pad)
}, error = function(e) {
  cat("pad_definitor error:", e$message, "\n")
})

# === fftshift/ifftshift ===
cat("Testing fftshift/ifftshift...\n")
tryCatch({
  test_mat <- matrix(1:16, 4, 4)
  shifted <- fftshift(test_mat)
  unshifted <- ifftshift(shifted)
  results$fftshift <- list(
    input = test_mat,
    shifted = shifted,
    roundtrip = unshifted,
    roundtrip_ok = all(test_mat == unshifted)
  )
}, error = function(e) {
  cat("fftshift error:", e$message, "\n")
})

# === colvars_sparse ===
cat("Testing colvars_sparse...\n")
tryCatch({
  sp_mat <- Matrix(expr_mat, sparse = TRUE)
  cv <- colvars_sparse(sp_mat)
  results$colvars_sparse <- list(
    values = head(cv, 10),
    length = length(cv)
  )
}, error = function(e) {
  cat("colvars_sparse error:", e$message, "\n")
})

# === aggregate_sparse ===
cat("Testing aggregate_sparse...\n")
tryCatch({
  sp_mat <- Matrix(expr_mat, sparse = TRUE)
  groups <- sample(c("A", "B", "C"), nrow(sp_mat), replace = TRUE)
  agg <- aggregate_sparse(sp_mat, groups)
  results$aggregate_sparse <- list(
    shape = dim(agg),
    sample_values = head(agg, 3)
  )
}, error = function(e) {
  cat("aggregate_sparse error:", e$message, "\n")
})

# === calculate_log2fc ===
cat("Testing calculate_log2fc...\n")
tryCatch({
  clusters <- sample(c("A", "B", "C"), nrow(expr_mat), replace = TRUE)
  lfc <- calculate_log2fc(expr_mat, "ABCC11", clusters)
  results$calculate_log2fc <- list(values = lfc)
}, error = function(e) {
  cat("calculate_log2fc error:", e$message, "\n")
})

# === Get_variogram_map (needs gsignal, imager, OpenImageR) ===
cat("Testing Get_variogram_map...\n")
tryCatch({
  source(file.path(r_dir, "Get_variogram_map.R"))
  test_im <- matrix(runif(64), 8, 8)
  vmap <- Get_variogram_map(test_im)
  results$Get_variogram_map <- list(
    shape = dim(vmap),
    sample_values = vmap[1:3, 1:3]
  )
}, error = function(e) {
  cat("Get_variogram_map error:", e$message, "\n")
})

# === Get_isotropic_vario ===
cat("Testing Get_isotropic_vario...\n")
tryCatch({
  source(file.path(r_dir, "Get_isotropic_vario.R"))
  test_vmap <- matrix(runif(256), 16, 16)
  iso <- Get_isotropic_vario(test_vmap)
  results$Get_isotropic_vario <- list(
    n_rows = nrow(iso),
    r_values = head(iso$r, 10),
    v_values = head(iso$Variogram, 10)
  )
}, error = function(e) {
  cat("Get_isotropic_vario error:", e$message, "\n")
})

# === Geary_C_score (needs RCDT - may fail) ===
cat("Testing Geary_C_score...\n")
tryCatch({
  source(file.path(r_dir, "Geary_C_score.R"))
  # Use small subset
  n_sub <- min(200, nrow(expr_mat))
  idx <- sample(nrow(expr_mat), n_sub)
  coords <- cbind(meta_df$cell_centroid_x[idx], meta_df$cell_centroid_y[idx])
  expr_sub <- expr_mat[idx, 1:min(5, ncol(expr_mat)), drop = FALSE]
  gc_result <- Geary_C_score(expr_sub, data.frame(cell_centroid_x = coords[,1], cell_centroid_y = coords[,2]),
                              tempdir(), "Test", "Test", pvalue_threshold = 0.05)
  results$Geary_C_score <- list(
    values = head(gc_result$Values, 5),
    n_selected = length(gc_result$Selected_genes),
    p_values = head(gc_result$P_values, 5)
  )
}, error = function(e) {
  cat("Geary_C_score error:", e$message, "\n")
})

# === Cell_clustering_function (needs N2R, igraph) ===
cat("Testing Cell_clustering_function...\n")
tryCatch({
  source(file.path(r_dir, "Cell_clustering_function.R"))
  # Use PCA of small subset
  n_sub <- min(500, nrow(expr_mat))
  idx <- sample(nrow(expr_mat), n_sub)
  pca_data <- prcomp(expr_mat[idx, ], center = TRUE, scale. = TRUE)$x[, 1:20]
  clust <- Cell_clustering_function(pca_data, K = 10, metric_used = "L2", nThreads = 1, resolution = 1)
  results$Cell_clustering <- list(
    n_clusters = length(unique(clust)),
    cluster_sizes = table(clust)
  )
}, error = function(e) {
  cat("Cell_clustering_function error:", e$message, "\n")
})

# Write results
cat("Writing results to", output_path, "\n")
json_str <- jsonlite::toJSON(results, auto_unbox = TRUE, pretty = TRUE, null = "null")
writeLines(json_str, output_path)

cat("Done! Results written.\n")
