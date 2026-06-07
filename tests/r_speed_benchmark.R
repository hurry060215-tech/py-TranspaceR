#!/usr/bin/env Rscript
# R speed benchmark for TranspaceR functions

suppressPackageStartupMessages({
  library(Matrix)
  library(stats)
})

r_dir <- "D:/test/TranspaceR-main/TranspaceR-main/R"
for (f in c("fftshift.R", "ifftshift.R", "pad_definitor.R", "C_normalisation.R",
            "Otsu_thresholding.R", "colvars_sparse.R", "Get_variogram_map.R",
            "Get_isotropic_vario.R", "Cell_clustering_function.R")) {
  fpath <- file.path(r_dir, f)
  if (file.exists(fpath)) tryCatch(source(fpath), error = function(e) {})
}

expr <- read.csv("D:/test/TranspaceR-main/TranspaceR-main/Example_data/Expression_file_example.csv.gz", row.names = 1)
meta <- read.csv("D:/test/TranspaceR-main/TranspaceR-main/Example_data/Meta_data_example.csv", row.names = 1)
expr_mat <- as.matrix(expr)

results <- list()

# 1. C_normalisation
t0 <- proc.time()
normed <- C_normalisation(expr_mat, meta$Area)
results$C_normalisation <- (proc.time() - t0)[3]
cat("C_normalisation:", results$C_normalisation, "s\n")

# 2. Otsu_thresholding
x <- log10(rowSums(expr_mat) + 1)
x <- x[is.finite(x)]
t0 <- proc.time()
threshold <- Otsu_thresholding(x)
results$Otsu_thresholding <- (proc.time() - t0)[3]
cat("Otsu_thresholding:", results$Otsu_thresholding, "s\n")

# 3. pad_definitor
t0 <- proc.time()
n_pad <- pad_definitor(meta)
results$pad_definitor <- (proc.time() - t0)[3]
cat("pad_definitor:", results$pad_definitor, "s\n")

# 4. colvars_sparse
sp <- Matrix(expr_mat, sparse = TRUE)
t0 <- proc.time()
cv <- colvars_sparse(sp)
results$colvars_sparse <- (proc.time() - t0)[3]
cat("colvars_sparse:", results$colvars_sparse, "s\n")

# 5. Get_variogram_map
set.seed(42)
test_im <- matrix(runif(64), 8, 8)
t0 <- proc.time()
vmap <- Get_variogram_map(test_im)
results$Get_variogram_map <- (proc.time() - t0)[3]
cat("Get_variogram_map:", results$Get_variogram_map, "s\n")

# 6. Get_isotropic_vario
test_vmap <- matrix(runif(256), 16, 16)
t0 <- proc.time()
iso <- Get_isotropic_vario(test_vmap)
results$Get_isotropic_vario <- (proc.time() - t0)[3]
cat("Get_isotropic_vario:", results$Get_isotropic_vario, "s\n")

# 7. Clustering (PCA + KNN + Louvain)
n_sub <- 1000
set.seed(42)
idx <- sample(nrow(expr_mat), n_sub)
expr_sub <- expr_mat[idx, ]
# Remove zero-variance columns
col_vars <- apply(expr_sub, 2, var)
expr_sub <- expr_sub[, col_vars > 0]
pca_data <- prcomp(expr_sub, center = TRUE, scale. = TRUE)$x[, 1:20]
tryCatch({
  t0 <- proc.time()
  clust <- Cell_clustering_function(pca_data, K = 10, metric_used = "L2", nThreads = 1, resolution = 1)
  results$Clustering <- (proc.time() - t0)[3]
  cat("Clustering:", results$Clustering, "s\n")
}, error = function(e) {
  cat("Clustering error:", e$message, "\n")
  results$Clustering <<- NA
})

# Write results
json_str <- jsonlite::toJSON(results, auto_unbox = TRUE, pretty = TRUE)
writeLines(json_str, "D:/test/TranspaceR-main/py-TranspaceR/tests/r_speed_results.json")
cat("\nResults written to r_speed_results.json\n")
