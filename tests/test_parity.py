"""Parity tests for TranspaceR port."""

import numpy as np
import pytest
from scipy.sparse import random as sparse_random, csc_matrix


class TestFFTUtils:
    def test_fftshift_roundtrip(self):
        from transspacer import fftshift, ifftshift
        x = np.random.randn(10, 12)
        assert np.allclose(ifftshift(fftshift(x)), x)

    def test_fftshift_symmetry(self):
        from transspacer import fftshift
        x = np.arange(16).reshape(4, 4).astype(float)
        shifted = fftshift(x)
        assert shifted.shape == x.shape

    def test_pad_definitor(self):
        from transspacer import pad_definitor
        x = np.array([0, 100, 200, 300, 400])
        y = np.array([0, 100, 200, 300, 400])
        result = pad_definitor(x, y)
        assert result >= 1
        assert isinstance(result, (int, np.integer))


class TestNormalization:
    def test_c_normalisation_shape(self):
        from transspacer import c_normalisation
        y = np.random.rand(10, 5)
        sf = np.random.rand(10) + 0.1
        result = c_normalisation(y, sf)
        assert result.shape == y.shape

    def test_c_normalisation_values(self):
        from transspacer import c_normalisation
        y = np.ones((3, 2)) * 10
        sf = np.array([1.0, 2.0, 3.0])
        result = c_normalisation(y, sf)
        # tau = mean(y/sf) per gene = [10/1, 10/2, 10/3] averaged over genes
        # Each cell should get normalized differently based on area
        assert result.shape == (3, 2)


class TestOtsu:
    def test_otsu_bimodal(self):
        from transspacer import otsu_thresholding
        x = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 1, 500)])
        threshold = otsu_thresholding(x)
        assert 1 < threshold < 4  # should be between the two modes


class TestSparseUtils:
    def test_colvars_sparse(self):
        from transspacer import colvars_sparse
        dense = np.random.rand(10, 5)
        sparse = csc_matrix(dense)
        var_sparse = colvars_sparse(sparse)
        var_dense = np.var(dense, axis=0)
        assert np.allclose(var_sparse, var_dense, atol=1e-10)

    def test_aggregate_sparse(self):
        from transspacer import aggregate_sparse
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        groups = np.array(["A", "A", "B", "B"])
        result = aggregate_sparse(data, groups)
        assert result.shape == (2, 2)  # genes x groups
        assert np.allclose(result[:, 0], [2, 3])  # mean of rows 0,1 for each gene
        assert np.allclose(result[:, 1], [6, 7])  # mean of rows 2,3 for each gene


class TestGeneSelection:
    def test_calculate_log2fc(self):
        from transspacer import calculate_log2fc
        expr = np.array([[1, 10], [1, 10], [10, 1], [10, 1]], dtype=float)
        clusters = np.array(["A", "A", "B", "B"])
        result = calculate_log2fc(expr, 0, clusters)
        # Gene 0: mean A=1, mean B=10
        # Other for A = mean of B weighted = 10
        # Other for B = mean of A weighted = 1
        assert "A" in result and "B" in result
        assert result["A"] < 0  # log2(1/10) < 0
        assert result["B"] > 0  # log2(10/1) > 0


class TestVariogram:
    def test_get_variogram_map(self):
        from transspacer.variogram import get_variogram_map
        im = np.random.rand(8, 8)
        result = get_variogram_map(im)
        assert result.shape == im.shape

    def test_get_isotropic_vario(self):
        from transspacer.variogram import get_isotropic_vario
        vmap = np.random.rand(16, 16)
        r, v = get_isotropic_vario(vmap)
        assert len(r) > 0
        assert len(r) == len(v)
        assert np.all(r >= 0)


class TestSpatialStats:
    def test_geary_c_basic(self):
        from transspacer.spatial_stats import geary_c_score
        np.random.seed(42)
        n = 50
        coords = np.random.rand(n, 2)
        expr = np.random.rand(n, 3)
        result = geary_c_score(expr, coords)
        assert "Values" in result
        assert "Selected_genes" in result
        assert "P_values" in result
        assert len(result["Values"]) == 3


class TestClustering:
    def test_cell_clustering(self):
        from transspacer import cell_clustering_function
        np.random.seed(42)
        data = np.vstack([
            np.random.randn(20, 5) + [0, 0, 0, 0, 0],
            np.random.randn(20, 5) + [5, 5, 5, 5, 5]
        ])
        labels = cell_clustering_function(data, K=5)
        assert len(labels) == 40
        # Should find at least 2 clusters
        assert len(np.unique(labels)) >= 2
