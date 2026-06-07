"""Cell clustering, UMAP, and Clusters_maker pipeline."""

import numpy as np
from scipy.sparse import issparse, csc_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

def _check_leiden():
    try:
        import igraph as ig
        import leidenalg
        return ig, leidenalg, True
    except ImportError:
        return None, None, False

def _check_umap():
    try:
        import umap as umap_lib
        return umap_lib, True
    except (ImportError, TypeError, Exception):
        return None, False

from .sparse_utils import colvars_sparse, aggregate_sparse
from .gene_selection import calculate_log2fc


def cell_clustering_function(data_correction: np.ndarray, K: int = 30,
                              metric: str = "euclidean", n_threads: int = 1,
                              resolution: float = 1.0) -> np.ndarray:
    """KNN graph + Louvain/Leiden clustering.

    Parameters
    ----------
    data_correction : np.ndarray
        Reduced data matrix (cells x components).
    K : int
        Number of neighbors.
    metric : str
        Distance metric.
    n_threads : int
        Number of threads.
    resolution : float
        Clustering resolution.

    Returns
    -------
    np.ndarray
        Cluster labels (string array, 1-indexed to match R).
    """
    # Build KNN graph
    nn = NearestNeighbors(n_neighbors=K + 1, metric=metric, n_jobs=n_threads)
    nn.fit(data_correction)
    distances, indices = nn.kneighbors(data_correction)

    # Build adjacency matrix (exclude self-loop at index 0)
    n = data_correction.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        for j_idx in range(1, K + 1):
            j = indices[i, j_idx]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # Symmetrize: KNN_matrix = t(KNN) + KNN (already symmetric from above)

    ig, la, has_leiden = _check_leiden()
    if has_leiden:
        # Use igraph + leiden
        sources, targets = np.where(adj > 0)
        mask = sources < targets
        edges = list(zip(sources[mask], targets[mask]))
        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = [adj[s, t] for s, t in edges]

        partition = la.find_partition(g, la.RBConfigurationVertexPartition,
                                     resolution_parameter=resolution,
                                     n_iterations=-1)
        labels = np.array([str(m + 1) for m in partition.membership])  # 1-indexed
    else:
        # Fallback: simple connected components + modularity-based split
        from sklearn.cluster import SpectralClustering
        n_clusters = max(2, int(n / 100))  # rough estimate
        sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                                assign_labels="discretize", random_state=42)
        labels = sc.fit_predict(adj) + 1  # 1-indexed
        labels = labels.astype(str)

    return labels


def umap_maker(pca_data: np.ndarray, n_components: int = 2,
               random_state: int = 42) -> np.ndarray:
    """Run UMAP on PCA embedding.

    Parameters
    ----------
    pca_data : np.ndarray
        PCA-reduced data (cells x components).
    n_components : int
        UMAP dimensions.
    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        UMAP coordinates (cells x 2).
    """
    umap_lib, has_umap = _check_umap()
    if has_umap:
        reducer = umap_lib.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(pca_data)
    else:
        from sklearn.manifold import TSNE
        return TSNE(n_components=n_components, random_state=random_state).fit_transform(pca_data)


def clusters_maker(expression, shared_genes=None, K: int = 30,
                   metric_used: str = "euclidean", n_threads: int = 1,
                   resolution: float = 1.0, nv: int = 50) -> dict:
    """End-to-end: PCA -> clustering -> mean expression -> log2FC.

    Parameters
    ----------
    expression : np.ndarray or sparse matrix
        Expression matrix (cells x genes).
    shared_genes : list, optional
        Gene indices/names to use. If None, uses all genes.
    K : int
        Number of neighbors for clustering.
    metric_used : str
        Distance metric.
    n_threads : int
        Number of threads.
    resolution : float
        Clustering resolution.
    nv : int
        Number of PCA components.

    Returns
    -------
    dict
        PCA_data, Data_correction, Clustering, Mean_expression, Log2FC_table.
    """
    from sklearn.decomposition import TruncatedSVD

    if shared_genes is not None:
        if issparse(expression):
            data_correction = expression[:, shared_genes]
        else:
            data_correction = expression[:, shared_genes]
    else:
        data_correction = expression

    # PCA via SVD
    if issparse(data_correction):
        svd = TruncatedSVD(n_components=min(nv, data_correction.shape[1] - 1),
                           random_state=42)
        pca_u = svd.fit_transform(data_correction)
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(nv, data_correction.shape[1]), random_state=42)
        pca_u = pca.fit_transform(data_correction)

    # Clustering
    clustering = cell_clustering_function(pca_u, K, metric_used, n_threads, resolution)

    # Mean expression per cluster
    if issparse(data_correction):
        mean_expression = aggregate_sparse(data_correction, clustering).T  # genes x groups -> groups x genes
    else:
        groups = np.unique(clustering)
        mean_expression = np.zeros((len(groups), data_correction.shape[1]))
        for i, g in enumerate(groups):
            idx = np.where(clustering == g)[0]
            mean_expression[i, :] = data_correction[idx, :].mean(axis=0)

    # Log2FC
    n_genes = expression.shape[1]
    genes = list(range(n_genes))
    log2fc_list = {}
    for g_idx in genes:
        log2fc_list[g_idx] = calculate_log2fc(
            expression if not issparse(expression) else expression.toarray(),
            g_idx, clustering
        )

    return {
        "PCA_data": pca_u,
        "Data_correction": data_correction,
        "Clustering": clustering,
        "Mean_expression": mean_expression,
        "Log2FC_table": log2fc_list
    }
