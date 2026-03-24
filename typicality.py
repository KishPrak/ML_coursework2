import numpy as np
from sklearn.neighbors import NearestNeighbors

MIN_CLUSTER_SIZE = 5
MAX_KNN          = 20

def compute_typicality(cluster_embeddings: np.ndarray) -> np.ndarray:
    M = len(cluster_embeddings)
    k = min(MAX_KNN, M - 1)
    if k < 1:
        return np.zeros(M)
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nbrs.fit(cluster_embeddings)
    distances, _ = nbrs.kneighbors(cluster_embeddings)
    mean_dist  = distances.mean(axis=1)
    typicality = 1.0 / (mean_dist + 1e-8)
    return typicality