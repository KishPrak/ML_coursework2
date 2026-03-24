import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

MAX_CLUSTERS_CIFAR = 500

def compute_k(n_labeled, budget, max_clusters = MAX_CLUSTERS_CIFAR):
    return min(n_labeled + budget, max_clusters)

def cluster(embeddings, k, random_state= 42):
    algo = "KMeans" if k <= 50 else "MiniBatchKMeans"
    if k <= 50:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    else:
        model = MiniBatchKMeans(
            n_clusters=k, n_init=10, batch_size=4096, random_state=random_state
        )
    assignments = model.fit_predict(embeddings)
    sizes= np.bincount(assignments)
    print(f"Sizes — min:{sizes.min()} max:{sizes.max()} mean:{sizes.mean():.1f}")
    return assignments