# refine_embeddings.py
import numpy as np
import optuna
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from active_learning import knn_smoothing

EMB_DIR = "./embeddings"
train_emb = normalize(np.load(f"{EMB_DIR}/train_embeddings.npy"))

def objective(trial):
    k      = trial.suggest_int("k", 5, 50)
    alpha  = trial.suggest_float("alpha", 0.1, 0.9)
    n_iter = trial.suggest_int("n_iters", 1, 5)

    smoothed = knn_smoothing(train_emb, k=k, alpha=alpha, n_iters=n_iter)

    # Subsample for speed
    idx = np.random.choice(len(smoothed), 5000, replace=False)
    sub = smoothed[idx]

    km   = KMeans(n_clusters=10, n_init=5, random_state=42)
    pred = km.fit_predict(sub)
    sil  = silhouette_score(sub, pred, sample_size=2000)
    return -sil   # optuna minimises

study = optuna.create_study()
study.optimize(objective, n_trials=50)

print("Best params:", study.best_params)
print(f"Best silhouette: {-study.best_value:.4f}")

# Apply best params and save
best = study.best_params
refined = knn_smoothing(train_emb,
                         k=best["k"],
                         alpha=best["alpha"],
                         n_iters=best["n_iters"])
refined = normalize(refined)
np.save(f"{EMB_DIR}/train_embeddings_refined.npy", refined)
print("Saved refined embeddings")