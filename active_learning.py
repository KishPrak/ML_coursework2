import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from cluster import cluster, compute_k, MAX_CLUSTERS_CIFAR
from typicality import compute_typicality, MIN_CLUSTER_SIZE
from sklearn.neighbors import NearestNeighbors



def run_tpcrp(embeddings,total_budget,max_clusters = MAX_CLUSTERS_CIFAR,random_state=42):
    N = len(embeddings)
    labeled_set = set()
    queried = []


    k = compute_k(n_labeled=0, budget=total_budget, max_clusters=max_clusters)
    assignments = cluster(embeddings, k=k, random_state=random_state)

    for step in range(total_budget):
        labeled_counts = np.zeros(k, dtype=int)
        cluster_sizes  = np.zeros(k, dtype=int)
        for c in range(k):
            idxs= np.where(assignments == c)[0]
            cluster_sizes[c] = len(idxs)
            labeled_counts[c] = sum(1 for i in idxs if i in labeled_set)

        valid = [c for c in range(k) if cluster_sizes[c] > MIN_CLUSTER_SIZE and cluster_sizes[c] > labeled_counts[c]]
        if not valid:
            print(f"No valid clusters at step {step+1}, stopping early.")
            break

        #get cluster with fewest labeled points
        min_labeled = min(labeled_counts[c] for c in valid)
        least_labeled = [c for c in valid if labeled_counts[c] == min_labeled]
        selected = max(least_labeled, key=lambda c: cluster_sizes[c])

        cluster_idxs = np.where(assignments == selected)[0]
        unlabeled_in_cluster = np.array([i for i in cluster_idxs if i not in labeled_set])

        #get tpicality
        scores= compute_typicality(embeddings[unlabeled_in_cluster])
        query_idx = unlabeled_in_cluster[np.argmax(scores)]

        queried.append(query_idx)
        labeled_set.add(query_idx)

        if (step + 1) % 100 == 0:
            print(f"TPCrp queried {step+1}/{total_budget}")

    return np.array(queried, dtype=int)



def knn_smoothing(embeddings: np.ndarray,k = 20,alpha = 0.5,n_iters = 1):
    X = embeddings.copy()
    for _ in range(n_iters):
        nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
        distances, indices = nbrs.kneighbors(X)
        X_new = np.zeros_like(X)
        for i in range(X.shape[0]):
            neighbors = X[indices[i]] 
            sims = 1 - distances[i]
            sims = np.maximum(sims, 0)
            sims /= sims.sum() + 1e-8
            neighbor_avg = np.sum(neighbors * sims[:, None], axis=0)
            X_new[i] = (1 - alpha) * X[i] + alpha * neighbor_avg
        X = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
    return X



def run_random(n_total,total_budget,random_state= 42):
    rand = np.random.default_rng(random_state)
    return rand.choice(n_total, size=total_budget, replace=False)



def evaluate_linear(train_embeddings,train_labels,test_embeddings,test_labels,queried_indices,budget):
    indices = queried_indices[:budget]

    X_train = train_embeddings[indices]
    y_train = train_labels[indices]
    X_test  = test_embeddings
    y_test  = test_labels

    X_train = normalize(X_train)
    X_test  = normalize(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        C=0.1,          
        solver="lbfgs",
        random_state=0,
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) * 100
    return acc


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    EMB_DIR = "./embeddings_external"
    #EMB_DIR = "./refined_embeddings"

    #get embeddings
    train_emb = np.load(f"{EMB_DIR}/train_embeddings.npy")
    train_lbl = np.load(f"{EMB_DIR}/train_labels.npy")
    test_emb  = np.load(f"{EMB_DIR}/test_embeddings.npy")
    test_lbl  = np.load(f"{EMB_DIR}/test_labels.npy")
    print(f"  Train: {train_emb.shape}  Test: {test_emb.shape}")


    max_budget = 1000
    eval_at    = [10, 20, 30, 40, 50, 100, 200, 500, 1000]
    eval_at    = [b for b in eval_at if b <= max_budget]

    #run tpcrp
    print(f"\nRunning TPCrp (budget={max_budget})")
    tpcrp_indices = run_tpcrp(train_emb, total_budget=max_budget)
    np.save(f"{EMB_DIR}/tpcrp_indices.npy", tpcrp_indices)

    tpcrp_acc = []
    for b in eval_at:
        acc = evaluate_linear(train_emb, train_lbl, test_emb, test_lbl, tpcrp_indices, b)
        tpcrp_acc.append(acc)
        print(f"TPCrp  budget={b:>5}  acc={acc:.2f}%")


    print(f"\nRunning random baseline (budget={max_budget})")
    n_seeds = 5
    random_acc = np.zeros((n_seeds, len(eval_at)))

    for seed in range(n_seeds):
        rand_indices = run_random(len(train_emb), max_budget, random_state=seed)
        for j, b in enumerate(eval_at):
            acc = evaluate_linear(train_emb, train_lbl, test_emb, test_lbl, rand_indices, b)
            random_acc[seed, j] = acc
        print(f"Seed {seed} done")

    random_mean = random_acc.mean(axis=0)
    random_std  = random_acc.std(axis=0)

    for j, b in enumerate(eval_at):
        print(f"  Random budget={b:>5}  acc={random_mean[j]:.2f}% ± {random_std[j]:.2f}%")

    #summary table
    print("\n" + "="*55)
    print(f"{'Budget':>8}  {'TPCrp':>10}  {'Random (mean±std)':>20}")
    print("-"*55)
    for j, b in enumerate(eval_at):
        print(f"{b:>8}  {tpcrp_acc[j]:>9.2f}%  "
              f"{random_mean[j]:>8.2f}% ± {random_std[j]:.2f}%")
    print("="*55)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    ax.plot(eval_at, tpcrp_acc, "o-",
            color="#4fa3e0", linewidth=2, markersize=6, label="TPCrp (ours)")
    ax.plot(eval_at, random_mean, "s--",
            color="#e07a4f", linewidth=2, markersize=6, label="Random baseline")
    ax.fill_between(
        eval_at,
        random_mean - random_std,
        random_mean + random_std,
        alpha=0.2, color="#e07a4f"
    )

    ax.set_xlabel("Labeling budget", color="#cccccc")
    ax.set_ylabel("Test accuracy (%)", color="#cccccc")
    ax.set_title("TPCrp vs Random — CIFAR-10", color="#ffffff", pad=12)
    ax.tick_params(colors="#cccccc")
    ax.legend(facecolor="#1a1a1a", labelcolor="#cccccc")
    ax.grid(True, alpha=0.2, color="#444444")
    ax.set_xscale("log")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    plt.tight_layout()
    plt.savefig(f"{EMB_DIR}/tpcrp_vs_random.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nPlot saved to {EMB_DIR}/tpcrp_vs_random.png")
    plt.show()
