import pickle
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt


def silhouette_score(om, distance_matrix):
    n, n_clusterings = om.shape
    silhouette_scores = []

    for clustering_index in range(n_clusterings):
        cluster_ids = om[:, clustering_index]
        unique_clusters = np.unique(cluster_ids)

        # Precompute masks for clusters
        cluster_masks = {cluster: (cluster_ids == cluster) for cluster in
                         unique_clusters}

        # Array to store silhouette scores for all samples
        scores = np.zeros(n)

        for i in range(n):
            # Get the cluster of the current sample
            current_cluster = cluster_ids[i]

            # Compute a(i): Mean intra-cluster distance
            intra_mask = cluster_masks[current_cluster].copy()
            intra_mask[i] = False  # Exclude the current sample
            intra_distances = distance_matrix[i, intra_mask]
            a_i = intra_distances.mean() if intra_distances.size > 0 else 0

            # Compute b(i): Mean nearest-cluster distance
            b_i = float('inf')
            for cluster, mask in cluster_masks.items():
                if cluster != current_cluster:
                    inter_distances = distance_matrix[i, mask]
                    if inter_distances.size > 0:
                        b_i = min(b_i, inter_distances.mean())

            # Compute silhouette score for the sample
            if max(a_i, b_i) > 0:
                scores[i] = (b_i - a_i) / max(a_i, b_i)
            else:
                scores[i] = 0

        # Compute the average silhouette score for this clustering
        silhouette_scores.append(scores.mean(dtype=float))
    return silhouette_scores


def reassign_clusters(clusters, distances, k):
    cluster_counts = Counter(clusters)
    largest_clusters = set(
        [cluster_id for cluster_id, _ in cluster_counts.most_common(k)])

    unique_clusters = np.unique(clusters)
    cluster_centers = {}
    for cluster_id in unique_clusters:
        cluster_points = np.where(clusters == cluster_id)[0]
        cluster_centers[cluster_id] = np.mean([
            np.mean(distances[i, cluster_points]) for i in cluster_points
        ])

    reassigned_clusters = clusters.copy()
    for cluster_id in unique_clusters:
        if cluster_id not in largest_clusters:
            small_cluster_points = np.where(clusters == cluster_id)[0]
            for point in small_cluster_points:
                distances_to_robust = {
                    robust_id: distances[
                        point, np.where(clusters == robust_id)[0]].mean()
                    for robust_id in largest_clusters
                }
                reassigned_clusters[point] = min(distances_to_robust,
                                                 key=distances_to_robust.get)

    un = np.unique(reassigned_clusters)
    map = {un[i]: i for i in range(un.shape[0])}
    reassigned_clusters = np.array([map[i] for i in reassigned_clusters])

    return reassigned_clusters


def plot_silhouette(ipddp, multiPC, scam, name, destination):
    plt.figure(figsize=(5, 3.8))
    plt.plot(np.arange(2, len(ipddp) + 2), ipddp, label="iPDDP")
    plt.plot(np.arange(2, len(multiPC) + 2), multiPC, label="PCA-MMDC")
    plt.plot(np.arange(2, len(scam) + 2), scam, label="PCA-MMDC-sc")
    # plt.title(name)
    plt.legend()
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.legend(loc='upper right')
    if name == "S4":
        plt.ylim(-0.18, 0.45)
    plt.savefig(destination + name + "_silhouette.pdf", bbox_inches='tight')
    plt.close()


def create_folder_if_not_exists(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")