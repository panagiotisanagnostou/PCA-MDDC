from __utils__ import silhouette_score, create_folder_if_not_exists
from algos.pca_mddc import PCA_MDDC
from algos.incremental_silhouette_score import incremental_silhouette_score
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
import pickle
import time

destination = "results/time_analysis/"
create_folder_if_not_exists(destination)

if __name__ == "__main__":
    # # Q2 dataset: Our dataset
    name = "Q2"
    with open('data/' + name + '.pkl', 'rb') as hf:
        X, y = pickle.load(hf)

    pca_mddc_sc = PCA_MDDC(
        max_clusters_number=1024,
        num_of_investigations=10 if X.shape[1] >= 10 else X.shape[1],
        min_sample_split=8,
        percentile=0.1,
        visualization_utility=True,
    ).fit(X)
    dist = pairwise_distances(X)
    output_matrix = pca_mddc_sc.output_matrix

    iteration = 1
    ticks = []
    silhouette_time = []
    inc_silhouette_time = []
    inc_silhouette_values = []

    for clus in range(1, 320, 20):
        print("Calculating for ", clus+1)
        print("Time " + time.strftime("%Y-%m-%d %H:%M:%S"))
        ticks.append(clus+1)
        calculate = output_matrix[:,:clus]

        tic = time.time()
        for i in range(iteration):
            silh = silhouette_score(calculate, dist)
        toc = time.time()
        silhouette_time.append((toc - tic) / iteration)

        tic = time.time()
        for i in range(iteration):
            inc_silh, noc = incremental_silhouette_score(dist, clustering_matrix=calculate, metric="precomputed")
            inc_silhouette_values.append(inc_silh[-1])
        toc = time.time()
        inc_silhouette_time.append((toc - tic) / iteration)

        with open(destination + "silh_time_" + name + ".pkl", "wb") as f:
            pickle.dump({
                "ticks": ticks,
                "silhouette_time": silhouette_time,
                "inc_silhouette_values": inc_silhouette_values,
                "inc_silhouette_time": inc_silhouette_time
            }, f)

    print("End Time " + time.strftime("%Y-%m-%d %H:%M:%S"))

    with open(destination + "silh_time_" + name + ".pkl", "rb") as f:
        results = pickle.load(f)

    plt.figure(figsize=(6, 4))
    plt.plot(results["ticks"], results["silhouette_time"], label="Silhouette Time")
    plt.plot(results["ticks"], results["inc_silhouette_time"], label="Incremental Silhouette Time")
    plt.legend()
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Cumulative Time (seconds)")
    # plt.title("Time Analysis")
    plt.savefig(destination + name + "_time.pdf")