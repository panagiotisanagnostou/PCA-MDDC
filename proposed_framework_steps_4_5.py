from __utils__ import reassign_clusters, create_folder_if_not_exists
from algos.pca_mddc import PCA_MDDC
from HiPart.clustering import IPDDP
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances, silhouette_score

import matplotlib.pyplot as plt
import pickle

destination = "results/proposed_framework_steps_4_5/"
create_folder_if_not_exists(destination)


def bar_plots_of_cluster_points(
        X, y,
        ipddp_point, pca_mddc_point, pca_mddc_sc_point,
        ipddp_desired_k, pca_mddc_desired_k, pca_mddc_sc_desired_k,
        dataset_name
):
    print(dataset_name)
    if X.shape[1] > 2:
        pca = PCA(n_components=3)
        PP = pca.fit_transform(X)
        x_label = "PC 1"
        y_label = "PC 2"
    else:
        PP = X
        x_label = "Feature 1"
        y_label = "Feature 2"

    ipddp = IPDDP(
        max_clusters_number=ipddp_point,
        min_sample_split=8,
        percentile=0.2,
        visualization_utility=True,
    ).fit(X)

    plt.hist(ipddp.labels_, bins=ipddp_point, rwidth=0.8)
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of points")
    plt.savefig(destination + dataset_name + "_on_point_" + str(ipddp_point) + "_ipddp.pdf")
    plt.close()

    y_pred_ipddp = reassign_clusters(ipddp.labels_, pairwise_distances(X), ipddp_desired_k)

    plt.figure(figsize=(5, 3.8))
    plt.scatter(PP[:, 0], PP[:, 1], c=y_pred_ipddp, s=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(destination + dataset_name + "_on_point_" + str(ipddp_point) + "_ipddp_scatter.png")
    plt.close()

    pca_mddc = PCA_MDDC(
        max_clusters_number=pca_mddc_point,
        num_of_investigations=10 if X.shape[1] >= 10 else X.shape[1],
        min_sample_split=10,
        percentile=0.2,
        visualization_utility=True,
        default=True,
    ).fit(X)

    plt.figure(figsize=(5, 3.8))
    plt.hist(pca_mddc.labels_, bins=pca_mddc_point, rwidth=0.8)
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of points")
    plt.savefig(destination + dataset_name + "_on_point_" + str(pca_mddc_point) + "_pca_mddc.pdf")
    plt.close()

    y_pred_pca_mddc = reassign_clusters(pca_mddc.labels_, pairwise_distances(X), pca_mddc_desired_k)

    plt.figure(figsize=(5, 3.8))
    plt.scatter(PP[:, 0], PP[:, 1], c=y_pred_pca_mddc, s=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(destination + dataset_name + "_on_point_" + str(pca_mddc_point) + "_pca_mddc_scatter.png")
    plt.close()

    scam = PCA_MDDC(
        max_clusters_number=pca_mddc_sc_point,
        num_of_investigations=10 if X.shape[1] >= 10 else X.shape[1],
        min_sample_split=10,
        percentile=0.1,
        visualization_utility=True,
    ).fit(X)

    plt.figure(figsize=(5, 3.8))
    plt.hist(scam.labels_, bins=pca_mddc_sc_point, rwidth=0.8)
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of points")
    plt.savefig(destination + dataset_name + "_on_point_" + str(pca_mddc_sc_point) + "_pca_mddc_sc.pdf")
    plt.close()

    y_pred_scam = reassign_clusters(scam.labels_, pairwise_distances(X), pca_mddc_sc_desired_k)

    plt.figure(figsize=(5, 3.8))
    plt.scatter(PP[:, 0], PP[:, 1], c=y_pred_scam, s=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(destination + dataset_name + "_on_point_" + str(pca_mddc_sc_point) + "_pca_mddc_sc_scatter.png")
    plt.close()

    with open(destination + "hard_assignments.csv", "a") as f:
        f.write(dataset_name + "\t")
        f.write("iPDDP\t" + str(adjusted_rand_score(y, y_pred_ipddp)) + "\t" + str(normalized_mutual_info_score(y, y_pred_ipddp)) + "\t" + str(silhouette_score(X, y_pred_ipddp)) + "\n")
        f.write(dataset_name + "\t")
        f.write("PCA-MDDC\t" + str(adjusted_rand_score(y, y_pred_pca_mddc)) + "\t" + str(normalized_mutual_info_score(y, y_pred_pca_mddc)) + "\t" + str(silhouette_score(X, y_pred_pca_mddc)) + "\n")
        f.write(dataset_name + "\t")
        f.write("PCA-MDDC-sc\t" + str(adjusted_rand_score(y, y_pred_scam)) + "\t" + str(normalized_mutual_info_score(y, y_pred_scam)) + "\t" + str(silhouette_score(X, y_pred_scam)) + "\n")


if __name__ == "__main__":

    with open(destination+ "hard_assignments.csv", "w") as f:
        f.write("Dataset\tAlgorithm\tARI\tNMI\tSilhouette\n")

    # # S1 dataset
    name = "S1"
    X = loadmat("data/S1.mat")["data"]
    y = loadmat("data/S1.mat")["label"][:, 0]
    bar_plots_of_cluster_points(
        X, y,
        141, 142, 15,
        15, 15, 15,
         name,
    )
    del X, y

    # # S2 dataset
    name = "S2"
    X = loadmat("data/S2.mat")["data"]
    y = loadmat("data/S2.mat")["label"][:, 0]
    bar_plots_of_cluster_points(
        X, y,
        121, 64, 15,
        15, 15, 15,
        name,
    )
    del X, y

    # # S3 dataset
    name = "S3"
    X = loadmat("data/S3.mat")["data"]
    y = loadmat("data/S3.mat")["label"][:, 0]
    bar_plots_of_cluster_points(
        X, y,
        382, 300, 17,
        15, 15, 15,
        name,
    )
    del X, y

    # # S4 dataset
    name = "S4"
    X = loadmat("data/S4.mat")["data"]
    y = loadmat("data/S4.mat")["label"][:, 0]
    bar_plots_of_cluster_points(
        X, y,
        275, 181, 23,
        15,15, 15,
        name,
    )
    del X, y

    # # Q1 dataset: Our dataset
    name = "Q1"
    with open('data/' + name + '.pkl', 'rb') as hf:
        X, y = pickle.load(hf)
    bar_plots_of_cluster_points(
        X, y,
        48, 49, 5,
        5,5, 5,
        name,
    )
    del X, y

    # # Q2 dataset: Our dataset
    name = "Q2"
    with open('data/' + name + '.pkl', 'rb') as hf:
        X, y = pickle.load(hf)
    bar_plots_of_cluster_points(
        X, y,
        139, 116, 22,
        10, 10, 10,
        name,
    )
    del X, y

    # # Q3 dataset: Our dataset
    name = "Q3"
    with open('data/' + name + '.pkl', 'rb') as hf:
        X, y = pickle.load(hf)
    bar_plots_of_cluster_points(
        X, y,
        233, 332, 18,
        9,9, 9,
        name,
    )
    del X, y

    # # Q4 dataset: Our dataset
    name = "Q4"
    with open('data/' + name + '.pkl', 'rb') as hf:
        X, y = pickle.load(hf)
    bar_plots_of_cluster_points(
        X, y,
        148, 111, 27,
        5, 5, 5,
        name,
    )
    del X, y

    # # ecoli dataset
    name = "ecoli"
    X = loadmat("data/ecoli.mat")["data"]
    y = loadmat("data/ecoli.mat")["label"][:, 0]
    bar_plots_of_cluster_points(
        X, y,
        22, 12, 11,
        8,8, 8,
        name,
    )
    del X, y

    # # landsat dataset
    name = "landsat"
    X = loadmat("data/landsat.mat")["data"]
    y = loadmat("data/landsat.mat")["label"][:, 0]
    bar_plots_of_cluster_points(
        X, y,
        12, 122, 8,
        6,6, 6,
        name,
    )
    del X, y

    # # waveform3 dataset
    name = "waveform3"
    X = loadmat("data/waveform3.mat")["data"]
    y = loadmat("data/waveform3.mat")["label"][:, 0]
    bar_plots_of_cluster_points(
        X, y,
        190, 208, 41,
        3,3, 3,
        name,
    )
    del X, y

    print("End!!!")
