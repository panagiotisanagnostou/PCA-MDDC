
from __utils__ import create_folder_if_not_exists
from algos.pca_mddc import PCA_MDDC
from HiPart.clustering import IPDDP
from scipy.io import loadmat
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import numpy as np
import pickle
import time
import warnings

warnings.filterwarnings("ignore")

destination = "results/simplistic_framework/"
create_folder_if_not_exists(destination)

models = ["iPDDP", "PCA-MDDC", "PCA-MDDC-sc"]

def experiment(X, y, name):
	print(name)
	algo_time = {i: [] for i in models}
	silh = {i: [] for i in models}
	aris = {i: [] for i in models}
	nmi = {i: [] for i in models}

	print("iPDDP: Time " + time.strftime("%Y-%m-%d %H:%M:%S"))
	tic = time.time()
	ipddp = IPDDP(
		max_clusters_number=np.unique(y).shape[0],
		min_sample_split=8,
		percentile=0.2,
		visualization_utility=True,
	).fit(X)
	algo_time["iPDDP"].append(time.time() - tic)

	silh["iPDDP"].append(silhouette_score(X, ipddp.labels_))
	aris["iPDDP"].append(adjusted_rand_score(y, ipddp.labels_))
	nmi["iPDDP"].append(normalized_mutual_info_score(y, ipddp.labels_))

	print("PCA-MDDC: Time " + time.strftime("%Y-%m-%d %H:%M:%S"))
	tic = time.time()
	pca_mddc = PCA_MDDC(
		max_clusters_number=np.unique(y).shape[0],
		num_of_investigations=10 if X.shape[1] >= 10 else X.shape[1],
		min_sample_split=10,
		percentile=0.2,
		visualization_utility=True,
		default=True,
	).fit(X)
	algo_time["PCA-MDDC"].append(time.time() - tic)

	silh["PCA-MDDC"].append(silhouette_score(X, pca_mddc.labels_))
	aris["PCA-MDDC"].append(adjusted_rand_score(y, pca_mddc.labels_))
	nmi["PCA-MDDC"].append(normalized_mutual_info_score(y, pca_mddc.labels_))

	print("PCA-MDDC-sc: Time " + time.strftime("%Y-%m-%d %H:%M:%S"))
	tic = time.time()
	pca_mddc_sc = PCA_MDDC(
		max_clusters_number=np.unique(y).shape[0],
		num_of_investigations=10 if X.shape[1] >= 10 else X.shape[1],
		min_sample_split=10,
		percentile=0.1,
		visualization_utility=True,
	).fit(X)
	algo_time["PCA-MDDC-sc"].append(time.time() - tic)

	silh["PCA-MDDC-sc"].append(silhouette_score(X, pca_mddc_sc.labels_))
	aris["PCA-MDDC-sc"].append(adjusted_rand_score(y, pca_mddc_sc.labels_))
	nmi["PCA-MDDC-sc"].append(normalized_mutual_info_score(y, pca_mddc_sc.labels_))

	with open(destination + "results_" + name + ".pkl", "wb") as f:
		results = [algo_time, silh, aris, nmi]
		pickle.dump(results, f)

	with open(destination + "baseline_results.csv", "a") as outf:
		outf.write("Dataset: " + name + "\n")

		for i in models:
			outf.write(i + "\t")
			outf.write(str(np.round(results[2][i][0], decimals=5)) + "\t")
			outf.write(str(np.round(results[3][i][0], decimals=5)) + "\t")
			outf.write(str(np.round(results[1][i][0], decimals=5)) + "\t")
			outf.write(str(np.round(results[0][i][0], decimals=5)) + "\n")

	return algo_time, silh, aris, nmi


if __name__ == "__main__":
	with open(destination + "baseline_results.csv", "w") as f:
		f.write("Algorithm\tARI\tNMI\tSilhouette\n")

	# # S1 dataset
	name = "S1"
	X = loadmat("data/S1.mat")["data"]
	y = loadmat("data/S1.mat")["label"][:, 0]
	experiment(X, y, name)
	del X, y

	# # # S2 dataset
	name = "S2"
	X = loadmat("data/S2.mat")["data"]
	y = loadmat("data/S2.mat")["label"][:, 0]
	experiment(X, y, name)
	del X, y

	# # S3 dataset
	name = "S3"
	X = loadmat("data/S3.mat")["data"]
	y = loadmat("data/S3.mat")["label"][:, 0]
	experiment(X, y, name)
	del X, y

	# # S4 dataset
	name = "S4"
	X = loadmat("data/S4.mat")["data"]
	y = loadmat("data/S4.mat")["label"][:, 0]
	experiment(X, y, name)
	del X, y

	# # Q1 dataset: Our dataset
	name = "Q1"
	with open('data/' + name + '.pkl', 'rb') as hf:
		X, y = pickle.load(hf)
	experiment(X, y, name)
	del X, y

	# # Q2 dataset: Our dataset
	name = "Q2"
	with open('data/' + name + '.pkl', 'rb') as hf:
		X, y = pickle.load(hf)
	experiment(X, y, name)
	del X, y

	# # Q3 dataset: Our dataset
	name = "Q3"
	with open('data/' + name + '.pkl', 'rb') as hf:
		X, y = pickle.load(hf)
	experiment(X, y, name)
	del X, y

	# # Q4 dataset: Our dataset
	name = "Q4"
	with open('data/' + name + '.pkl', 'rb') as hf:
		X, y = pickle.load(hf)
	experiment(X, y, name)
	del X, y

	# # ecoli dataset
	name = "ecoli"
	X = loadmat("data/ecoli.mat")["data"]
	y = loadmat("data/ecoli.mat")["label"][:, 0]
	experiment(X, y, name)
	del X, y

	# # landsat dataset
	name = "landsat"
	X = loadmat("data/landsat.mat")["data"]
	y = loadmat("data/landsat.mat")["label"][:, 0]
	experiment(X, y, name)
	del X, y

	# # waveform3 dataset
	name = "waveform3"
	X = loadmat("data/waveform3.mat")["data"]
	y = loadmat("data/waveform3.mat")["label"][:, 0]
	experiment(X, y, name)
	del X, y

	print("End!!!")
