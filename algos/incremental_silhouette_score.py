from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from treelib import Tree

import numpy as np


def incremental_silhouette_score(X, clustering_matrix=None, max_clusters=None, metric="euclidean"):
    """Compute the Incremental Silhouette Coefficient of all samples.

    The Silhouette Coefficient of each sample is calculated using the mean
    intra-cluster distance (a) and the mean nearest-cluster distance (b) for
    each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b).
    To clarify, b is the distance between a sample and the nearest cluster that
    the sample is not a part of.

    Note: This implementation uses the sklearn.metrics.pairwise_distances
    function which is not optimized for sparse matrices.

    Parameters
    ----------
    X  : numpy.ndarray, shape = (n, n) if metric == "precomputed", or, (n, d) otherwise
        Data array representing the clustered samples in either the form of a
        symmetric distance (similarity) matrix if metric == "precomputed", or,
        a data array containing n sample rows and d feature columns.
    clustering_matrix : numpy.ndarray, shape = (n, clusterings-1), default=None
        Clustering matrix containing the hierarchical clustering information of
        a dataset. Each column of the matrix represents a clustering of the data
        and the number of columns represents the number of clusterings, from 2
        to the maximum desired number of clusters.
    max_clusters : int, default=None
        The maximum number of clusters in the linkage matrix to be evaluated by
        the incremental silhouette score.
    metric : str or callable, default="euclidean"
        The distance metric to use when calculating pairwise distances between
        samples in X. If metric is a string, it must be one of the options
        allowed by `scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in `sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. If X is
        a distance matrix, metric must be "precomputed". If the metric parameter
        is a function, it is invoked for every pair of instances (rows). This
        function should accept two rows from 'X' as arguments and return a value
        that represents the distance between them. The outcome of this function
        is then recorded.

    Returns
    -------
    incremental_silhouette : numpy.ndarray, shape = (n_clusters - 1,)
        The silhouette score for each clustering form 2 to the num_of_clusters
        clusters.
    clusters_indicator : numpy.ndarray, shape = (n_clusters - 1,)
        The number of clusters for each silhouette score.

    """

    if metric == "precomputed":
        dist = X
    else:
        dist = pairwise_distances(X, metric=metric)


    silhouette_scores, clusters_indicator = _clustering_matrix_incremental_silhouette_score(dist, clustering_matrix, max_clusters)

    return silhouette_scores, clusters_indicator


def true_labels(tree, n):
    # Regenerate the tl each time from the tree leafs
    tl = np.ones(n)
    # Iterate through the leaves of the tree and assign the labels in
    # increasing order based on the j counter variable
    for i in tree.leaves():
        tl[i.data["indices"]] = i.identifier

    return tl.astype("int64")


def find_silhouette(tree, node_split, splits, dist):
    leaves = tree.leaves()
    tl = true_labels(tree, dist.shape[0])
    le = LabelEncoder()
    labels = le.fit_transform(tl)

    # Precompute intra-cluster distances for clusters that need updating
    for node in leaves:
        idxs = node.data["indices"]
        if node.data["silh_update"]:

            if len(idxs) == 1:
                intra_distances = np.array([0])
            else:
                intra_distances = np.sum(dist[np.ix_(idxs, idxs)], axis=1) / (len(idxs) - 1)
            node.data["intra_distances"] = intra_distances
            node.data["silh_update"] = False  # Reset flag

        # Update inter-cluster distances only for relevant clusters
        s = np.zeros(len(idxs))
        a =  node.data["intra_distances"] # node.data.get("intra_distances", np.zeros(len(idxs)))
        b = np.full(len(idxs), np.inf)
        indicators = np.full(len(idxs), np.inf)

        # Identify clusters to update inter-cluster distances
        inter_clusters_ids = node.data["inter_cluster_ind"].get(splits, set())
        if not inter_clusters_ids:
            my_flag = True
            leaf_ids = [i.identifier for i in leaves]
            inter_clusters_ids = leaf_ids
        inter_clusters = [tree.get_node(cid) for cid in inter_clusters_ids if cid != node.identifier]

        for other_node in inter_clusters:
            other_idxs = other_node.data["indices"]
            distances = dist[np.ix_(idxs, other_idxs)]
            mean_distances = np.mean(distances, axis=1)
            b = np.minimum(b, mean_distances)
            if my_flag:
                indicators.fill(other_node.identifier)
            else:
                indicators = np.argmin(mean_distances, axis=1)
        # Compute silhouette scores
        s = (b - a) / np.maximum(a, b)
        s[np.isnan(s)] = 0  # Handle NaNs
        node.data["silh"][splits] = s
        # node.data["inter_cluster_ind"][splits] = le.inverse_transform(np.unique(indicators))
        node.data["inter_cluster_ind"][splits] = np.unique(indicators)


def incremental_silhouette(tree, max_clusters_number):
    """
    Calculation of the silhouette score for each cluster of the tree.

    Parameters
    ----------
    max_clusters_number : int
        The maximum number of clusters to calculate the incremental silhouette score.
    tree : treelib.Tree
        The object which contains all the information about the execution of
        the iPDDP algorithm.

    Returns
    -------
    scores : numpy.ndarray
        The silhouette score for each clustering form 2 to the num_of_clusters clusters.
    number_of_clusters : numpy.ndarray
        The number of clusters for each silhouette score.

    """

    number_of_clusters = np.arange(max_clusters_number) + 1
    silhouette = {i + 1: [] for i in range(max_clusters_number)}

    for i in tree.all_nodes():
        for j in i.data["silh"]:
            silhouette[j] += list(i.data["silh"][j])

    average_silhouette_score = np.array(
        [np.mean(silhouette[i]) for i in silhouette])[1:]
    number_of_clusters = number_of_clusters[1:]

    return average_silhouette_score, number_of_clusters


def _clustering_matrix_incremental_silhouette_score(dist, clustering_matrix,
                                                    max_clusters_number):
    """
    Transform a clustering matrix into a tree object and calculate the silhouette
    score for each clustering of the tree.

    The transformation of the clustering matrix into a tree object is done by the
    transform_clustering_matrix_to_tree function.

    Parameters
    ----------
    dist : numpy.ndarray
        The distance matrix of the data.
    clustering_matrix : numpy.ndarray
        The clustering matrix containing the hierarchical clustering information
        of a dataset.
    max_clusters_number : int
        The maximum number of clusters to calculate the incremental silhouette
        score.

    Returns
    -------
    incremental_silhouette : numpy.ndarray, shape = (n_clusters - 1,)
        The silhouette score for each clustering form 2 to the num_of_clusters
        clusters.
    clusters_indicator : numpy.ndarray, shape = (n_clusters - 1,)
        The number of clusters for each silhouette score.

    """

    if max_clusters_number is not None:
        if max_clusters_number > clustering_matrix.shape[1] + 1:
            raise ValueError(
                "The maximum number of clusters should be less than the number"
                " of columns of the clustering matrix.")
    else:
        max_clusters_number = clustering_matrix.shape[1] + 1

    tree = transform_clustering_matrix_to_tree(clustering_matrix,
                                               max_clusters_number)

    root = tree.get_node(tree.root)
    root.data["indices"] = np.array(list(root.data["indices"]), dtype="int64")
    root.data["silh"] = {1: [-1]}
    root.data["inter_cluster_ind"] = {1: [0]}
    root.data["silh_update"] = True

    new_tree = Tree()
    new_tree.add_node(root, parent=None)

    splits = 1
    count = 1
    for i in tree.all_nodes()[1:]:
        i.data["indices"] = np.array(list(i.data["indices"]), dtype="int64")
        i.data["silh"] = {i: [] for i in range(1, max_clusters_number + 1)}
        i.data["inter_cluster_ind"] = {i: [] for i in range(1, max_clusters_number + 1)}
        i.data["silh_update"] = True

        node_split = tree.parent(i.identifier)
        new_tree.add_node(i, node_split)

        if count == 2:
            splits += 1
            find_silhouette(new_tree, node_split, splits, dist)
            count = 0

        count += 1

    return incremental_silhouette(new_tree, max_clusters_number)


def transform_clustering_matrix_to_tree(clustering_matrix, max_clusters_number):
    """
    Transform a clustering matrix into a tree object.

    Parameters
    ----------
    clustering_matrix : numpy.ndarray
        The clustering matrix containing the hierarchical clustering information
        of a dataset.
    max_clusters_number : int
        The maximum number of clusters to calculate the incremental silhouette
        score.

    Returns
    -------
    tree : treelib.Tree
        The complete tree object containing all

    """

    indices = np.arange(clustering_matrix.shape[0])

    tr = Tree()
    tr.create_node(
        tag=0,
        identifier=0,
        data={
            "indices": indices,
            "split": 0,
            "cluster": 0,
            "silh": {1: [-1]},
            "inter_cluster_ind": {1: [0]},
            "silh_update": True,
            "intra_distances": np.zeros(indices.shape[0]),
        },
    )

    if np.unique(clustering_matrix[:, 0]).shape[0] == 2:
        loop_adjastments = 0
        previous_clustering = np.zeros(clustering_matrix.shape[0])
    elif np.unique(clustering_matrix[:, 0]).shape[0] == 1:
        loop_adjastments = 1
        previous_clustering = clustering_matrix[:, 0]
    else:
        raise ValueError(
            "The clustering matrix should start from number of clusters equal"
            " to either 1 or 2")

    node_idx = 1
    for column in range(loop_adjastments,
                        max_clusters_number + loop_adjastments - 1):
        cc = clustering_matrix[:, column]

        p_clusters = np.unique(previous_clustering)
        for i in p_clusters:
            new_clusters = np.unique(cc[previous_clustering == i])
            if new_clusters.shape[0] == 2:
                lr = new_clusters[0]
                ll = new_clusters[1]
            if new_clusters.shape[0] > 2:
                raise ValueError(
                    "clustering_matrix was not generated by a divisive clustering algorithm")

        new_clsuters = indices[np.logical_or(cc == ll, cc == lr)]

        parent_cluster = np.unique(previous_clustering[new_clsuters])
        if parent_cluster.shape[0] != 1:
            raise ValueError(
                "clustering_matrix was not generated by a divisive clustering algorithm")
        parent_cluster = parent_cluster[0]
        parent_node = [i.identifier for i in tr.all_nodes() if
                       i.data["cluster"] == parent_cluster][0]

        tr.create_node(
            tag=node_idx,
            identifier=node_idx,
            data={
                "indices": indices[cc == lr],
                "cluster": lr,
                "silh": {},
                "inter_cluster_ind": {},
                "silh_update": True,
                "intra_distances": np.zeros(indices.shape[0]),
            },
            parent=parent_node,
        )
        node_idx += 1

        tr.create_node(
            tag=node_idx,
            identifier=node_idx,
            data={
                "indices": indices[cc == ll],
                "cluster": ll,
                "silh": {},
                "inter_cluster_ind": {},
                "silh_update": True,
                "intra_distances": np.zeros(indices.shape[0]),
            },
            parent=parent_node,
        )
        node_idx += 1

        previous_clustering = cc

    return tr
