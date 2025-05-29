import HiPart.__utility_functions as util
import numpy as np

from HiPart.__partition_class import Partition
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from treelib import Tree


class PCA_MDDC(Partition):
    """
    Class PCA_MDDC. It executes the PCA_MDDC and PCA_MDDC-sc algorithms.
    """

    decreasing = True

    def __init__(
        self,
        max_clusters_number=100,
        num_of_investigations=5,
        percentile=0.1,
        min_sample_split=5,
        visualization_utility=True,
        distance_matrix=False,
        default=False,
        debug=False,
        **decomposition_args,
    ):
        super().__init__(
            max_clusters_number=max_clusters_number,
            min_sample_split=min_sample_split,
            visualization_utility=visualization_utility,
            distance_matrix=distance_matrix,
            **decomposition_args,
        )
        self.num_of_investigations = num_of_investigations
        self.percentile = percentile
        self.default = default
        self.debug = debug

    def fit(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns.

        Returns
        -------
        self
            A iPDDP class type object, with complete results on the
            algorithm's analysis.

        """
        self.X = X
        self.samples_number = X.shape[0]

        # check for the correct form of the input data matrix
        if self.distance_matrix:
            if X.shape[0] != X.shape[1]:
                raise ValueError("IPDDP: distance_matrix: Should be a square matrix")

        # create an id vector for the samples of X
        indices = np.array([int(i) for i in range(X.shape[0])])

        # precompute the distance matrix if necessary # TODO: optimize memory usage
        self.dist = pairwise_distances(X, metric="euclidean")

        # initialize tree and root node
        proj_tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        # Root initialization
        proj_tree.create_node(  # step (0)
            tag="cl_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.cluster_color),
        )
        proj_tree.get_node(self.node_ids).data["silh"] = {1: [-1]}
        proj_tree.get_node(self.node_ids).data["inter_cluster_ind"] = {1: [0]}
        proj_tree.get_node(self.node_ids).data["silh_update"] = True
        # indicator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data
        if not proj_tree.get_node(0).data["split_permission"]:
            raise RuntimeError("iPDDP: cannot split the data at all!!!")

        # Initialize the stopping criterion counter that counts the number
        # of clusters
        self.splits = 1
        while (selected_node is not None) and (self.splits < self.max_clusters_number): #  or self.stop_with_silhouette):
            self.split_function(proj_tree, selected_node)  # step (1 ,2)
            self.splits += 1

            # silhouette score calculation for each individual cluster
            self.tree = proj_tree

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(
                proj_tree.leaves(), decreasing=self.decreasing
            )  # step (3)

        self.tree = proj_tree

        return self

    def calculate_node_data(self, indices, key):
        """

        Parameters
        ----------
        indices : ndarray of shape (n_samples,)
            The index of the samples in the original data matrix.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necessary data for each node which are splitting point.

        """

        proj_vectors = None
        projection = None
        best_solution = None
        splitpoint = None
        split_criterion = None
        flag = False

        # Application of the minimum sample number split
        # =========================
        if indices.shape[0] > self.min_sample_split:
            # Apply the decomposition method on the data matrix
            proj_vectors = PCA(n_components=self.num_of_investigations, svd_solver="full", **self.decomposition_args)
            projection = proj_vectors.fit_transform(self.X[indices, :])

            solutions = []
            for i in range(self.num_of_investigations):
                one_dimension = projection[:, i]
                one_dimension = np.sort(one_dimension)

                quantile_value = np.quantile(
                    one_dimension, (self.percentile, (1 - self.percentile))
                )
                within_limits = np.where(
                    np.logical_and(
                        one_dimension > quantile_value[0],
                        one_dimension < quantile_value[1],
                    )
                )[0]

                # if there is at least one split the allowed percentile of the data
                if within_limits.size > 1:
                    distances = np.diff(one_dimension[within_limits])
                    loc = np.where(np.isclose(distances, np.max(distances)))[0][0]

                    splitpoint = one_dimension[within_limits][loc] + distances[loc] / 2
                    split_criterion = distances[loc]
                    flag = True

                    solutions.append((i, splitpoint, split_criterion, flag))
                else:
                    splitpoint = None
                    split_criterion = 0
                    flag = False

                    solutions.append((i, splitpoint, split_criterion, flag))

            if solutions:
                split = max(solutions, key=lambda x: x[2])
                if split:
                    best_solution = split[0]
                    splitpoint = split[1]
                    if self.visualization_utility:
                        projection = np.concatenate(
                            (projection[:, best_solution][:, np.newaxis], projection[:, 0][:, np.newaxis]),
                            axis=1,
                        ) if best_solution != 0 else np.concatenate(
                            (projection[:, best_solution][:, np.newaxis], projection[:, 1][:, np.newaxis]),
                            axis=1,
                        )
                    else:
                        projection = projection[:, best_solution][:, np.newaxis]

                    split_criterion = split[2]  if self.default else self.scatter_calculation(indices)
                    flag = True

        return {
            "indices": indices,
            "projection": projection,
            "projection_vectors": proj_vectors,
            "best_solution": best_solution,
            "splitpoint": splitpoint,
            "split_criterion": split_criterion,
            "split_permission": flag,
            "color_key": key,
            "dendrogram_check": False,
            "silh": {i: [] for i in range(1, self.max_clusters_number + 1)},
            "inter_cluster_ind": {i: [] for i in range(1, self.max_clusters_number + 1)},
            "silh_update": True,
        }

    def scatter_calculation(self, indices):
        """
        Calculation of the scatter of the data matrix with indices.

        Parameters
        ----------
        indices : ndarray of shape (n_samples,)
            The index of the samples in the original data matrix.

        Returns
        -------
        scat : float
            The scatter of the data matrix with indices.

        """
        centered = util.center_data(self.X[indices, :])
        scat = np.linalg.norm(centered, ord="fro")
        return scat

    @property
    def num_of_investigations(self):
        return self._num_of_investigations

    @num_of_investigations.setter
    def num_of_investigations(self, v):
        if v <= 0 or (not isinstance(v, int)):
            raise ValueError("IPDDP: num_of_investigations: Should be int and > 0")
        self._num_of_investigations = v

    @property
    def percentile(self):
        return self._percentile

    @percentile.setter
    def percentile(self, v):
        if v >= 0.5 or v < 0:
            raise ValueError("IPDDP: percentile: Should be between [0,0.5) interval")
        self._percentile = v