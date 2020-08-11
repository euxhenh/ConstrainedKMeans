import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from bidict import bidict

FORMAT = '%(levelname)s:%(name)s: %(message)s'
logging.basicConfig(format=FORMAT)


class ConstrainedKMeans:
    """
    Modified version of KMeans algorithm that takes into account
    partial information about the data.

    Given a partial list of known labels `init_labels`, Constrained KMeans
    finds a cluster configuration that complies with `init_labels`.
    `init_labels` is the same length as x.shape[0], which is why
    a second array `can_change` masks out which labels should be
    marked as known and which labels can change.
    Formally, the output of the algorithm is an array `labels` such that
    np.all((labels[can_change == 0] == init_labels[can_change == 0])) is True.
    """

    def __init__(
            self,
            n_clusters: int = 2,
            metric: str = 'euclidean',
            max_steps: int = 300,
            runs: int = 20,
            eps: float = 1e-9,
            manage_empty: str = 'random',
            logging_lvl: str = 'WARNING'):
        """
        Parameters
        __________

        n_clusters: Number of clusters to use.

        metric: Metric to use for finding distances between points.

        max_steps: Maximum number of steps to run the algorithm for,
            in case convergence is not achieved.

        runs: Will run the algorithm `runs` times using different
            random seeds and will choose the run that achieved the
            best MSE.

        eps: Value to use when checking for convergence.

        manage_empty: Method to use when an empty cluster is determined.
            Can be "random" or "furthest".

        logging_lvl: Specify logging level.
        """
        assert(int(runs) >= 1)
        assert(int(n_clusters) >= 2)
        assert(int(max_steps) >= 1)

        self.n_clusters = int(n_clusters)
        self.metric = str(metric)
        self.max_steps = int(max_steps)
        self.runs = int(runs)
        self.eps = float(eps)
        self.manage_empty = str(manage_empty)
        self.no_constraints = False
        self.no_change = False

        self.logger = logging.getLogger('Constrained KMeans')
        self.logger.setLevel(getattr(logging, logging_lvl))

    def initialize(
            self,
            x: np.ndarray,
            can_change: np.ndarray,
            init_labels: np.ndarray):
        """
        Compute initial centroids for cluster `i` by averaging out the points
        x[(init_labels == i) & (can_change == 0)]. The remaining centroids
        are initialized randomly from the pool of points that are not constrained.

        Parameters
        __________

        x: 2d array containing the data matrix of shape (n_samples, n_features).

        can_change: 1d array of shape (n_samples,). If can_change[i] == 0,
            then the i-th label shall not be changed by the algorithm.
            All other values are ignored.

        init_labels: 1d array of shape (n_samples,). A list of known labels.
            Only init_labels[can_change == 0] are considered. The rest of the
            labels are ignored.
        """
        if self.n_clusters > x.shape[0]:
            raise ValueError("There are more clusters than points.")

        can_change = can_change.astype(int)
        init_labels = init_labels.astype(int)
        if (not np.all(init_labels >= 0)):
            raise ValueError("Negative values found in init_labels.")

        temp_init_labels = init_labels + np.max(init_labels) + 1

        unq_no_change_labels = np.unique(init_labels[can_change == 0])

        # Dicionary that transforms the labels into categoricals
        self.keys = bidict(
            {i: unq_no_change_labels[i] for i in range(len(unq_no_change_labels))})
        for unq_label in unq_no_change_labels:
            temp_init_labels[(init_labels ==
                              unq_label) & (can_change == 0)] = self.keys.inverse[unq_label]

        init_labels = temp_init_labels
        unq_no_change_labels = np.unique(init_labels[can_change == 0])

        self.can_change = can_change
        self.init_labels = init_labels

        # Initialize with zeros
        self.centroids = np.zeros((self.n_clusters, x.shape[1]))

        # In case no constraints are passed, initialize all centroids at random.
        if unq_no_change_labels.size == 0:
            self.no_constraints = True
            self.centroids = x[np.random.choice(x.shape[0], self.n_clusters)]
            return

        # In case all points are constrained, there is nothing to do
        if np.all((can_change == 0)):
            self.no_change = True
            return

        # Initialize centroids by the mean of the constrained points
        for unq_label in unq_no_change_labels:
            self.centroids[unq_label] = np.mean(
                x[(init_labels == unq_label) & (can_change == 0)], axis=0)

        # Determine the indices for the remaining centroids
        # and initialize at random
        centroids_remaining = set(
            np.arange(self.n_clusters)) - set(unq_no_change_labels)
        n_centroids_remaining = len(centroids_remaining)

        random_points = iter(np.random.choice(
            x[can_change != 0].shape[0], n_centroids_remaining))

        for i in centroids_remaining:
            self.centroids[i] = x[can_change != 0][next(random_points)]

    def e_step(self, x: np.ndarray):
        """
        Compute all pairwise distances between points in x and centroids.
        Determine smallest distances and assign labels to points.
        """
        dists = cdist(x, self.centroids, metric=self.metric)
        labels = np.argmin(dists, axis=1)

        if not self.no_constraints:
            labels[self.can_change == 0] = self.init_labels[self.can_change == 0]

        self.min_dists = dists[np.arange(labels.shape[0]), labels]
        self.labels = labels

    def m_step(self, x: np.ndarray):
        """
        Update centroids by taking averages of points in the corresponding
        cluster. A check is added to see if the cluster ended up being
        empty in which case a re-assignment is performed.
        """
        for label in np.arange(self.n_clusters):
            if np.all(((self.labels == label) == False)):
                # Manage empty cluster
                self.logger.info("Encountered empty cluster.")
                self.manage_empty_cluster(x, label)
            else:
                self.centroids[label] = np.mean(
                    x[self.labels == label], axis=0)

    def manage_empty_cluster(self, x: np.ndarray, cluster_index: int):
        """
        Manage an empty cluster by assigning to the corresponding
        centroid a random datapoint if manage_empty == 'random',
        or by determining the "biggest" cluster and finding
        the point which is furthest away from it as measured by
        self.metric.
        """
        if self.manage_empty == "furthest":
            lbs, counts = np.unique(self.labels, return_counts=True)
            biggest_cluster_i = lbs[np.argmax(counts)]
            furthest_point_i = np.argmax(
                cdist(x, self.centroids[biggest_cluster_i].reshape(1, -1),
                      metric=self.metric))
            self.centroids[cluster_index] = x[furthest_point_i]
            self.labels[furthest_point_i] = cluster_index
            self.min_dists[furthest_point_i] = 0
        elif self.manage_empty == "random":
            rand_point = np.random.randint(0, x.shape[0])
            self.centroids[cluster_index] = x[rand_point]
            self.labels[rand_point] = cluster_index
            self.min_dists[rand_point] = 0
        else:
            raise ValueError("Empty cluster manage method not found.")

    def compute_mse(self):
        """
        Mean Squared Error
        """
        return np.mean(self.min_dists)

    def fit(self, x, can_change, init_labels):
        """
        Parameters
        __________

        x: 2d array containing the data matrix of shape (n_samples, n_features).

        can_change: 1d array of shape (n_samples,). If can_change[i] == 0,
            then the i-th label shall not be changed by the algorithm.
            All other values are ignored.

        init_labels: 1d array of shape (n_samples,). A list of known labels.
            Only init_labels[can_change == 0] are considered. The rest of the
            labels are ignored.
        """
        labels = np.zeros((self.runs, x.shape[0]))
        mses = np.zeros(self.runs)

        for run in range(self.runs):
            self.initialize(x, can_change, init_labels)
            mse, new_mse = -1, 0
            i = 0  # step counter

            if self.no_change:
                self.labels = self.init_labels
                self.logger.info("No free points found.")
                return

            while i < self.max_steps:
                i += 1
                self.e_step(x)
                self.m_step(x)

                new_mse = self.compute_mse()
                #self.logger.info(f'Iteration {i} :: MSE {new_mse}')
                if abs(new_mse - mse) < self.eps:
                    self.logger.info(
                        f"Converged in {i} steps. MSE: {new_mse}.")
                    break
                mse = new_mse
                if np.isnan(mse):
                    # Should never run
                    self.logger.info(
                        "Empty clusters encountered. Rerunning algorithm.")
                    self.fit(x, can_change, init_labels)
                    break
            else:
                self.logger.info(
                    f"No convergence attained. Terminated in {i} steps.")

            mses[run] = new_mse
            labels[run] = self.labels

        lowest_mse_i = np.argmin(mses)
        self.labels = labels[lowest_mse_i]

        self.logger.info(f"Best MSE achieved: {mses[lowest_mse_i]}.")

    def fit_predict(self, x, can_change, init_labels):
        """
        Parameters
        __________

        x: 2d array containing the data matrix of shape (n_samples, n_features).

        can_change: 1d array of shape (n_samples,). If can_change[i] == 0,
            then the i-th label shall not be changed by the algorithm.
            All other values are ignored.

        init_labels: 1d array of shape (n_samples,). A list of known labels.
            Only init_labels[can_change == 0] are considered. The rest of the
            labels are ignored.
        """
        self.fit(x, can_change, init_labels)
        return self.get_labels()

    def get_labels(self):
        """
        Convert the categorical labels back to their original values.
        Use this function if you wish to get cluster labels from the object.
        """
        # Hack: convert to negative so that there are no collisions
        converted_labels = -self.labels.copy()
        unq_labels = np.unique(self.labels)
        for key in self.keys:
            if key in unq_labels:
                converted_labels[self.labels == key] = self.keys[key]
        new_unq_labels = np.unique(converted_labels)
        busy_labels = new_unq_labels[new_unq_labels >= 0]
        free_labels = iter(set(range(len(new_unq_labels))) - set(busy_labels))
        neg_labels = new_unq_labels[new_unq_labels < 0]
        for neg_label in neg_labels:
            converted_labels[converted_labels == neg_label] = next(free_labels)
        return converted_labels

