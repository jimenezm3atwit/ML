import numpy as np

class KMeans:
    """
    K-Means clustering algorithm.
    """

    def __init__(self, hyperparameters):
        """
        Initialize the KMeans class.
        :param hyperparameters: Dictionary containing 'K', 'tau', and 's'.
        """
        self.K = hyperparameters.get('K', 3)       # Number of clusters
        self.tau = hyperparameters.get('tau', 1e-4)  # Convergence threshold
        self.s = hyperparameters.get('s', 100)       # Max iterations
        self.parameters = None  # Centroids
        self.cluster_assignments = None

    def train(self, X, y):
        """
        Train the KMeans model.
        :param X: Input data matrix.
        :param y: Ignored. Included for compatibility.
        """
        # Check hyperparameters
        if self.K <= 0 or not isinstance(self.K, int):
            raise ValueError("Number of clusters K must be a positive integer.")
        if self.tau < 0:
            raise ValueError("Threshold tau must be non-negative.")
        if self.s <= 0 or not isinstance(self.s, int):
            raise ValueError("Maximum iterations s must be a positive integer.")

        n_samples, n_features = X.shape
        self.parameters = np.zeros((self.K, n_features))
        previous_parameters = np.zeros_like(self.parameters)

        # Randomly assign clusters
        cluster_assignments = np.random.randint(0, self.K, size=n_samples)
        sample_counts = np.zeros(self.K, dtype=int)

        for iteration in range(self.s):
            # Reset centroids and counts
            self.parameters.fill(0)
            sample_counts.fill(0)

            # Update centroids
            for idx in range(n_samples):
                cluster = cluster_assignments[idx]
                self.parameters[cluster] += X[idx]
                sample_counts[cluster] += 1

            # Avoid division by zero
            for k in range(self.K):
                if sample_counts[k] == 0:
                    sample_counts[k] = 1  # To prevent division by zero
            self.parameters /= sample_counts[:, np.newaxis]

            # Check convergence
            if np.all(np.linalg.norm(self.parameters - previous_parameters, axis=1) < self.tau):
                break
            previous_parameters = self.parameters.copy()

            # Reassign clusters
            for idx in range(n_samples):
                distances = np.linalg.norm(X[idx] - self.parameters, axis=1)
                cluster_assignments[idx] = np.argmin(distances)

        self.cluster_assignments = cluster_assignments

    def predict(self, X):
        """
        Predict cluster assignments for X.
        :param X: Input data matrix.
        :return: Cluster assignments.
        """
        if self.parameters is None:
            raise ValueError("Model has not been trained yet. Call train() before predict().")

        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=int)

        for idx in range(n_samples):
            distances = np.linalg.norm(X[idx] - self.parameters, axis=1)
            predictions[idx] = np.argmin(distances)

        return predictions


class SemiSupervisedKMeans(KMeans):
    """
    Semi-Supervised K-Means clustering algorithm.
    """

    def train(self, X, y):
        """
        Train the Semi-Supervised KMeans model.
        :param X: Input data matrix.
        :param y: Labels for some samples (use -1 or None for unlabeled samples).
        """
        # Initialize variables
        n_samples, n_features = X.shape
        self.parameters = np.zeros((self.K, n_features))
        previous_parameters = np.zeros_like(self.parameters)
        cluster_assignments = np.full(n_samples, -1, dtype=int)
        sample_counts = np.zeros(self.K, dtype=int)

        # Assign clusters based on labels where available
        labeled_indices = np.where(y >= 0)[0]
        unlabeled_indices = np.where(y < 0)[0]

        cluster_assignments[labeled_indices] = y[labeled_indices].astype(int)

        # Randomly assign clusters to unlabeled data
        cluster_assignments[unlabeled_indices] = np.random.randint(0, self.K, size=len(unlabeled_indices))

        for iteration in range(self.s):
            # Reset centroids and counts
            self.parameters.fill(0)
            sample_counts.fill(0)

            # Update centroids (skip labeled data during centroid calculation if specified)
            for idx in unlabeled_indices:
                cluster = cluster_assignments[idx]
                self.parameters[cluster] += X[idx]
                sample_counts[cluster] += 1

            # Include labeled data in centroid calculation
            for idx in labeled_indices:
                cluster = cluster_assignments[idx]
                self.parameters[cluster] += X[idx]
                sample_counts[cluster] += 1

            # Avoid division by zero
            for k in range(self.K):
                if sample_counts[k] == 0:
                    sample_counts[k] = 1  # Prevent division by zero
            self.parameters /= sample_counts[:, np.newaxis]

            # Check convergence
            if np.all(np.linalg.norm(self.parameters - previous_parameters, axis=1) < self.tau):
                break
            previous_parameters = self.parameters.copy()

            # Reassign clusters for unlabeled data only
            for idx in unlabeled_indices:
                distances = np.linalg.norm(X[idx] - self.parameters, axis=1)
                cluster_assignments[idx] = np.argmin(distances)

        self.cluster_assignments = cluster_assignments
