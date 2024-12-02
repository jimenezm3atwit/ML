import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    """

    def __init__(self, n_components):
        """
        Initialize the PCA class.
        :param n_components: Number of principal components to retain.
        """
        self.n_components = n_components
        self.mean = None
        self.std = None
        self.components = None

    def train(self, X, y=None):
        """
        Fit the PCA model to the dataset X.
        :param X: Input data matrix.
        :param y: Ignored. Included for compatibility with superclass.
        """
        # Standardize the dataset
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_standardized = self.data_standardization(X, self.mean, self.std)

        # Compute SVD
        U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)

        # Select the top n_components
        self.components = Vt[:self.n_components]

    def transform(self, X):
        """
        Project the data onto the principal components.
        :param X: Input data matrix.
        :return: Reduced dimensionality dataset.
        """
        if self.components is None:
            raise ValueError("Model has not been trained yet. Call train() before transform().")

        # Standardize the input data
        X_standardized = self.data_standardization(X, self.mean, self.std)

        # Project data onto the principal components
        return np.dot(X_standardized, self.components.T)

    def predict(self, X):
        """
        Apply PCA transformation and reconstruct the original dataset.
        :param X: Input data matrix.
        :return: Reconstructed data matrix.
        """
        if self.components is None:
            raise ValueError("Model has not been trained yet. Call train() before predict().")

        # Project data to reduced dimensions
        X_compressed = self.transform(X)

        # Reconstruct the data
        X_reconstructed_standardized = np.dot(X_compressed, self.components)

        # Undo standardization to approximate original data
        return self.data_undo_standardization(X_reconstructed_standardized, self.mean, self.std)

    def data_standardization(self, X, mean_vector, std_vector):
        """
        Standardize the dataset.
        :param X: Input data matrix.
        :param mean_vector: Mean of each feature.
        :param std_vector: Standard deviation of each feature.
        :return: Standardized data matrix.
        """
        X_centered = X - mean_vector
        std_vector[std_vector == 0] = 1  # Avoid division by zero
        return X_centered / std_vector

    def data_undo_standardization(self, X_standardized, mean_vector, std_vector):
        """
        Revert the standardization of the dataset.
        :param X_standardized: Standardized data matrix.
        :param mean_vector: Mean of each feature used during standardization.
        :param std_vector: Standard deviation of each feature used during standardization.
        :return: Original scale data matrix.
        """
        return (X_standardized * std_vector) + mean_vector
