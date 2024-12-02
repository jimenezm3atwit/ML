import numpy as np
from HW4.pca import PCA
from HW4.kmeans import KMeans, SemiSupervisedKMeans

def main():
    # Generate synthetic data for testing PCA
    from sklearn.datasets import load_iris
    iris = load_iris()
    X_iris = iris.data

    # PCA Example
    print("Testing PCA Implementation")
    pca = PCA(n_components=2)
    pca.train(X_iris)
    X_reconstructed = pca.predict(X_iris)

    # Print original and reconstructed data shapes
    print("Original data shape:", X_iris.shape)
    print("Reconstructed data shape:", X_reconstructed.shape)

    # Calculate reconstruction error
    reconstruction_error = np.mean((X_iris - X_reconstructed) ** 2)
    print("Reconstruction Error:", reconstruction_error)

    # Generate synthetic data for testing KMeans
    from sklearn.datasets import make_blobs
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

    # KMeans Example
    print("\nTesting KMeans Implementation")
    hyperparameters = {'K': 4, 'tau': 1e-4, 's': 100}
    kmeans = KMeans(hyperparameters)
    kmeans.train(X_blobs, y_blobs)
    predictions = kmeans.predict(X_blobs)

    # Print cluster assignments
    print("Cluster assignments:", np.unique(predictions))

    # Semi-Supervised KMeans Example
    print("\nTesting Semi-Supervised KMeans Implementation")
    # Assume we have labels for 10% of the data
    n_labeled = int(0.1 * len(y_blobs))
    y_semi = np.full_like(y_blobs, -1)
    indices = np.random.choice(len(y_blobs), n_labeled, replace=False)
    y_semi[indices] = y_blobs[indices]

    semi_kmeans = SemiSupervisedKMeans(hyperparameters)
    semi_kmeans.train(X_blobs, y_semi)
    semi_predictions = semi_kmeans.predict(X_blobs)

    # Print cluster assignments
    print("Semi-Supervised Cluster assignments:", np.unique(semi_predictions))

if __name__ == "__main__":
    main()
