from pca import PCA
from kmeans import KMeans, SemiSupervisedKMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target  # True labels

# PCA Implementation
print("Testing PCA Implementation")
pca = PCA(n_components=2)
pca.train(X_iris)
X_reduced = pca.transform(X_iris)
X_reconstructed = pca.predict(X_iris)
reconstruction_error = np.mean((X_iris - X_reconstructed) ** 2)

print("Original data shape:", X_iris.shape)
print("Reconstructed data shape:", X_reconstructed.shape)
print("Reconstruction Error:", reconstruction_error)

# KMeans Implementation
print("\nTesting KMeans Implementation")
kmeans_hyperparameters = {'K': 3, 'tau': 1e-4, 's': 100}
kmeans = KMeans(kmeans_hyperparameters)
kmeans.train(X_reduced, y_iris)
kmeans_clusters = kmeans.predict(X_reduced)

print("Cluster assignments:", np.unique(kmeans_clusters))

# Semi-Supervised KMeans Implementation
print("\nTesting Semi-Supervised KMeans Implementation")
semi_kmeans_hyperparameters = {'K': 3, 'tau': 1e-4, 's': 100}
y_semi = np.full_like(y_iris, -1)  # Mark all as unlabeled
y_semi[:30] = y_iris[:30]  # Assume 30 samples are labeled
semi_kmeans = SemiSupervisedKMeans(semi_kmeans_hyperparameters)
semi_kmeans.train(X_reduced, y_semi)
semi_kmeans_clusters = semi_kmeans.predict(X_reduced)

print("Semi-Supervised Cluster assignments:", np.unique(semi_kmeans_clusters))

# Visualizations
plt.figure(figsize=(12, 6))

# Plot KMeans results
plt.subplot(1, 2, 1)
for cluster in np.unique(kmeans_clusters):
    plt.scatter(
        X_reduced[kmeans_clusters == cluster, 0],
        X_reduced[kmeans_clusters == cluster, 1],
        label=f"Cluster {cluster}"
    )
plt.title("K-Means Clustering (PCA Reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()

# Plot Semi-Supervised KMeans results
plt.subplot(1, 2, 2)
for cluster in np.unique(semi_kmeans_clusters):
    plt.scatter(
        X_reduced[semi_kmeans_clusters == cluster, 0],
        X_reduced[semi_kmeans_clusters == cluster, 1],
        label=f"Cluster {cluster}"
    )
plt.title("Semi-Supervised K-Means Clustering (PCA Reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
