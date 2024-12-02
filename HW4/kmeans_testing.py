from kmeans import KMeans
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic data, Creating some fake data for clustering.
X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# Set K-Means hyperparameters
hyperparameters = {'K': 4, 'tau': 1e-4, 's': 100}

# Instantiate and train K-Means , This is where the K-Means algorithm figures out the clusters.
kmeans = KMeans(hyperparameters)
kmeans.train(X_blobs, y_blobs)

# Predict cluster assignments
predictions = kmeans.predict(X_blobs) # Assigns each data point to a cluster based on its features.

# Output centroids
print("Centroids:\n", kmeans.parameters)

# Plot the data points colored by cluster assignments
plt.figure(figsize=(8, 6))
for cluster in np.unique(predictions):
    plt.scatter(
        X_blobs[predictions == cluster, 0],
        X_blobs[predictions == cluster, 1],
        label=f"Cluster {cluster}"
    )

# Plot centroids
plt.scatter(
    kmeans.parameters[:, 0],
    kmeans.parameters[:, 1],
    color='black',
    marker='x',
    s=200,
    label='Centroids'
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()
