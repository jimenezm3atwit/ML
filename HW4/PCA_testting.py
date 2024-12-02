from HW4.pca import PCA
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target  # True labels

# Instantiate PCA with 2 components
pca = PCA(n_components=2)

# Train PCA on the dataset
pca.train(X_iris)

# Transform the data into reduced dimensionality space
X_compressed = pca.transform(X_iris)

# Plot the reduced data
plt.figure(figsize=(8, 6))
for label in np.unique(y_iris):
    plt.scatter(
        X_compressed[y_iris == label, 0],  # Principal Component 1
        X_compressed[y_iris == label, 1],  # Principal Component 2
        label=f"Class {label}"
    )

plt.title("PCA - 2D Visualization of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()
