import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# 2D synthetic dataset
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
D, y = make_blobs(n_samples=100, centers=1, random_state=42)
X = np.dot(D, transformation)  # Anisotropic blobs

# PCA without mean centering
pca = PCA(n_components=2, whiten=False).fit(X)
X_pca = pca.transform(X)

# PCA with mean centering
pca_centered = PCA(n_components=2, whiten=False).fit(X - np.mean(X, axis=0))
X_pca_centered = pca_centered.transform(X - np.mean(X, axis=0))

# Data with first eigenvec
plt.figure(figsize=(14, 7))

# Plot for non-centered
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
eigen_vectors_non_centered = pca.components_
plt.quiver(pca.mean_[0], pca.mean_[1],
           eigen_vectors_non_centered[0][0], eigen_vectors_non_centered[0][1],
           angles='xy', scale_units='xy', scale=1, color='r')
plt.title('PCA without Mean Centering')

# Plot for centered
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
eigen_vectors_centered = pca_centered.components_
plt.quiver(0, 0, eigen_vectors_centered[0][0], eigen_vectors_centered[0][1],
           angles='xy', scale_units='xy', scale=1, color='g')
plt.title('PCA with Mean Centering')
plt.savefig('/meanCenter.png', dpi=300)
plt.show()
