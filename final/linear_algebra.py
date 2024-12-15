import numpy as np
import scipy
import scipy.linalg

class LDA:
    def __init__(self):
        self.W = None

    def fit(self, X, y, num_components=2):
        n_features, n_samples = X.shape
        class_labels = np.unique(y)

        mean_overall = np.mean(X, axis=1, keepdims=True)

        S_w = np.zeros((n_features, n_features))
        S_b = np.zeros((n_features, n_features))

        for label in class_labels:
            X_class = X[:, y == label]
            n_class_samples = X_class.shape[1]
            mean_class = np.mean(X_class, axis=1, keepdims=True)

            S_w += (X_class - mean_class) @ (X_class - mean_class).T

            mean_diff = mean_class - mean_overall
            S_b += n_class_samples * (mean_diff @ mean_diff.T)

        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_w) @ S_b)

        sorted_indices = np.argsort(-eigvals.real)
        eigvecs = eigvecs[:, sorted_indices]
        eigvals = eigvals[sorted_indices]

        self.W = eigvecs[:, :num_components].real

    def transform(self, X):
        return self.W.T @ X

class PCA:
    def __init__(self):
        self.mean = None
        self.W = None
    def fit(self, X,num_components=2):
        self.mean = np.mean(X,axis=1,keepdims=True)
        U,s,Vt = scipy.linalg.svd(X - self.mean, full_matrices=False)

        self.W = U[:,:num_components]
    def transform(self, X):
        X_centered = X - self.mean
        return self.W.T @ X_centered


