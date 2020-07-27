import numpy as np

def featureNormalize(X):
    X_norm = X
    mu = np.mean(X)
    sigma = np.std(X)
    m = X.shape[0]
    mu_matrix = np.ones((m, 1)) * mu
    sigma_matrix = np.ones((m, 1)) * sigma
    return X_norm
