import numpy as np

#TODO: fix error with multivariate linear regression
def gradientDescent(X, y, theta, hypothesis, alpha=0.01, num_iters=1500):
    m = y.shape[0]
    for i in range(num_iters):
            h = hypothesis(X, theta)
            errs = h - y
            change = np.dot(X.T, errs)
            theta = theta - alpha*(1/m)*change

    return theta

def featureNormalize(X):
    X_norm = X
    mu = np.mean(X)
    sigma = np.std(X)
    m = X.shape[0]
    mu_matrix = np.ones((m, 1)) * mu
    sigma_matrix = np.ones((m, 1)) * sigma
    return X_norm
