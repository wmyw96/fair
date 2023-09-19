import numpy as np

def least_squares(X, y):
	cov_x = np.matmul(X.T, X)
	cov_xy = np.matmul(X.T, y)
	return np.squeeze(np.dot(np.linalg.inv(cov_x), cov_xy))

