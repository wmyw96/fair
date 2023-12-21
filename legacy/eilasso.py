import numpy as np
import heapq
import cvxpy as cp

def environment_invariant_lasso(x_list, y_list, gamma, show_log=False):
	num_envs = len(x_list)
	covs_xx = []
	covs_xy = []
	covs_yy = []
	dim_x = np.shape(x_list[0])[1]

	ns = 0
	for i in range(num_envs):
		x, y = x_list[i], y_list[i]
		n = np.shape(x)[0]
		ns += n
		if show_log:
			print(f'Env = {i}, number of samples = {n}')

		# x: [n, p] matrix
		# y: [n, 1] matrix
		covs_xx.append(np.matmul(np.transpose(x), x) / n)
		covs_xy.append(np.matmul(np.transpose(x), y) / n)
		covs_yy.append(np.matmul(np.transpose(y), y) / n)

	beta = cp.Variable(dim_x)
	beta_l = cp.Variable(dim_x)
	beta_c = cp.Variable(dim_x)

	xx_bar = sum(cov_xx for cov_xx in covs_xx) / len(covs_xx)
	xy_bar = np.squeeze(sum(cov_xy for cov_xy in covs_xy) / len(covs_xy))

	print(xx_bar)
	print(xy_bar)

	constraints = [beta_l >= beta, beta_l >= -beta]
	for i in range(num_envs):
		constraints += [beta_c >= covs_xx[i] @ beta - np.squeeze(covs_xy[i])]
		constraints += [beta_c >= np.squeeze(covs_xy[i]) - covs_xx[i] @ beta]

	prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(beta, xx_bar) - xy_bar.T @ beta + gamma * (beta_l @ beta_c))) 
	#					constraints)
	prob.solve(solver=cp.SCS)
	return beta.value

