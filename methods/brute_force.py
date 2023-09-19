import numpy as np


def fair_ll_brute_force(x_list, y_list, gamma, show_log=True):
	num_envs = len(x_list)
	covs_xx = []
	covs_xy = []
	covs_yy = []
	dim_x = np.shape(x_list[0])[1]

	if show_log:
		print(f'FairLL linear model: brute force search, d = {dim_x}')
	
	for i in range(num_envs):
		x, y = x_list[i], y_list[i]
		n = np.shape(x)[0]
		print(f'Env = {i}, number of samples = {n}')
		covs_xx.append(np.matmul(np.transpose(x), x) / n)
		covs_xy.append(np.matmul(np.transpose(x), y) / n)
		covs_yy.append(np.matmul(np.transpose(y), y) / n)

	null_loss = sum([yy for yy in covs_yy]) / num_envs

	min_loss = null_loss
	min_beta = np.zeros((dim_x, 1))
	min_set = []

	for sel in range(2 ** dim_x):
		var_set = []
		for j in range(dim_x):
			if (sel & (2 ** j)):
				var_set.append(j)
		
		if len(var_set) == 0:
			# case 1: don't select any variable
			cur_loss = null_loss
			cur_beta = np.zeros((dim_x, 1))
		else:
			var_set = np.array(var_set, dtype=np.int)
			A = np.zeros((len(var_set), len(var_set)))
			b = np.zeros((len(var_set), 1))
			c = 0
			for i in range(num_envs):
				xx = covs_xx[i][var_set, :]
				xx = xx[:, var_set]
				xy = covs_xy[i][var_set, :]
				A = A + (1 + gamma) * xx / num_envs
				b = b + (1 + gamma) * xy / num_envs
				c = c + covs_yy[i] / num_envs + \
					gamma * np.matmul(np.matmul(np.transpose(xy), np.linalg.inv(xx)), xy) / num_envs

			cur_beta = np.matmul(np.linalg.inv(A), b)
			cur_loss = 0.5 * np.matmul(np.matmul(np.transpose(cur_beta), A), cur_beta) - \
						np.matmul(np.transpose(cur_beta), b) + 0.5 * c
		#print(f'varset = {var_set}, beta = {np.squeeze(cur_beta)}, loss = {cur_loss}')

		if cur_loss < min_loss:
			min_loss = cur_loss
			min_set = var_set
			min_beta = np.zeros((dim_x, 1))
			for i, idx in enumerate(var_set):
				min_beta[idx] = cur_beta[i]

	print(f'FAIR-LL brute force search [gamma = {gamma}]: var_set = {min_set}, loss = {min_loss}')
	return np.squeeze(min_beta)


def eills_brute_force(features, responses, omegas=None, hyper_gamma=1.0, prior_var=None, usel2=True, verbose=False):
	num_envs = len(features)
	assert num_envs >= 2
	assert num_envs == len(responses)
	# default omega: use uniform weight
	if omegas is None:
		omegas = [(1.0 / num_envs) for i in range(num_envs)]
	assert num_envs == len(omegas)
	assert abs(sum(omegas) - 1) < 1e-9

	samples = np.array(np.zeros(num_envs), dtype=np.int32)
	samples[0], dim = features[0].shape

	for e in range(num_envs):
		assert len(features[e].shape) == 2
		samples[e], cur_dim = features[e].shape
		assert cur_dim == dim
		responses[e] = np.reshape(responses[e], (samples[e], 1))

	# calculate several matrices
	A = [np.zeros((dim, dim)) for i in range(dim + 1)]
	b = [np.zeros((1, dim)) for i in range(dim + 1)]
	c = [0 for i in range(dim + 1)]
	for e in range(num_envs):
		A[0] += omegas[e] / samples[e] * np.matmul(features[e].T, features[e])
		b[0] += 2 * omegas[e] / samples[e] * np.matmul(responses[e].T, features[e])
		c[0] += np.squeeze(omegas[e] / samples[e] * np.matmul(responses[e].T, responses[e]))
		for j in range(dim):
			vec_js = np.matmul(features[e][:, j:j + 1].T, features[e]) / samples[e]
			val_jy = np.matmul(features[e][:, j:j + 1].T, responses[e]) / samples[e]
			A[j + 1] += omegas[e] * np.matmul(vec_js.T, vec_js)
			b[j + 1] += 2 * omegas[e] * val_jy * vec_js
			c[j + 1] += np.squeeze(omegas[e] * val_jy * val_jy)

	min_loss = c[0]
	min_beta = np.zeros((dim, 1))
	min_set = []
	if prior_var is None:
		prior_var = {}
	for sel in range(2 ** dim):
		var_set = []
		for j in range(dim):
			if (sel & (2 ** j)) > 0 or (j in prior_var):
				var_set.append(j)
		if len(var_set) == 0:
			cur_loss = c[0]
			cur_beta = np.zeros((dim, 1))
			loss_l2 = c[0]
			loss_reg = 0.0
		else:
			var_set = np.array(var_set, dtype=np.int32)
			Qr, pr, rr = 0, 0, 0
			for j in var_set:
				tmp_mat = A[j + 1][var_set, :]
				Qr += np.reshape(tmp_mat[:, var_set], (len(var_set), len(var_set)))
				pr += np.reshape(b[j + 1][:, var_set], (1, len(var_set)))
				rr += c[j + 1]
			tmp_mat = A[0][var_set, :]
			Q2 = tmp_mat[:, var_set]
			p2 = b[0][:, var_set]
			r2 = c[0]
			if usel2:
				Q = Qr * hyper_gamma + Q2
				p = pr * hyper_gamma + p2
				r = rr * hyper_gamma + r2
			else:
				Q, p, r = Qr, pr, rr
			cur_beta = np.matmul(np.linalg.inv(Q), 0.5 * p.T)
			cur_loss = np.squeeze(np.matmul(np.matmul(cur_beta.T, Q), cur_beta) - np.matmul(p, cur_beta) + r)
			loss_l2 = np.squeeze(np.matmul(np.matmul(cur_beta.T, Q2), cur_beta) - np.matmul(p2, cur_beta) + r2)
			loss_reg = np.squeeze(np.matmul(np.matmul(cur_beta.T, Qr), cur_beta) - np.matmul(pr, cur_beta) + rr)

		if cur_loss < min_loss:
			if verbose:
				print(f'var_set = {var_set}, loss = {cur_loss}, loss_l2 = {loss_l2}, '
					  f'loss_reg = {loss_reg}, beta = {cur_beta}')
			min_loss = cur_loss
			min_set = var_set
			min_beta = np.zeros((dim, 1))
			for i, idx in enumerate(var_set):
				min_beta[idx] = cur_beta[i]
	print(f'EILLS brute force search [gamma = {hyper_gamma}]: var_Set = {min_set}, loss = {min_loss}')
	return np.squeeze(min_beta)





