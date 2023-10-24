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
			cir_beta, cur_loss = calc_fair_ll_loss(np.array(var_set, dtype=np.int32), covs_xx, covs_xy)
		#print(f'varset = {var_set}, beta = {np.squeeze(cur_beta)}, loss = {cur_loss}')

		if cur_loss < min_loss:
			min_loss = cur_loss
			min_set = var_set
			min_beta = np.zeros((dim_x, 1))
			for i, idx in enumerate(var_set):
				min_beta[idx] = cur_beta[i]

	print(f'FAIR-LL brute force search [gamma = {gamma}]: var_set = {min_set}, loss = {min_loss}')
	return np.squeeze(min_beta)
