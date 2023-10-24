from data.model import *
from methods.brute_force import greedy_search, brute_force, pooled_least_squares, support_set
from methods.predessors import *
import numpy as np

##############################################
#
#              Utility Functions
#
##############################################


def broadcast(beta_restricted, var_inds, p):
	beta_broadcast = np.zeros(p)
	if len(var_inds) == 1:
		beta_broadcast[var_inds[0]] = beta_restricted
		return beta_broadcast
	for i, ind in enumerate(var_inds):
		beta_broadcast[ind] = beta_restricted[i]
	return beta_broadcast


dim_x = 12


##############################################
#
#                Methods
#
##############################################


def oracle_irm(x_list, y_list, true_para):
	data_list = []
	for i in range(len(x_list)):
		data_list.append((torch.tensor(x_list[i]).float(), torch.tensor(y_list[i]).float()))

	error_min = 1e9
	beta = 0
	for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
		model = InvariantRiskMinimization(data_list, args={'n_iterations': 100000, 'lr': 1e-3, 'verbose': False, 'reg': reg})
		cand = np.squeeze(model.solution().detach().numpy())
		error = np.sum(np.square(cand - true_para))
		if error < error_min:
			error_min = error
			beta = cand

	return beta


def erm(x_list, y_list, true_para=None):
	return pooled_least_squares(x_list, y_list)


def oracle_icp(x_list, y_list, true_para):
	data_list = []
	for i in range(len(x_list)):
		data_list.append((torch.tensor(x_list[i]).float(), torch.tensor(y_list[i]).float()))

	error_min = 1e9
	beta = 0
	for alpha in [0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]:
		model = InvariantCausalPrediction(data_list, args={'alpha': alpha, "verbose": False})
		cand = np.squeeze(model.solution().numpy())
		error = np.sum(np.square(cand - true_para))
		if error < error_min:
			error_min = error
			beta = cand

	return beta


def oracle_anchor(x_list, y_list, true_para):
	xs, ys, anchors = [], [], []
	for i in range(len(x_list)):
		xs.append(x_list[i])
		ys.append(y_list[i])
		onehot = np.zeros(len(x_list)-1)
		if i + 1 < len(x_list):
			onehot[i] = 1
		anchors.append([onehot] * np.shape(x_list[i])[0])
	
	X, y, A = np.concatenate(xs, 0), np.squeeze(np.concatenate(ys, 0)), np.concatenate(anchors, 0)
	error_min = 1e9
	beta = 0

	for reg in [0, 1, 2, 4, 8, 10, 15, 20, 30, 40, 60, 80, 90, 100, 150, 200, 500, 1000, 5000, 10000]:
		model = AnchorRegression(lamb=reg)
		model.fit(X, y, A)
		cand = np.squeeze(model.coef_)
		error = np.sum(np.square(cand - true_para))
		if error < error_min:
			error_min = error
			beta = cand

	return beta


def eills(x_list, y_list, true_para=None):
	return brute_force(x_list, y_list, 20, loss_type='eills')

def fair(x_list, y_list, true_para=None):
	return brute_force(x_list, y_list, 20, loss_type='fair')

def lse_s_star(x_list, y_list, true_para=None):
	var_set = [0, 1, 2]
	return broadcast(pooled_least_squares([x[:, var_set] for x in x_list], y_list), var_set, dim_x)


def lse_gc(x_list, y_list, true_para=None):
	var_set = [0, 1, 2, 3, 4, 5, 9, 10, 11]
	return broadcast(pooled_least_squares([x[:, var_set] for x in x_list], y_list), var_set, dim_x)


def eills_refit(x_list, y_list, true_para=None):
	eills_sol = brute_force(x_list, y_list, 20, loss_type='eills')
	var_set = []
	for i in range(np.shape(eills_sol)[0]):
		if np.abs(eills_sol[i]) > 1e-9:
			var_set.append(i)
	return broadcast(pooled_least_squares([x[:, var_set] for x in x_list], y_list), var_set, dim_x)


##############################################
#
#                Tests
#
##############################################


dim_z = dim_x + 1
models = [StructuralCausalModel1(dim_z), StructuralCausalModel2(dim_z)]
true_coeff = np.array([3, 2, -0.5] + [0] * (dim_z - 4))

candidate_n = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
set_s_star = [0, 1, 2]
set_g = [6, 7, 8]
set_lse = [6, 7, 8, 9]

sets_interested = [
	set_s_star,
	set_g,
	set_lse
]

num_repeats = 50

np.random.seed(0)

methods = [
	eills,
	fair,
	eills_refit,
	lse_s_star,
	lse_gc,
	oracle_icp,
	oracle_irm,
	oracle_anchor,
	erm
]

threshold = [
	1e-5,
	1e-5,
	1e-5,
	1e-5,
	1e-5,
	1e-5,
	1e-1,
	1e-1,
	1e-1,
]

result = np.zeros((len(candidate_n), num_repeats, len(methods), 7))

for (ni, n) in enumerate(candidate_n):
	for t in range(num_repeats):
		print(f'Running Case: n = {n}, t = {t}')
		# generate data
		xs, ys = [], []
		oracle_var = 0
		for i in range(2):
			x, y, _ = models[i].sample(n)
			xs.append(x)
			ys.append(y)

		for mid, method in enumerate(methods):
			beta = method(xs, ys, true_coeff)
			
			# l2 error
			result[ni, t, mid, 0] = np.sum(np.square(beta - true_coeff))       

			for sid in range(3):
				# l2 error in this set
				set_it = sets_interested[sid]
				result[ni, t, mid, 2 * sid + 1] = np.sum(np.square(beta[set_it] - true_coeff[set_it]))
				result[ni, t, mid, 2 * sid + 2] = np.sum(np.abs(beta[set_it]) > threshold[mid])

np.save('eills_demo.npy', result)


