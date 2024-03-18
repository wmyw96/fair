from data.model import *
from methods.brute_force import greedy_search, brute_force, pooled_least_squares, support_set
from methods.predessors import *
from methods.fair_gumbel_one import fair_ll_sgd_gumbel_uni
import numpy as np
import time
from eills_demo_script.demo_wrapper import *
from utils import get_linear_SCM

TEST_ID = 3

if TEST_ID == 2:
	candidate_n = [100, 300, 700, 1000, 2000]

	num_repeats = 50

	np.random.seed(0)

	methods = [
		eills,
		fair,
		fair_ll_sgd_gumbel_uni,
		lse_s_star,
		oracle_irm,
		oracle_anchor,
		erm
	]

	result = np.zeros((len(candidate_n), num_repeats, len(methods) + 2, 15))

	for (ni, n) in enumerate(candidate_n):
		for t in range(num_repeats):
			start_time = time.time()
			np.random.seed(t)
			#generate random graph with 20 nodes
			models, true_coeff, parent_set, child_set, offspring_set = \
				get_linear_SCM(num_vars=16, num_envs=2, y_index=8, 
								min_child=5, min_parent=5, nonlinear_id=5, 
								bias_greater_than=0.5, log=False)
			
			result[ni, t, 0, :] = true_coeff

			# generate data
			xs, ys = [], []
			for i in range(2):
				x, y, _ = models[i].sample(n)
				xs.append(x)
				ys.append(y)

			for mid, method in enumerate(methods):
				if mid == 2:
					packs = fair_ll_sgd_gumbel_uni(xs, ys, hyper_gamma=36, learning_rate=1e-3, 
												niters=50000, batch_size=64, init_temp=5,
												final_temp=0.1, log=False)
					beta = packs['weight']
					mask = packs['gate_rec'][-1] > 0.9

					# Refit using LS
					full_var = (np.arange(15))
					var_set = full_var[mask].tolist()
					beta3 = broadcast(pooled_least_squares([x[:, var_set] for x in xs], ys), var_set, 15)

					result[ni, t, mid + 1, :] = beta
					result[ni, t, len(methods) + 1, :] = beta3
				else:
					beta = method(xs, ys, true_coeff)

					# restore the estimated coeffs
					result[ni, t, mid + 1, :] = beta
			end_time = time.time()
			print(f'Running Case: n = {n}, t = {t}, secs = {end_time - start_time}s')


	np.save('uni_unit_test_2.npy', result)

if TEST_ID == 3:
	candidate_n = [500, 1000, 2000, 5000, 10000]

	num_repeats = 50

	np.random.seed(0)

	methods = [
		fair_ll_sgd_gumbel_uni,
		lse_s_star,
		lse_s_rd,
		erm
	]

	result = np.zeros((len(candidate_n), num_repeats, len(methods) + 2, 70))

	for (ni, n) in enumerate(candidate_n):
		for t in range(num_repeats):
			start_time = time.time()
			np.random.seed(t)
			#generate random graph with 20 nodes
			models, true_coeff, parent_set, child_set, offspring_set = \
				get_linear_SCM(num_vars=71, num_envs=2, y_index=35, 
								min_child=5, min_parent=5, nonlinear_id=5, 
								bias_greater_than=0.5, log=False)
			
			result[ni, t, 0, :] = true_coeff

			# generate data
			xs, ys = [], []
			for i in range(2):
				x, y, _ = models[i].sample(n)
				xs.append(x)
				ys.append(y)

			for mid, method in enumerate(methods):
				if mid == 0:
					packs = fair_ll_sgd_gumbel_uni(xs, ys, hyper_gamma=36, learning_rate=1e-3, 
												niters=50000, batch_size=64, init_temp=5,
												final_temp=0.1, log=False)
					beta = packs['weight']
					mask = packs['gate_rec'][-1] > 0.9

					# Refit using LS
					full_var = (np.arange(70))
					var_set = full_var[mask].tolist()
					beta3 = broadcast(pooled_least_squares([x[:, var_set] for x in xs], ys), var_set, 70)

					result[ni, t, mid + 1, :] = beta
					result[ni, t, len(methods) + 1, :] = beta3
				else:
					beta = method(xs, ys, true_coeff)

					result[ni, t, mid + 1, :] = beta
				
				print(f'method {mid}, l2 error = {np.sum(np.square(true_coeff - beta))}')
			end_time = time.time()
			print(f'Running Case: n = {n}, t = {t}, secs = {end_time - start_time}s')


	np.save('uni_unit_test_3.npy', result)
