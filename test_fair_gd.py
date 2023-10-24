from data.model import linear_SCM, SCM_ex1
from methods.brute_force import fair_ll_brute_force, eills_brute_force
from methods.fair_est import fair_ll_sgd
import numpy as np
#np.random.seed(1151)

models, true_func, true_coeff = SCM_ex1()

xs,  ys = [], []
oracle_var = 0
for i in range(2):
	x, y, _ = models[i].sample(500)
	xs.append(x)
	ys.append(y)
	oracle_var += np.mean(np.square(y - _))
print(oracle_var / 2)

print(f'True parameter = {true_coeff}')

beta4 = pooled_least_squares(xs, ys)
print(f'LSE solution {beta4}')

beta = fair_ll_brute_force(xs, ys, 30)
print(f'FAIR Brute Force solution {beta}')

beta2 = eills_brute_force(xs, ys, hyper_gamma=30)
print(f'EILLS Brute Force solution {beta2}')

beta3 = fair_ll_sgd(xs, ys, hyper_gamma=20)
print(f'FAIR SGD solution {beta3}')

print(f'l2 error (LSE) = {np.sum(np.square(true_coeff - beta4))}')
print(f'l2 error (FAIR) = {np.sum(np.square(true_coeff - beta))}')
print(f'l2 error (EILLS) = {np.sum(np.square(true_coeff - beta2))}')
