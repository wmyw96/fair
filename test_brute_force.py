from data.model import linear_SCM
from methods.brute_force import fair_ll_brute_force, eills_brute_force
import numpy as np
#np.random.seed(1001)

models, true_func, true_coeff = linear_SCM(15, 2)
true_func, true_coeff = true_func[:-1], true_coeff[:-1]

xs,  ys = [], []
oracle_var = 0
for i in range(2):
	x, y, _ = models[i].sample(100)
	xs.append(x)
	ys.append(y)
	oracle_var += np.mean(np.square(y - _))
print(oracle_var / 2)
print(true_coeff)
beta = fair_ll_brute_force(xs, ys, 20)
print(beta)

beta2 = eills_brute_force(xs, ys, hyper_gamma=20)
print(beta2)

print(f'l2 error (FAIR) = {np.sum(np.square(true_coeff - beta))}')
print(f'l2 error (EILLS) = {np.sum(np.square(true_coeff - beta2))}')
