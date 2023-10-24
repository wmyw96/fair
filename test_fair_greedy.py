from data.model import *
from methods.brute_force import greedy_search, brute_force, pooled_least_squares, support_set
from methods.fair_est import fair_ll_sgd
import numpy as np

#np.random.seed(123)

dim_z = 80
#np.random.seed(102)
models, true_func, true_coeff, parent_set = linear_SCM(num_vars=60, num_envs=2, y_index=20, min_child=5, min_parent=5, nonlinear_id=5)
#models = [StructuralCausalModel1(dim_z), StructuralCausalModel2(dim_z)]
#true_coeff = np.array([3, 2, -0.5] + [0] * (dim_z - 4))

xs,  ys = [], []
oracle_var = 0
for i in range(2):
	x, y, _ = models[i].sample(600)
	xs.append(x)
	ys.append(y)
	oracle_var += np.mean(np.square(y - _))
print(oracle_var / 2)

print(f'True parameter = {true_coeff}')

beta4 = pooled_least_squares(xs, ys)
print(f'LSE solution {beta4}')
print(f'l2 error (LSE) = {np.sum(np.square(true_coeff - beta4))}')

beta23 = greedy_search(xs, ys, 0, loss_type='fair', iters=3000)
print(f'L0 LSE Greedy Search solution {beta23}')
print(f'l2 error (L0 LSE), no init = {np.sum(np.square(true_coeff - beta23))}')

beta20 = greedy_search(xs, ys, 30, loss_type='fair', cand_sets=[parent_set], iters=3000)
print(f'FAIR Greedy Search solution {beta20}')
print(f'l2 error (FAIR), with good init = {np.sum(np.square(true_coeff - beta20))}')


beta21 = greedy_search(xs, ys, 30, loss_type='fair', cand_sets=[support_set(beta23)], iters=3000)
print(f'FAIR Greedy Search solution {beta21}')
print(f'l2 error (FAIR), no good init = {np.sum(np.square(true_coeff - beta21))}')


#beta30 = greedy_search(xs, ys, 30, loss_type='eills', cand_sets=[parent_set], iters=100)
#print(f'EILLS Greedy Search solution {beta30}')
#print(f'l2 error (EILLS), with good init = {np.sum(np.square(true_coeff - beta30))}')

#beta31 = greedy_search(xs, ys, 30, loss_type='eills', cand_sets=[], iters=100)
#print(f'EILLS Greedy Search solution {beta31}')
#print(f'l2 error (EILLS), no good init = {np.sum(np.square(true_coeff - beta31))}')
