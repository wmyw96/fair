from data.model import *
from methods.eilasso import environment_invariant_lasso
import numpy as np

#np.random.seed(123)

dim_z = 80
#np.random.seed(102)
#models, true_func, true_coeff, parent_set = linear_SCM(num_vars=60, num_envs=2, y_index=20, min_child=5, min_parent=5, nonlinear_id=5)
#models = [StructuralCausalModel1(dim_z), StructuralCausalModel2(dim_z)]
#true_coeff = np.array([3, 2, -0.5] + [0] * (dim_z - 4))

models, _, true_coeff = SCM_ex1()

xs, ys = [], []
oracle_var = 0
for i in range(2):
	x, y, _ = models[i].sample(600)
	xs.append(x)
	ys.append(y)
	oracle_var += np.mean(np.square(y - _))
print(oracle_var / 2)

print(f'True parameter = {true_coeff}')

beta1 = environment_invariant_lasso(xs, ys, 10)
print(f'EIL solution {beta1}')