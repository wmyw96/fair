from data.model import *
from methods.brute_force import brute_force
from methods.fair_gumbel import fair_ll_sgd_gumbel
from methods.tools import pooled_least_squares
import numpy as np
#np.random.seed(1151)

dim_z = 60
#models, true_func, true_coeff = SCM_ex1()
#models = [StructuralCausalModel1(dim_z), StructuralCausalModel2(dim_z)]
#true_coeff = np.array([3, 2, -0.5] + [0] * (dim_z - 4))

models, true_func, true_coeff, parent_set, offspring_set = linear_SCM(num_vars=60, num_envs=2, y_index=20, min_child=5, min_parent=5, nonlinear_id=5)

xs,  ys = [], []
oracle_var = 0
for i in range(2):
	x, y, _ = models[i].sample(1000)
	xs.append(x)
	ys.append(y)
	oracle_var += np.mean(np.square(y - _))
print(oracle_var / 2)

print(f'True parameter = {true_coeff}')

beta4 = pooled_least_squares(xs, ys)
print(f'LSE solution {beta4}, L2 error = {np.sum(np.square(beta4 - true_coeff))}')

#beta = brute_force(xs, ys, loss_type='fair', gamma=30)
#print(f'FAIR Brute Force solution {beta}')

#beta2 = brute_force(xs, ys, loss_type='eills', gamma=30)
#print(f'EILLS Brute Force solution {beta2}')

niters = 50000
packs = fair_ll_sgd_gumbel(xs, ys, hyper_gamma=34, niters=niters)
beta = packs['weight']
print(f'FAIR SGD solution {beta}, L2 error = {np.sum(np.square(beta - true_coeff))}')


# visualize gate during training

import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)

gate = packs['gate_rec']
para = packs['weight_rec']
print(np.shape(para))

print(para)
color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#05348b',  # dark blue
	'#9acdc4',  # pain blue
]

var_color = []
#var_color = [2, 2, 2, 1, 1, 1, 0, 0, 0] + [1] * (dim_z - 10)
for i in range(dim_z):
	if i in parent_set:
		var_color.append(2)
	elif i in offspring_set:
		var_color.append(0)
	else:
		var_color.append(1)

plt.figure(figsize=(12, 16))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

it_display = 30000
it_arr = np.arange(it_display)

for i in range(dim_z - 1):
	ax1.plot(it_arr, gate[:it_display, i], color=color_tuple[var_color[i]])
	ax2.plot(it_arr, para[:it_display, i] * gate[:it_display, i], color=color_tuple[var_color[i]])

plt.show()
#plt.savefig('traj.pdf')
#plt.close()