from data.model import *
from methods.brute_force import brute_force
from methods.fair_gumbel_one import *
from methods.tools import pooled_least_squares
from utils import *
import numpy as np
import os
import argparse
import time
import pandas as pd

colors = [
		'#ae1908',  # red
		'#ec813b',  # orange
		'#05348b',  # dark blue
		'#9acdc4',  # pain blue
		'#6bb392',  # green
		'#e5a84b',   # yellow
		'#743096',   #purple
]

def get_bootstrap_sample(xs, ys, n, replace=True):
	new_xs, new_ys = [], []
	for i in range(len(xs)):
		population_n = len(xs[i])
		index = np.random.choice(population_n, n, replace=replace)
		new_xs.append(xs[i][index, :])
		new_ys.append(ys[i][index, :])
	return new_xs, new_ys

def standardize(xs, ys, xs_test, ys_test):
	x_mean = np.mean(np.concatenate(xs, 0), 0, keepdims=True)
	y_mean = np.mean(np.concatenate(ys, 0))
	x_std = np.std(np.concatenate(xs, 0), 0, keepdims=True)
	y_std = np.std(np.concatenate(ys, 0), 0)
	xs_new = [(x - x_mean) / x_std for x in xs]
	xs_test_new = [(x - x_mean) / x_std for x in xs_test]
	ys_new = [(y - y_mean) / y_std for y in ys]
	ys_test_new = [(y - y_mean) / y_std for y in ys_test]
	return xs_new, ys_new, xs_test_new, ys_test_new


def linear_eval_worst_test(xs, ys, beta):
	risk = []
	beta_vec = np.reshape(beta, (np.shape(xs[0])[1], 1))
	for e in range(len(xs)):
		x, y = xs[e], ys[e]
		y_hat = np.matmul(x, beta_vec)
		risk.append(np.mean(np.square(y - y_hat)))
	return np.array(risk)


def train_valid_split(xs, ys, n_train):
	valid_xs = [x[n_train:, :] for x in xs]
	valid_ys = [y[n_train:, :] for y in ys]
	valid_x = np.concatenate(valid_xs, 0)
	valid_y = np.concatenate(valid_ys, 0)
	return [x[:n_train, :] for x in xs], [y[:n_train, :] for y in ys], valid_x, valid_y


def augument_xye(xstr, ystr, n_train):
	aug_x = np.concatenate([xstr[0], xstr[1]], 0)
	aug_y = np.concatenate([ystr[0], ystr[1]], 0)
	aug_e = np.concatenate([np.zeros((n_train, 1)), np.ones((n_train, 1))], 0)
	return np.concatenate([aug_x, aug_y, aug_e], 1)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", type=int, default=1234)
parser.add_argument("--n", help="number of samples", type=int, default=1000)
parser.add_argument("--batch_size", help="batch size", type=int, default=36)
parser.add_argument("--dim_x", help="number of explanatory vars", type=int, default=60)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
parser.add_argument('--init_temp', help='initial temperature', type=float, default=5)
parser.add_argument('--final_temp', help='final temperature', type=float, default=0.1)
parser.add_argument('--gamma', help='hyper parameter gamma', type=float, default=36)
parser.add_argument("--log", help="show log", type=bool, default=True)
parser.add_argument("--diter", help="discriminator iters", type=int, default=3)
parser.add_argument("--niter", help="number of interations", type=int, default=50000)
parser.add_argument("--riter", help="number of interations", type=int, default=20000)
parser.add_argument("--threshold", help="threshold", type=float, default=0.9)
parser.add_argument("--nrep", help="n replications in mode 3", type=int, default=30)
parser.add_argument("--mode", help="mode", type=int, default=1)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

def load_data_pd(dir_prefix, envs_name):
	data = []
	for ename in envs_name:
		dir_name = dir_prefix + ename + '.csv'
		data.append(pd.read_csv(dir_name))
	return data


obs_data = load_data_pd(
	dir_prefix='dataset/lightchamber/lt_interventions_standard_v1/',
	envs_name=[
		'uniform_reference',
		'uniform_t_ir_3_weak',
		'uniform_t_vis_2_weak',
		'uniform_t_ir_2_weak',
		'uniform_t_ir_1_weak',
		'uniform_t_ir_1_strong',
		'uniform_t_vis_2_strong',
		'uniform_t_ir_3_strong',
		'uniform_t_ir_2_strong',
	])


xs, ys = [], []
hd = obs_data[0].head()
pd.options.display.max_columns = None

for e in range(len(obs_data)):
	#x = obs_data[e][['red', 'green', 'blue', 'pol_1', 'pol_2', 'vis_3']].to_numpy()
	x = obs_data[e][['vis_1', 'ir_1', 'vis_2', 'ir_3', 'ir_2', 'current', 'angle_1', 'angle_2']].to_numpy()
	y = obs_data[e][['vis_3']].to_numpy()
	xs.append(x)
	ys.append(y)

#xs_train, ys_train = [xs[0], np.concatenate([xs[1], xs[1], xs[4], xs[4], xs[2], xs[3]], 0)], [ys[0], np.concatenate([ys[1], ys[1], ys[4], ys[4], ys[2], ys[3]], 0)]
xs_train, ys_train = [xs[0], np.concatenate([xs[1], xs[2], xs[3]], 0)], \
						[ys[0], np.concatenate([ys[1], ys[2], ys[3]], 0)]
xs_test, ys_test = [xs[5], xs[6], xs[7], xs[8]], [ys[5], ys[6], ys[7], ys[8]]

n_train, n_valid = args.n, args.n * 3 // 10
dim_x = np.shape(xs_train[0])[1]

xs0, ys0 = get_bootstrap_sample(xs_train, ys_train, n_train + n_valid, replace=False)
xs1, ys1, xstt, ystt = standardize(xs0, ys0, xs_test, ys_test)

np.set_printoptions(precision=3)

if args.mode == 1:
	# linear model estimation
	beta = pooled_least_squares(xs1, ys1)
	beta1 = least_squares(xs1[0], ys1[0])
	beta2 = least_squares(xs1[1], ys1[1])
	betastar = pooled_least_squares(xs1, ys1, [0, 1, 5, 6, 7])

	print(f'causal: risk = {linear_eval_worst_test(xstt, ystt, betastar)}, beta = {betastar}')
	print(f'pooled: risk = {linear_eval_worst_test(xstt, ystt, beta)}, beta = {beta}')
	print(f'e1: risk = {linear_eval_worst_test(xstt, ystt, beta1)}, beta = {beta1}')
	print(f'e2: risk = {linear_eval_worst_test(xstt, ystt, beta2)}, beta = {beta2}')

	packs = fair_ll_sgd_gumbel_uni(xs1, ys1, hyper_gamma=args.gamma, learning_rate=args.lr, niters_d=args.diter,
								niters=args.niter, batch_size=args.batch_size, init_temp=args.init_temp,
								final_temp=args.final_temp, iter_save=100, log=True)
	mask = packs['gate_rec'][-1] > 0.9
	var_set = (np.arange(np.shape(beta)[0]))[mask].tolist()
	print(var_set)
	betafair = pooled_least_squares(xs1, ys1, var_set)
	print(f'fair: risk = {linear_eval_worst_test(xstt, ystt, betafair)}, beta = {betafair}')
elif args.mode == 2:
	xstr, ystr, validx, validy = train_valid_split(xs1, ys1, n_train)
	px = np.concatenate(xstr, 0)
	print(np.matmul(px.T, px) / np.shape(px)[0])
	print(np.matmul(px.T, np.concatenate(ystr, 0)) / np.shape(px)[0])

	for e in range(4):
		print(np.mean(xstt[e][:, 1+e], 0))
		xstt[e][:, 1+e] -= np.mean(xstt[e][:, 1+e])
		#xstt[e][:, 2+e] /= np.std(xstt[e][:, 2+e])
		
	eval_data = (validx, validy, xstt, ystt)

	if True:
		mask1 = np.ones((dim_x, ))
		packs1 = fairnn_sgd_gumbel_refit(xstr, ystr, mask1, eval_data, learning_rate=args.lr, niters=args.riter, 
									batch_size=args.batch_size, log=False)
		eval_loss1 = packs1['loss_rec']
		print('ERM Test Error: {}'.format((eval_loss1[np.argmin(eval_loss1[:, 0]), 1:])))

		mask2 = np.array([1, 1, 0, 0, 0, 1, 1, 1])
		packs2 = fairnn_sgd_gumbel_refit(xstr, ystr, mask2, eval_data, learning_rate=args.lr, niters=args.riter, 
									batch_size=args.batch_size, log=False)
		eval_loss2 = packs2['loss_rec']
		print('Causal Test Error: {}'.format((eval_loss2[np.argmin(eval_loss2[:, 0]), 1:])))

	packs3 = fairnn_sgd_gumbel_uni(xstr, ystr, eval_data=eval_data, depth_g=1, width_g=128, depth_f=2, width_f=196, offset=-1,
						hyper_gamma=args.gamma, learning_rate=args.lr, niters=args.niter, niters_d=args.diter, niters_g=1, anneal_rate=0.993, 
						batch_size=args.batch_size, init_temp=args.init_temp, final_temp=args.final_temp, iter_save=100, log=True)
	mask3 = (packs3['gate_rec'][-1] > args.threshold) * 1.0
	print(mask3)
elif args.mode == 3:

	n_rep = args.nrep
	myvarsel = np.zeros((n_rep, 4, dim_x))
	risk = np.zeros((n_rep, 8, 5))

	for exp_id in range(n_rep):
		start_time = time.time()
		np.random.seed(1000 + exp_id)
		torch.manual_seed(1000 + exp_id)
		xs0, ys0 = get_bootstrap_sample(xs_train, ys_train, n_train + n_valid, replace=False)
		xs1, ys1, xstt, ystt = standardize(xs0, ys0, xs_test, ys_test)
		xstr, ystr, validx, validy = train_valid_split(xs1, ys1, n_train)

		for e in range(5):
			xstt[e][:, 0+e] -= np.mean(xstt[e][:, 0+e])
		eval_data = (validx, validy, xstt, ystt)

		# linear erm
		beta = pooled_least_squares(xstr, ystr)
		risk[exp_id, 0, :] = linear_eval_worst_test(xstt, ystt, beta)
		print(f'n = {args.n}, exp_id = {exp_id}, ERM risk = {np.max(risk[exp_id, 0, :])}')

		# linear oracle
		beta1 = pooled_least_squares(xstr, ystr, [0, 1, 2, 3, 4])
		risk[exp_id, 1, :] = linear_eval_worst_test(xstt, ystt, beta1)
		print(f'n = {args.n}, exp_id = {exp_id}, Oracle-Linear risk = {np.max(risk[exp_id, 1, :])}')

		# nonlinear erm
		mask3 = np.ones((dim_x, ))
		packs3 = fairnn_sgd_gumbel_refit(xstr, ystr, mask3, eval_data, learning_rate=args.lr, niters=args.riter, 
									batch_size=args.batch_size, log=False)
		eval_loss3 = packs3['loss_rec']
		risk[exp_id, 4, :] = eval_loss3[np.argmin(eval_loss3[:, 0]), 1:]
		print(f'n = {args.n}, exp_id = {exp_id}, ERM risk = {np.max(risk[exp_id, 4, :])}')

		# nonlinear oracle
		mask6 = np.array([1] * 5 + [0] * (dim_x - 5))
		packs6 = fairnn_sgd_gumbel_refit(xstr, ystr, mask6, eval_data, learning_rate=args.lr, niters=args.riter, 
									batch_size=args.batch_size, log=False)
		eval_loss6 = packs6['loss_rec']
		risk[exp_id, 7, :] = eval_loss6[np.argmin(eval_loss6[:, 0]), 1:]
		print(f'n = {args.n}, exp_id = {exp_id}, Oracle-Nonlinear risk = {np.max(risk[exp_id, 7, :])}')


		# fair-linear
		packs1 = fair_ll_sgd_gumbel_uni(xstr, ystr, hyper_gamma=args.gamma, learning_rate=args.lr, niters_d=args.diter,
									niters=args.niter, batch_size=args.batch_size, init_temp=args.init_temp,
									final_temp=args.final_temp, iter_save=100, log=False)
		beta2 = packs1['weight']
		risk[exp_id, 2, :] = linear_eval_worst_test(xstt, ystt, beta2)
		print(f'n = {args.n}, exp_id = {exp_id}, FAIR-Linear risk = {np.max(risk[exp_id, 2, :])}')

		# fair-linear-nn-refit
		mask2 = packs1['gate_rec'][-1] > 0.9
		myvarsel[exp_id, 0, :] = mask2
		print(f'n = {args.n}, exp_id = {exp_id}, FAIR-Linear Var Selection = {myvarsel[exp_id, 0, :]}')

		packs2 = fairnn_sgd_gumbel_refit(xstr, ystr, mask2, eval_data, learning_rate=args.lr, niters=args.riter, 
									batch_size=args.batch_size, log=False)
		eval_loss2 = packs2['loss_rec']
		risk[exp_id, 3, :] = eval_loss2[np.argmin(eval_loss2[:, 0]), 1:]
		print(f'n = {args.n}, exp_id = {exp_id}, FAIR-Linear-NN-Refit risk = {np.max(risk[exp_id, 3, :])}')

		# fair-nn
		packs4 = fairnn_sgd_gumbel_uni(xstr, ystr, eval_data=eval_data, depth_g=1, width_g=128, depth_f=2, width_f=196, offset=-1,
										hyper_gamma=args.gamma, learning_rate=args.lr, niters=args.niter, niters_d=args.diter, niters_g=1, anneal_rate=0.993, 
										batch_size=args.batch_size, init_temp=args.init_temp, final_temp=args.final_temp, iter_save=100, log=False)
		eval_loss4 = packs4['loss_rec']
		risk[exp_id, 5, :] = eval_loss4[-1, 1:]
		print(f'n = {args.n}, exp_id = {exp_id}, FAIR-NN risk = {np.max(risk[exp_id, 5, :])}')

		# fair-nn refit
		mask5 = packs4['gate_rec'][-1] > 0.9
		myvarsel[exp_id, 1, :] = mask5
		print(f'n = {args.n}, exp_id = {exp_id}, FAIR-NN Var Selection = {myvarsel[exp_id, 1, :]}')

		packs5 = fairnn_sgd_gumbel_refit(xstr, ystr, mask5, eval_data, learning_rate=args.lr, niters=args.riter, 
									batch_size=args.batch_size, log=False)
		eval_loss5 = packs5['loss_rec']
		risk[exp_id, 6, :] = eval_loss5[np.argmin(eval_loss5[:, 0]), 1:]
		print(f'n = {args.n}, exp_id = {exp_id}, FAIR-NN-Refit risk = {np.max(risk[exp_id, 6, :])}')

		# call R script
		#aug_data = augument_xye(xstr, ystr, n_train)

		#subprocess.call("/usr/bin/Rscript nonlinearicp.r", shell=True)
		end_time = time.time()
		print(f'Running Case: exp_id = {exp_id}, secs = {end_time - start_time}s\n')


	np.save(f'lightchamber{args.n}_risk.npy', risk)
	np.save(f'lightchamber{args.n}_var.npy', myvarsel)

elif args.mode == 4:
	n_rep = args.nrep
	myvarsel = np.zeros((n_rep, 4, dim_x))
	risk = np.zeros((n_rep, 8, 5))

	import os
	if not os.path.exists('chamber_tmp'):
		os.mkdir('chamber_tmp/')

	for exp_id in range(n_rep):
		start_time = time.time()
		np.random.seed(1000 + exp_id)
		torch.manual_seed(1000 + exp_id)
		xs0, ys0 = get_bootstrap_sample(xs_train, ys_train, n_train + n_valid, replace=False)
		xs1, ys1, xstt, ystt = standardize(xs0, ys0, xs_test, ys_test)
		xstr, ystr, validx, validy = train_valid_split(xs1, ys1, n_train)

		aug_data = augument_xye(xstr, ystr, n_train)
		np.savetxt(f"chamber_tmp/data_tmp_{exp_id}.csv", aug_data, delimiter=",")

elif args.mode == 5:
	var_sel = np.load(f'lightchamber{args.n}_var.npy')

	matrix = np.mean(var_sel, 0)
	#print(result)

	for e in range(args.nrep):
		matrix[2:, :] += np.genfromtxt(f'chamber_tmp/result_tmp_{e}.csv', delimiter=',')

	matrix[2:, :] /= args.nrep


	import matplotlib.pyplot as plt
	from matplotlib import rc

	plt.rcParams["font.family"] = "Times New Roman"
	plt.rc('font', size=12)
	rc('text', usetex=True)

	# Define color maps for each column
	colormaps = [colors[2]] * 5 + [colors[3]] * 2 + [colors[0]] + [colors[3]] * 3

	fig, ax = plt.subplots()

	names = [r'$R$', r'$G$', r'$B$', r'$\theta_1$', r'$\theta_2$', r'$\tilde{V}_1$', r'$\tilde{V}_2$', r'$\tilde{V}_3$', r'$\tilde{I}_1$', r'$\tilde{I}_2$', r'$\tilde{C}$']
	for j in range(matrix.shape[1]):
		ax.text(j * 2 + 1, -1, str(names[j]), va='center', ha='center', color=colormaps[j])
	# Loop over data dimensions and create color mappings and text annotations
	for j in range(matrix.shape[1]):
		for i in range(matrix.shape[0]):
			value = round(matrix[i, j], 2)
			ax.add_patch(plt.Rectangle((j * 2, i * 2), 2, 2, facecolor=colormaps[j], alpha=0.5 * value, edgecolor='none'))
			ax.text(j * 2 + 1, i * 2 + 1, str(value), va='center', ha='center', color=colormaps[j])
	ax.text(-4, 1, 'FAIR-Linear', va='center', ha='center')
	ax.text(-4, 3, 'FAIR-NN', va='center', ha='center')
	ax.text(-4, 5, 'ForestVarSel', va='center', ha='center')
	ax.text(-4, 7, 'NonlinearICP', va='center', ha='center')


	# Set the limits and aspect of the plot
	ax.set_xlim(-5, matrix.shape[1] * 2)
	ax.set_ylim(-2, matrix.shape[0] * 2)
	ax.set_aspect('equal')

	# Hide the axes
	ax.axis('off')

	# Display the plot
	plt.gca().invert_yaxis()
	plt.show()

elif args.mode == 6:
	risk = np.load(f'lightchamber{args.n}_risk.npy')
	risk = np.mean(risk, 0)
	print(risk)

	risk = np.concatenate([risk[:,2:], risk[:,:2]], 1)

	import matplotlib.pyplot as plt
	from matplotlib import rc

	plt.rc('font', size=15)
	plt.rcParams["font.family"] = "Times New Roman"
	rc('text', usetex=True)

	# create spider graph background 
	# number of variable
	categories=[r'$\tilde{V}_3$', r'$\tilde{I}_1$', r'$\tilde{I}_2$', r'$\tilde{V}_1$', r'$\tilde{V}_2$']
	N = len(categories)

	# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
	angles = [n / float(N) * 2 * np.pi for n in range(N)]
	angles += angles[:1]
 
	# Initialise the spider plot
	ax = plt.subplot(111, polar=True)
 
	ax.set_theta_offset(np.pi / 2)
	ax.set_theta_direction(-1)
 
	plt.xticks(angles[:-1], categories)

 	# Draw ylabels
	ax.set_rlabel_position(0)

	values = (1-risk[7, :]).tolist()
	values += values[:1]
	# oracle
	ax.plot(angles, values, colors[0], linewidth=1, linestyle='solid')
	ax.fill(angles, values, alpha=0.25, label='Oracle-NN', facecolor="none", hatch='++', edgecolor=colors[0])
 
	values = (1-risk[1, :]).tolist()
	values += values[:1]
	# oracle
	ax.plot(angles, values, colors[1], linewidth=1, linestyle='solid') 
	ax.fill(angles, values, colors[1], alpha=0.25, label='Oracle-Linear')

	values = (1-risk[6, :]).tolist()
	values += values[:1]
	ax.plot(angles, values, colors[2], linewidth=1, linestyle='dashed')
	ax.fill(angles, values, colors[2], alpha=0.25, label='FAIR-NN-RF')

	#values = (1-risk[5, :]).tolist()
	#values += values[:1]
	#ax.plot(angles, values, colors[6], linewidth=1, linestyle='solid', label="FAIR-NN-GB")


	values = (np.maximum(1-risk[4, :], 0.52)).tolist()
	values += values[:1]
	ax.plot(angles, values, colors[4], linewidth=1, linestyle='solid')
	ax.fill(angles, values, colors[4], alpha=0.25, label="PoolLS-NN")

	plt.yticks([0.71, 0.87, 0.91, 0.95], ['0.71', '0.87', '0.91', '0.95'], color="gray", size=12)
	plt.ylim(0.67,1)

	#values = (1-risk[0, :]).tolist()
	#values += values[:1]
	#ax.plot(angles, values, colors[5], linewidth=1, linestyle='solid', label="PoolLS-Linear")

	plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15))	
	plt.show()

