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
		'uniform_t_vis_3_weak',
		'uniform_t_vis_2_weak',
		'uniform_t_vis_1_weak',
		'uniform_diode_vis_3_mid',
		'uniform_diode_vis_2_mid',
		'uniform_diode_vis_1_mid',
		'uniform_t_vis_1_strong',
		'uniform_t_vis_2_strong',
		'uniform_t_vis_3_strong',
		'uniform_t_ir_1_strong',
		'uniform_t_ir_2_strong',
	])


xs, ys = [], []
hd = obs_data[0].head()
pd.options.display.max_columns = None

for e in range(len(obs_data)):
	#x = obs_data[e][['red', 'green', 'blue', 'pol_1', 'pol_2', 'vis_3']].to_numpy()
	x = obs_data[e][['red', 'green', 'blue', 'pol_1', 'pol_2', 'vis_3', 'vis_2', 'vis_1', 'ir_2', 'ir_1', 'current']].to_numpy()
	y = obs_data[e][['ir_3']].to_numpy()
	xs.append(x)
	ys.append(y)

#xs_train, ys_train = [xs[0], np.concatenate([xs[1], xs[1], xs[4], xs[4], xs[2], xs[3]], 0)], [ys[0], np.concatenate([ys[1], ys[1], ys[4], ys[4], ys[2], ys[3]], 0)]
xs_train, ys_train = [xs[0], np.concatenate([xs[1], xs[2], xs[3]], 0)], [ys[0], np.concatenate([ys[1], ys[2], ys[3]], 0)]
xs_test, ys_test = [xs[7], xs[8], xs[9], xs[10], xs[11]], [ys[7], ys[8], ys[9], ys[10], ys[11]]

n_train, n_valid = args.n, args.n * 3 // 10
dim_x = np.shape(xs_train[0])[1]

xs0, ys0 = get_bootstrap_sample(xs_train, ys_train, n_train + n_valid, replace=False)
xs1, ys1, xstt, ystt = standardize(xs0, ys0, xs_test, ys_test)

np.set_printoptions(precision=2)

if args.mode == 1:
	# linear model estimation
	beta = pooled_least_squares(xs1, ys1)
	beta1 = least_squares(xs1[0], ys1[0])
	beta2 = least_squares(xs1[1], ys1[1])
	betastar = pooled_least_squares(xs1, ys1, [0, 1, 2, 3, 4])

	packs = fair_ll_sgd_gumbel_uni(xs1, ys1, hyper_gamma=args.gamma, learning_rate=args.lr, niters_d=args.diter,
								niters=args.niter, batch_size=args.batch_size, init_temp=args.init_temp,
								final_temp=args.final_temp, iter_save=100, log=True)
	mask = packs['gate_rec'][-1] > 0.9
	var_set = (np.arange(np.shape(beta)[0]))[mask].tolist()
	betafair = pooled_least_squares(xs1, ys1, var_set)

	print(f'causal: risk = {linear_eval_worst_test(xstt, ystt, betastar)}, beta = {betastar}')
	print(f'fair: risk = {linear_eval_worst_test(xstt, ystt, betafair)}, beta = {betafair}')
	print(f'pooled: risk = {linear_eval_worst_test(xstt, ystt, beta)}, beta = {beta}')
	print(f'e1: risk = {linear_eval_worst_test(xstt, ystt, beta1)}, beta = {beta1}')
	print(f'e2: risk = {linear_eval_worst_test(xstt, ystt, beta2)}, beta = {beta2}')
elif args.mode == 2:
	xstr, ystr, validx, validy = train_valid_split(xs1, ys1, n_train)
	px = np.concatenate(xstr, 0)
	print(np.matmul(px.T, px) / np.shape(px)[0])
	print(np.matmul(px.T, np.concatenate(ystr, 0)) / np.shape(px)[0])
	eval_data = (validx, validy, xstt, ystt)

	if True:
		mask1 = np.ones((dim_x, ))
		packs1 = fairnn_sgd_gumbel_refit(xstr, ystr, mask1, eval_data, learning_rate=args.lr, niters=args.riter, 
									batch_size=args.batch_size, log=False)
		eval_loss1 = packs1['loss_rec']
		print('ERM Test Error: {}'.format(np.max(eval_loss1[np.argmin(eval_loss1[:, 0]), 1:])))

		mask2 = np.array([1, 1, 1, 1, 1] + [0] * (dim_x - 5))
		packs2 = fairnn_sgd_gumbel_refit(xstr, ystr, mask2, eval_data, learning_rate=args.lr, niters=args.riter, 
									batch_size=args.batch_size, log=False)
		eval_loss2 = packs2['loss_rec']
		print('Causal Test Error: {}'.format(np.max(eval_loss2[np.argmin(eval_loss2[:, 0]), 1:])))

		mask3 = np.array([1, 1, 1] + [0] * (dim_x - 3))
		packs3 = fairnn_sgd_gumbel_refit(xstr, ystr, mask3, eval_data, learning_rate=args.lr, niters=args.riter, 
									batch_size=args.batch_size, log=False)
		eval_loss3 = packs3['loss_rec']
		print('Linear Causal Test Error: {}'.format(np.max(eval_loss3[np.argmin(eval_loss3[:, 0]), 1:])))


	packs3 = fairnn_sgd_gumbel_uni(xstr, ystr, eval_data=eval_data, depth_g=1, width_g=128, depth_f=2, width_f=196, offset=-1,
						hyper_gamma=args.gamma, learning_rate=args.lr, niters=args.niter, niters_d=args.diter, niters_g=1, anneal_rate=0.993, 
						batch_size=args.batch_size, init_temp=args.init_temp, final_temp=args.final_temp, iter_save=100, log=True)
	mask3 = (packs3['gate_rec'][-1] > args.threshold) * 1.0
elif args.mode == 3:

	n_rep = 10
	myvarsel = np.zeros((n_rep, 2, dim_x))
	risk = np.zeros((n_rep, 8, 5))

	for exp_id in range(n_rep):
		start_time = time.time()
		np.random.seed(1000 + exp_id)
		torch.manual_seed(1000 + exp_id)
		xs0, ys0 = get_bootstrap_sample(xs_train, ys_train, n_train + n_valid, replace=False)
		xs1, ys1, xstt, ystt = standardize(xs0, ys0, xs_test, ys_test)
		xstr, ystr, validx, validy = train_valid_split(xs1, ys1, n_train)
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
		print(mask2)
		myvarsel[exp_id, 0, :] = mask2
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
		print(mask5)
		myvarsel[exp_id, 1, :] = mask5
		packs5 = fairnn_sgd_gumbel_refit(xstr, ystr, mask5, eval_data, learning_rate=args.lr, niters=args.riter, 
									batch_size=args.batch_size, log=False)
		eval_loss5 = packs5['loss_rec']
		risk[exp_id, 6, :] = eval_loss5[np.argmin(eval_loss5[:, 0]), 1:]
		print(f'n = {args.n}, exp_id = {exp_id}, FAIR-NN risk = {np.max(risk[exp_id, 6, :])}')

		end_time = time.time()
		print(f'Running Case: exp_id = {exp_id}, secs = {end_time - start_time}s\n')


	np.save(f'lightchamber{args.n}_risk.npy', risk)
	np.save(f'lightchamber{args.n}_var.npy', myvarsel)

elif args.mode == 4:

	xs0, ys0 = get_bootstrap_sample(xs_train, ys_train, n_train + n_valid, replace=False)
	xs1, ys1, xstt, ystt = standardize(xs0, ys0, xs_test, ys_test)

	print(np.mean(xs1[0], 0), np.std(xs1[0], 0))
	print(np.mean(xs1[1], 0), np.std(xs1[1], 0))

	print(np.mean(xstt[2], 0), np.std(xstt[2], 0))

















