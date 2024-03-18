import numpy as np
import torch
import torch.optim as optim 
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
from data.utils import MultiEnvDataset
from methods.modules import *


def fair_ll_sgd_gumbel_uni(features, responses, hyper_gamma=10, learning_rate=1e-3, niters=50000, niters_d=2, niters_g=1, 
						batch_size=32, mask=None, init_temp=0.5, final_temp=0.05, iter_save=100, log=False):
	'''
		Implementation of FAIR-LL estimator with gumbel discrete approximation

		Parameter
		----------
		features : list 
			list of numpy matrices with shape (n_k, p) representing the explanatory variables
		responses : list
			list of numpy matrices with shape (n_k, 1) representing the response variable
		hyper_gamma : float
			hyper-parameter gamma control the degree of invariance
		learning_rate : float
			learning rate for stochastic gradient descent
		niters : int
			number of outer iterations
		niters_d : int
			number of inner iterations for discriminator
		niters_g : int
			number of inner iterations for generator
		batch_size : int
			batch_size for stochastic gradient descent
		init_temp : float
			initial temperature for gumbel approximation
		final_temp: float
			final temperature for gumbel approximation
		log : bool
			whether to show logs during training

		Returns
		----------
		a dict collecting things of interests
	'''
	num_envs = len(features)
	dim_x = np.shape(features[0])[1]

	if log:
		print(f'================================================================================')
		print(f'================================================================================')
		print(f'==')
		print(f'==  FAIR Linear/Linear Model Gumbel: num of envs = {num_envs}, x dim = {dim_x}')
		print(f'==')
		print(f'================================================================================')
		print(f'================================================================================')

	model = FairLinear(dim_x, num_envs)
	model_var = GumbelGate(dim_x, init_offset=-3, device='cpu')
	optimizer_var = optim.Adam(model_var.parameters(), lr=learning_rate)

	optimizer_g = optim.Adam(model.params_g(), lr=learning_rate)
	optimizer_f = optim.Adam(model.params_f(), lr=learning_rate)

	# construct dataset from numpy array
	dataset = MultiEnvDataset(features, responses)
	gate_rec = []
	weight_rec = []
	loss_rec = []
	# start training
	if log:
		it_gen = tqdm(range(niters))
	else:
		it_gen = range(niters)

	anneal_rate = (final_temp / init_temp)
	for it in it_gen:
		# calculate the temperature
		tau = max(final_temp, init_temp * (anneal_rate ** ((it + 0.0) / niters)))
		if it % 10000 == 0 and log:
			print(f'annealed tau: {tau}')
		tau_logits = 1

		# train the discriminator
		for i in range(niters_d):
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()
			optimizer_g.zero_grad()
			xs, ys = dataset.next_batch(batch_size)
			gate = model_var.generate_mask((tau_logits, tau)).detach()

			out_gs, out_fs = model([gate * x for x in xs])
			loss_de = - sum([torch.mean((ys[e] - out_gs[e].detach()) * out_fs[e] - 0.5 * out_fs[e] * out_fs[e]) for e in range(num_envs)])
			loss_de.backward()
			optimizer_f.step()

		my_loss = np.zeros((niters_g, 2))

		# train the generator
		for i in range(niters_g):
			optimizer_g.zero_grad()
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()

			xs, ys = dataset.next_batch(batch_size)
			gate = model_var.generate_mask((tau_logits, tau))
			out_gs, out_fs = model([gate * x for x in xs])

			loss_r = 0.5 * sum([torch.mean((out_gs[e] - ys[e]) ** 2) for e in range(num_envs)])
			loss_j = sum([torch.mean((ys[e] - out_gs[e]) * out_fs[e] - 0.5 * out_fs[e] * out_fs[e]) for e in range(num_envs)])
			loss = loss_r + hyper_gamma * loss_j
			loss.backward()
			my_loss[i, 0], my_loss[i, 1] = loss_r.item(), loss_j.item()

			optimizer_g.step()
			optimizer_var.step()

		# save the weight/logits for linear model
		if it % iter_save == 0:
			with torch.no_grad():
				weight = model.g.weight.detach().cpu()
				logits = model_var.get_logits_numpy()
				gate_rec.append(sigmoid(logits / tau))
				weight_rec.append(np.squeeze(weight.numpy() + 0.0))
			#print(logits, np.squeeze(weight.numpy() + 0.0))
			loss_rec.append(np.mean(my_loss, 0))



	ret = {'weight': weight_rec[-1] * sigmoid(logits / tau_logits),
			'weight_rec': np.array(weight_rec),
			'gate_rec': np.array(gate_rec),
			'model': model,
			'fair_var': model_var,
			'loss_rec': np.array(loss_rec)}

	return ret



def fairnn_sgd_gumbel_uni(features, responses, depth_g=2, width_g=32, depth_f=2, width_f=64,
						hyper_gamma=10, learning_rate=1e-3, niters=50000, niters_d=5, niters_g=1, 
						batch_size=32, mask=None, init_temp=0.5, final_temp=0.05, iter_save=100, log=False):
	'''
		Implementation of FAIR-LL estimator with gumbel discrete approximation

		Parameter
		----------
		features : list 
			list of numpy matrices with shape (n_k, p) representing the explanatory variables
		responses : list
			list of numpy matrices with shape (n_k, 1) representing the response variable
		hyper_gamma : float
			hyper-parameter gamma control the degree of invariance
		learning_rate : float
			learning rate for stochastic gradient descent
		niters : int
			number of outer iterations
		niters_d : int
			number of inner iterations for discriminator
		niters_g : int
			number of inner iterations for generator
		batch_size : int
			batch_size for stochastic gradient descent
		init_temp : float
			initial temperature for gumbel approximation
		final_temp: float
			final temperature for gumbel approximation
		log : bool
			whether to show logs during training

		Returns
		----------
		a dict collecting things of interests
	'''
	num_envs = len(features)
	dim_x = np.shape(features[0])[1]

	if log:
		print(f'================================================================================')
		print(f'================================================================================')
		print(f'==')
		print(f'==  FAIR NN Model Gumbel: num of envs = {num_envs}, x dim = {dim_x}')
		print(f'==')
		print(f'================================================================================')
		print(f'================================================================================')

	model = FairNN(dim_x, depth_g, width_g, depth_f, width_f, num_envs)
	model_var = GumbelGate(dim_x, init_offset=-3, device='cpu')
	optimizer_var = optim.Adam(model_var.parameters(), lr=learning_rate)

	optimizer_g = optim.Adam(model.params_g(log), lr=learning_rate, weight_decay=1e-3)
	optimizer_f = optim.Adam(model.params_f(log), lr=learning_rate, weight_decay=1e-3)

	# construct dataset from numpy array
	dataset = MultiEnvDataset(features, responses)
	gate_rec = []
	weight_rec = []
	loss_rec = []
	# start training
	if log:
		it_gen = tqdm(range(niters))
	else:
		it_gen = range(niters)

	anneal_rate = (final_temp / init_temp)
	for it in it_gen:
		# calculate the temperature
		tau = max(final_temp, init_temp * (anneal_rate ** ((it + 0.0) / niters)))
		if it % 10000 == 0 and log:
			print(f'annealed tau: {tau}')
		tau_logits = 1

		# train the discriminator
		for i in range(niters_d):
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()
			optimizer_g.zero_grad()
			xs, ys = dataset.next_batch(batch_size)
			gate = model_var.generate_mask((tau_logits, tau)).detach()

			out_gs, out_fs = model([gate * x for x in xs])
			loss_de = - sum([torch.mean((ys[e] - out_gs[e].detach()) * out_fs[e] - 0.5 * out_fs[e] * out_fs[e]) for e in range(num_envs)])
			loss_de.backward()
			optimizer_f.step()

		my_loss = np.zeros((niters_g, 2))

		# train the generator
		for i in range(niters_g):
			optimizer_g.zero_grad()
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()

			xs, ys = dataset.next_batch(batch_size)
			gate = model_var.generate_mask((tau_logits, tau))
			out_gs, out_fs = model([gate * x for x in xs])

			loss_r = 0.5 * sum([torch.mean((out_gs[e] - ys[e]) ** 2) for e in range(num_envs)])
			loss_j = sum([torch.mean((ys[e] - out_gs[e]) * out_fs[e] - 0.5 * out_fs[e] * out_fs[e]) for e in range(num_envs)])
			loss = loss_r + hyper_gamma * loss_j
			loss.backward()
			my_loss[i, 0], my_loss[i, 1] = loss_r.item(), loss_j.item()

			optimizer_g.step()
			optimizer_var.step()

		# save the weight/logits for linear model
		if it % iter_save == 0 and log:
			with torch.no_grad():
				logits = model_var.get_logits_numpy()
				gate_rec.append(sigmoid(logits / tau))
			print(logits)
			loss_rec.append(np.mean(my_loss, 0))



	ret = {'gate_rec': np.array(gate_rec),
			'model': model,
			'fair_var': model_var,
			'loss_rec': np.array(loss_rec)}

	return ret

