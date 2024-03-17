import numpy as np
import torch
import torch.optim as optim 
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict


def sample_gumbel(shape, device, eps=1e-20):
	'''
		The function to sample gumbel random variables

		Parameters
		----------
		shape : tuple/np.array/torch.tensor
			a int tuple characterizing the shape of the tensor, usually is (p,)
			
	'''
	U = torch.rand(shape, device=device)
	return -torch.log(-torch.log(U + eps) + eps)


def sigmoid(x):
	'''
		The function to apply sigmoid activation to each entry of a numpy array

		Parameters
		----------
		x : np.array
			the numpy array
			
	'''
	return 1/(1+np.exp(-np.minimum(np.maximum(x, -30), 30)))


class GumbelGate(torch.nn.Module):
	'''
		A class to implement the gumbel gate for the input (with dimension p)

		...
		Attributes
		----------
		logits : nn.Parameter
			the parameter characeterizing log(pi_i), where pi_i is the probability that the i-th gate is set to be 1.

		Methods
		----------
		__init__()
			Initialize the module

		generate_mask(temperature, shape)
			Implementation to generate a gumbel gate mask

	'''
	def __init__(self, input_dim, device):
		super(GumbelGate, self).__init__()
		self.logits = torch.nn.Parameter((torch.rand(input_dim, device=device) - 0.5) * 1e-5)
		self.input_dim = input_dim
		self.device = device

	def generate_mask(self, temperatures, shape=None):
		if shape is None:
			shape = (1, self.input_dim)
		gumbel_softmax_sample = self.logits / temperatures[0] \
							+ sample_gumbel(self.logits.shape, self.device) \
							- sample_gumbel(self.logits.shape, self.device)
		mask = torch.sigmoid(gumbel_softmax_sample / temperatures[1])
		return torch.reshape(mask, shape)

	def get_logits_numpy(self):
		return self.logits.detach().cpu().numpy()


class FairLinear(torch.nn.Module):
	'''
		A class to implement the linear model

		...
		Attributes
		----------
		linear : nn.module
			the linear model

		Methods
		----------
		__init__()
			Initialize the module

		forward(x, is_traininig=False)
			Implementation of forwards pass

	'''
	def __init__(self, input_dim, num_envs):
		'''
			Parameters
			----------
			input_dim : int
				input dimension
		'''
		super(FairLinear, self).__init__()
		self.g = torch.nn.Linear(in_features=input_dim, out_features=1, bias=True)
		self.num_envs = num_envs
		self.fs = []
		for e in range(num_envs):
			self.fs.append(torch.nn.Linear(in_features=input_dim, out_features=1, bias=True))

	def params_g(self):
		return self.g.parameters()

	def params_f(self):
		paras = []
		for i in range(self.num_envs):
			paras += self.fs[i].parameters()
		return paras

	def forward(self, xs):
		'''
			Parameters
			----------
			x : torch.tensor
				potential (batch_size, p) torch tensor

			Returns
			----------
			y : torch.tensor
				potential (batch_size, 1) torch tensor

		'''
		out_gs, out_fs = [], []
		for e in range(self.num_envs):
			out_gs.append(self.g(xs[e]))
			out_fs.append(self.fs[e](xs[e]))
		return out_gs, out_fs


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
	fairll_var = GumbelGate(dim_x, 'cpu')
	optimizer_var = optim.Adam(fairll_var.parameters(), lr=learning_rate)

	optimizer_g = optim.Adam(model.params_g(), lr=learning_rate)
	optimizer_f = optim.Adam(model.params_f(), lr=learning_rate)

	# construct dataset from numpy array
	xs = [torch.tensor(features[e]).float() for e in range(num_envs)]
	ys = [torch.tensor(responses[e]).float() for e in range(num_envs)]

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
		if it % 10000 == 0:
			print(f'annealed tau: {tau}')
		#print(tau)
		tau_logits = 1

		# train the discriminator
		for i in range(niters_d):
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()
			optimizer_g.zero_grad()
			gate = fairll_var.generate_mask((tau_logits, tau)).detach()

			out_gs, out_fs = model([gate * x for x in xs])
			loss_de = - sum([torch.mean((ys[e] - out_gs[e].detach()) * out_fs[e] - 0.5 * out_fs[e] * out_fs[e]) for e in range(num_envs)])
			loss_de.backward()
			optimizer_f.step()

		my_loss = np.zeros((niters_g, 2))

		# train the generator
		for i in range(niters_g):
			# set the gradient to be zero
			optimizer_g.zero_grad()
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()

			gate = fairll_var.generate_mask((tau_logits, tau))
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
				logits = fairll_var.get_logits_numpy()
				gate_rec.append(sigmoid(logits / tau))
				weight_rec.append(np.squeeze(weight.numpy() + 0.0))
			print(logits, np.squeeze(weight.numpy() + 0.0))
			loss_rec.append(np.mean(my_loss, 0))


	ret = {'weight': weight_rec[-1] * sigmoid(logits / tau_logits),
			'weight_rec': np.array(weight_rec),
			'gate_rec': np.array(gate_rec),
			'model': model,
			'fair_var': fairll_var,
			'loss_rec': np.array(loss_rec)}

	return ret
