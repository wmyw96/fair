import numpy as np
import torch
import torch.optim as optim 
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict


class FairLinearModel(torch.nn.Module):
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
	def __init__(self, input_dim):
		'''
			Parameters
			----------
			input_dim : int
				input dimension
		'''
		super(FairLinearModel, self).__init__()
		self.linear = torch.nn.Linear(in_features=input_dim, out_features=1, bias=True)

	def standardize(self, train_x):
		self.x_mean = torch.tensor(np.mean(train_x, 0, keepdims=True)).float()
		#print(self.x_mean)
		self.x_std = torch.tensor(np.std(train_x, 0, keepdims=True)).float()
		#print(self.x_std)

	def forward(self, x):
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
		x = (x - self.x_mean) / self.x_std
		y = self.linear(x)
		return y


class FairNeuralNetwork(torch.nn.Module):
	'''
		A class to implemented fully connected neural network

		...
		Attributes
		----------
		relu_stack: nn.module
			the relu neural network module

		Methods
		----------
		__init__()
			Initialize the module
		forward(x)
			Implementation of forwards pass
	'''
	def __init__(self, input_dim, depth, width, out_act=None, res_connect=True):
		'''
			Parameters
			----------
			input_dim : int
				input dimension
			depth : int
				the number of hidden layers of neural network, depth = 0  ===>  linear model
			width : int
				the number of hidden units in each layer
			res_connect : bool
				whether or not to use residual connection
		'''
		super(FairNeuralNetwork, self).__init__()

		if depth >= 1:
			self.relu_nn = [('linear1', nn.Linear(input_dim, width)), ('relu1', nn.ReLU())]
			for i in range(depth - 1):
				self.relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
				self.relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))

			self.relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))
			self.relu_stack = nn.Sequential(
				OrderedDict(self.relu_nn)
			)

			self.res_connect = res_connect
			if self.res_connect:
				self.linear_res = torch.nn.Linear(in_features=input_dim, out_features=1, bias=False)
		else:
			self.relu_stack = torch.nn.Linear(in_features=input_dim, out_features=1, bias=True)
			self.res_connect = False

		self.out_act = out_act


	def standardize(self, train_x):
		self.x_mean = torch.tensor(np.mean(train_x, 0, keepdims=True)).float()
		self.x_std = torch.tensor(np.std(train_x, 0, keepdims=True)).float()

	def forward(self, x):
		'''
			Parameters
			----------
			x : torch.tensor
				(n, p) matrix of the input

			Returns
			----------
			out : torch.tensor
				(n, 1) matrix of the prediction
		'''
		x = (x - self.x_mean) / self.x_std
		out = self.relu_stack(x)
		if self.res_connect:
			out = out + self.linear_res(x)
		if self.out_act is not None:
			out = self.out_act(out)
		return out


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


class FixedGate(torch.nn.Module):
	'''
	'''
	def __init__(self, input_dim, mask, device):
		self.mask = torch.tensor(mask, device=device).float()
		self.logits = self.mask * 100 - (1 - self.mask) * 100
		self.input_dim = input_dim
		self.device = device

	def generate_mask(self, temperatures, shape=None):
		if shape is None:
			shape = (1, self.input_dim)
		return torch.reshape(self.mask, shape)

	def get_logits_numpy(self):
		return self.logits.detach().cpu().numpy()


def fair_ll_sgd_gumbel(features, responses, hyper_gamma=10, learning_rate=1e-3, niters=50000, niters_d=2, niters_g=1, 
						batch_size=32, mask=None, init_temp=0.5, final_temp=0.005, iter_save=100, log=False):
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
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if log:
		print(f'================================================================================')
		print(f'================================================================================')
		print(f'==')
		print(f'==  FAIR Linear/Linear Model Gumbel: num of envs = {num_envs}, x dim = {dim_x}')
		print(f'==')
		print(f'================================================================================')
		print(f'================================================================================')

	# build the gumbel gate
	use_gumbel = False
	if mask is None:
		fairll_var = GumbelGate(dim_x, device)
		optimizer_var = optim.Adam(fairll_var.parameters(), lr=1e-3)
		use_gumbel = True
	else:
		print(f'use fixed mask: {mask}')
		fairll_var = FixedGate(dim_x, mask, device)

	# build predictor class G
	fairll_g = FairLinearModel(dim_x).to(device)
	optimizer_g = optim.Adam(fairll_g.parameters(), lr=learning_rate)
	fairll_g.standardize(np.concatenate(features, 0))
	
	# build discriminator class F in |E| environments
	fairll_ds = []
	optimizer_ds = []
	for e in range(num_envs):
		fairll_d = FairLinearModel(dim_x).to(device)
		fairll_d.standardize(features[e])
		optimizer_d = optim.Adam(fairll_d.parameters(), lr=learning_rate)
		fairll_ds.append(fairll_d)
		optimizer_ds.append(optimizer_d)

	# construct dataset from numpy array
	train_datasets = []
	train_loaders = []
	sample_size = 1e9
	for e in range(num_envs):
		sample_size = min(sample_size, np.shape(responses[e])[0])
		train_dataset = torch.utils.data.TensorDataset(torch.tensor(features[e]).float(), torch.tensor(responses[e]).float())
		train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
		train_loaders.append(train_loader)
		train_datasets.append(train_dataset)

	gate_rec = []
	weight_rec = []
	loss_rec = []
	# start training
	if log:
		it_gen = tqdm(range(niters))
	else:
		it_gen = range(niters)

	print(sample_size)
	lda_weight = np.log(dim_x) / sample_size * 5

	for it in it_gen:
		# calculate the temperature
		tau = max(init_temp, final_temp * np.exp(-(1/niters) * it))
		tau_logits = 1

		# train the discriminator
		for i in range(niters_d):
			for e in range(num_envs):
				optimizer_ds[e].zero_grad()

				feature_e, label_e = next(iter(train_loaders[e]))
				feature_e = feature_e.to(device)
				label_e = label_e.to(device)

				gate = fairll_var.generate_mask((tau_logits, tau))
				out_g = fairll_g(gate * feature_e)
				out_de = fairll_ds[e](gate * feature_e)

				loss_de = - torch.mean((label_e - out_g) * out_de - 0.5 * out_de * out_de)
				loss_de.backward()
				optimizer_ds[e].step()

		my_loss = np.zeros((niters_g, 2))

		# train the generator
		for i in range(niters_g):
			loss_r = 0
			loss_j = 0
			# set the gradient to be zero
			optimizer_g.zero_grad()
			if use_gumbel:
				optimizer_var.zero_grad()

			# generate the same mask for all the environment
			gate = fairll_var.generate_mask((tau_logits, tau))

			for e in range(num_envs):
				optimizer_ds[e].zero_grad()
				
				feature_e, label_e = next(iter(train_loaders[e]))
				feature_e = feature_e.to(device)
				label_e = label_e.to(device)

				out_g = fairll_g(gate * feature_e)
				out_de = fairll_ds[e](gate * feature_e)
				residual = (out_g - label_e)

				loss_r += 0.5 * torch.mean(residual ** 2)
				loss_j += torch.mean((label_e - out_g) * out_de - 0.5 * out_de * out_de)
			loss = loss_r + hyper_gamma * loss_j + lda_weight * torch.sum(torch.tanh(torch.sigmoid(fairll_var.logits) / tau))
			loss.backward()
			my_loss[i, 0], my_loss[i, 1] = loss_r.item(), loss_j.item()

			optimizer_g.step()
			if use_gumbel:
				optimizer_var.step()

		# save the weight/logits for linear model
		if it % iter_save == 0:
			with torch.no_grad():
				weight = fairll_g.linear.weight.detach().cpu() / np.squeeze(fairll_g.x_std.numpy())
				logits = fairll_var.get_logits_numpy()
				gate_rec.append(sigmoid(logits / tau))
				weight_rec.append(np.squeeze(weight.numpy() + 0.0))
			loss_rec.append(np.mean(my_loss, 0))


	ret = {'weight': weight_rec[-1] * sigmoid(logits / tau_logits),
			'weight_rec': np.array(weight_rec),
			'gate_rec': np.array(gate_rec),
			'fair_g': fairll_g,
			'fair_ds': fairll_ds,
			'fair_var': fairll_var,
			'loss_rec': np.array(loss_rec)}

	return ret


def ho_ll_sgd_gumbel(features, responses, hyper_gamma=10, learning_rate=1e-3, niters=50000, niters_d=1, niters_g=1, 
						batch_size=32, mask=None, init_temp=0.5, final_temp=0.005, iter_save=100, log=False):
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
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if log:
		print(f'================================================================================')
		print(f'================================================================================')
		print(f'==')
		print(f'==  FAIR Linear/Linear (DO) Model Gumbel: num of envs = {num_envs}, x dim = {dim_x}')
		print(f'==')
		print(f'================================================================================')
		print(f'================================================================================')

	# build the gumbel gate
	use_gumbel = False
	if mask is None:
		fairll_var = GumbelGate(dim_x, device)
		optimizer_var = optim.Adam(fairll_var.parameters(), lr=1e-3)
		use_gumbel = True
	else:
		print(f'use fixed mask: {mask}')
		fairll_var = FixedGate(dim_x, mask, device)

	# build predictor class G
	fairll_g = FairLinearModel(dim_x).to(device)
	optimizer_g = optim.Adam(fairll_g.parameters(), lr=learning_rate)
	fairll_g.standardize(np.concatenate(features, 0))
	
	# build discriminator class F in |E| environments
	fairll_ds = []
	optimizer_ds = []
	for e in range(num_envs):
		fairll_d = FairLinearModel(dim_x).to(device)
		fairll_d.standardize(features[e])
		optimizer_d = optim.Adam(fairll_d.parameters(), lr=learning_rate)
		fairll_ds.append(fairll_d)
		optimizer_ds.append(optimizer_d)

	# construct dataset from numpy array
	train_datasets = []
	train_loaders = []
	sample_size = 1e9
	for e in range(num_envs):
		sample_size = min(sample_size, np.shape(responses[e])[0])
		train_dataset = torch.utils.data.TensorDataset(torch.tensor(features[e]).float(), torch.tensor(responses[e]).float())
		train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
		train_loaders.append(train_loader)
		train_datasets.append(train_dataset)

	gate_rec = []
	weight_rec = []
	loss_rec = []
	# start training
	if log:
		it_gen = tqdm(range(niters))
	else:
		it_gen = range(niters)

	print(sample_size)
	lda_weight = np.log(dim_x) / sample_size * 0

	for it in it_gen:
		# calculate the temperature
		tau = max(init_temp, final_temp * np.exp(-(1/niters) * it))
		tau_logits = 1

		# train the discriminator
		for i in range(niters_d):
			for e in range(num_envs):
				optimizer_ds[e].zero_grad()

				feature_e, label_e = next(iter(train_loaders[e]))
				feature_e = feature_e.to(device)
				label_e = label_e.to(device)

				gate = fairll_var.generate_mask((tau_logits, tau))
				print(gate.shape, feature_e.shape)
				out_de = fairll_ds[e](gate * feature_e)

				loss_de = torch.mean(0.5 * (out_de - label_e) * (out_de - label_e))
				loss_de.backward()
				optimizer_ds[e].step()

		my_loss = np.zeros((niters_g, 2))

		# train the generator
		for i in range(niters_g):
			loss_r = 0
			loss_j = 0
			# set the gradient to be zero
			optimizer_g.zero_grad()
			if use_gumbel:
				optimizer_var.zero_grad()

			# generate the same mask for all the environment
			gate = fairll_var.generate_mask((tau_logits, tau))

			for e in range(num_envs):
				optimizer_ds[e].zero_grad()
				
				feature_e, label_e = next(iter(train_loaders[e]))
				feature_e = feature_e.to(device)
				label_e = label_e.to(device)

				out_g = fairll_g(gate * feature_e)
				out_de = fairll_ds[e](gate * feature_e)
				residual = (out_g - label_e)

				loss_r += 0.5 * torch.mean(residual ** 2)
				loss_j += torch.mean(0.5 * (out_g - out_de) * (out_g - out_de))
			loss = loss_r + hyper_gamma * loss_j + lda_weight * torch.sum(torch.tanh(torch.sigmoid(fairll_var.logits) / tau))
			loss.backward()
			my_loss[i, 0], my_loss[i, 1] = loss_r.item(), loss_j.item()

			optimizer_g.step()
			if use_gumbel:
				optimizer_var.step()

		# save the weight/logits for linear model
		if it % iter_save == 0:
			with torch.no_grad():
				weight = fairll_g.linear.weight.detach().cpu() / np.squeeze(fairll_g.x_std.numpy())
				logits = fairll_var.get_logits_numpy()
				gate_rec.append(sigmoid(logits / tau))
				weight_rec.append(np.squeeze(weight.numpy() + 0.0))
			loss_rec.append(np.mean(my_loss, 0))


	ret = {'weight': weight_rec[-1] * sigmoid(logits / tau_logits),
			'weight_rec': np.array(weight_rec),
			'gate_rec': np.array(gate_rec),
			'fair_g': fairll_g,
			'fair_ds': fairll_ds,
			'fair_var': fairll_var,
			'loss_rec': np.array(loss_rec)}

	return ret




def fair_nn_gumbel_regression(features, responses, validset, testset, hyper_gamma=10, depth=2, width=64,
						learning_rate=1e-3, niters=50000, niters_d=3, niters_g=2, gate_samples=20,
						batch_size=32, init_temp=0.5, final_temp=0.005, eval_iter=3000, log=False):
	'''
		Implementation of FAIR-LL estimator with gumbel discrete approximation

		Parameter
		----------
		features : list 
			list of numpy matrices with shape (n_k, p) representing the explanatory variables
		responses : list
			list of numpy matrices with shape (n_k, 1) representing the response variable
		validset : tuple
			valid set, tuple of ((nv, p) numpy array, (nv, 1) numpy array)
		testset : tuple
			test set, tuple of ((nt, p) numpy array, (nt, 1) numpy array)
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

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if log:
		print(f'================================================================================')
		print(f'================================================================================')
		print(f'==')
		print(f'==  FAIR NN Model Gumbel: num of envs = {num_envs}, x dim = {dim_x}, depth = {depth}, width = {width}')
		print(f'==')
		print(f'================================================================================')
		print(f'================================================================================')

	# build the gumbel gate
	fairnn_var = GumbelGate(dim_x, device)
	optimizer_var = optim.Adam(fairnn_var.parameters(), lr=learning_rate)

	# build predictor class G
	fairnn_g = FairNeuralNetwork(dim_x, depth - 2, width, res_connect=True).to(device)
	optimizer_g = optim.Adam(fairnn_g.parameters(), lr=learning_rate)
	fairnn_g.standardize(np.concatenate(features, 0))

	# build discriminator class F in |E| environments
	fairnn_ds = []
	optimizer_ds = []
	for e in range(num_envs):
		fairnn_d = FairNeuralNetwork(dim_x, depth, 2*width, res_connect=True).to(device)
		fairnn_d.standardize(features[e])
		optimizer_d = optim.Adam(fairnn_d.parameters(), lr=learning_rate)
		fairnn_ds.append(fairnn_d)
		optimizer_ds.append(optimizer_d)

	# construct dataset from numpy array
	train_datasets = []
	train_loaders = []
	sample_size = 1e9
	for e in range(num_envs):
		sample_size = min(sample_size, np.shape(responses[e])[0])
		train_dataset = torch.utils.data.TensorDataset(torch.tensor(features[e]).float(), torch.tensor(responses[e]).float())
		train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
		train_loaders.append(train_loader)
		train_datasets.append(train_dataset)

	valid_x, valid_y = torch.tensor(validset[0]).float(), torch.tensor(validset[1]).float()
	test_x, test_y = torch.tensor(testset[0]).float(), torch.tensor(testset[1]).float()

	gate_rec = []
	# start training
	if log:
		it_gen = tqdm(range(niters))
	else:
		it_gen = range(niters)

	print(f'sample size = {sample_size}')
	lda_weight = np.log(dim_x) / sample_size * 0.

	for it in it_gen:
		# calculate the temperature
		tau = max(init_temp, final_temp * np.exp(-(1/niters) * it))
		tau_logits = 1

		if it % eval_iter == 0:
			# save the logits
			with torch.no_grad():
				logits = fairnn_var.logits.detach().cpu()
				gate_rec.append(sigmoid(logits.numpy()))

			# calculate valid loss
			preds = []
			for k in range(gate_samples):
				gate = fairnn_var.generate_mask((tau_logits, tau))
				pred = fairnn_g(gate * valid_x)
				preds.append(pred.detach().cpu().numpy())
			out = sum(preds) / len(preds)
			valid_loss = np.mean(np.square(out - validset[1]))

			# calculate test loss
			preds = []
			gates = []
			for k in range(gate_samples):
				gate = fairnn_var.generate_mask((tau_logits, tau))
				pred = fairnn_g(gate * test_x)
				preds.append(pred.detach().cpu().numpy())
				gates.append((gate).detach().cpu().numpy() + 0.0)
			out = sum(preds) / len(preds)
			test_loss = np.mean(np.square(out - testset[1]))

			print(f'iter = {it}, test_loss = {test_loss}, gate = {sigmoid(logits.numpy())}, gate2 = {sum(gates) / len(gates)}')


		# train the discriminator
		for i in range(niters_d):
			for e in range(num_envs):
				optimizer_ds[e].zero_grad()
				feature_e, label_e = next(iter(train_loaders[e]))

				gate = fairnn_var.generate_mask((tau_logits, tau))
				out_g = fairnn_g(gate * feature_e)
				out_de = fairnn_ds[e](gate * feature_e)
				loss_de = - torch.mean((label_e - out_g) * out_de - 0.5 * out_de * out_de)
				loss_de.backward()
				optimizer_ds[e].step()

		# train the generator
		for i in range(niters_g):
			loss_r = 0
			loss_j = 0
			# set the gradient to be zero
			optimizer_g.zero_grad()
			optimizer_var.zero_grad()

			# generate the same mask for all the environment
			gate = fairnn_var.generate_mask((tau_logits, tau))

			for e in range(num_envs):
				optimizer_ds[e].zero_grad()
				
				feature_e, label_e = next(iter(train_loaders[e]))
				feature_e = feature_e.to(device)
				label_e = label_e.to(device)

				out_g = fairnn_g(gate * feature_e)
				out_de = fairnn_ds[e](gate * feature_e)
				residual = (out_g - label_e)

				loss_r += 0.5 * torch.mean(residual ** 2)
				loss_j += torch.mean((label_e - out_g) * out_de - 0.5 * out_de * out_de)
			loss = loss_r + hyper_gamma * loss_j + lda_weight * torch.sum(torch.tanh(torch.sigmoid(fairnn_var.logits) / tau))
			loss.backward()

			optimizer_g.step()
			optimizer_var.step()

	ret = { 'gate_rec': np.array(gate_rec),
			'fairnn_g': fairnn_g,
			'fairnn_ds': fairnn_ds,
			'fairnn_var': fairnn_var}

	return ret
