import numpy as np
import torch
import torch.optim as optim 
from tqdm import tqdm


class FairLinearPredictor(torch.nn.Module):
	def __init__(self, input_dim):
		super(FairLinearPredictor, self).__init__()
		self.linear = torch.nn.Linear(in_features=input_dim, out_features=1, bias=False)

	def forward(self, x):
		y = self.linear(x)
		return y


class FairLinearDiscriminator(torch.nn.Module):
	def __init__(self, input_dim, variable_weight):
		super(FairLinearDiscriminator, self).__init__()
		self.linear = torch.nn.Linear(in_features=input_dim, out_features=1, bias=False)

	def forward(self, x):
		y = self.linear(x)
		return y


def sample_gumbel(shape, eps=1e-20):
	U = torch.rand(shape)
	return -torch.log(-torch.log(U + eps) + eps)


def sigmoid(x):
	return 1/(1+np.exp(-x))


class GumbelGate(torch.nn.Module):
	def __init__(self, input_dim):
		super(GumbelGate, self).__init__()
		self.logits = torch.nn.Parameter((torch.rand(input_dim) - 0.5) * 1e-5)
		self.input_dim = input_dim

	def generate_mask(self, temperature, shape=None):
		if shape is None:
			shape = (1, self.input_dim)
		gumbel_softmax_sample = self.logits \
							+ sample_gumbel(self.logits.shape) \
							- sample_gumbel(self.logits.shape)
		mask = torch.sigmoid(gumbel_softmax_sample / temperature)
		return torch.reshape(mask, shape)


def fair_ll_sgd_gumbel(features, responses, hyper_gamma=10, learning_rate=1e-3, niters=50000, niters_d=3, niters_g=2, batch_size=32):
	num_envs = len(features)
	dim_x = np.shape(features[0])[1]

	# the gumbel gate
	fairll_var = GumbelGate(dim_x)
	# build predictor class G
	fairll_g = FairLinearPredictor(dim_x)
	optimizer_g = optim.Adam(fairll_g.parameters(), lr=learning_rate)
	optimizer_var = optim.Adam(fairll_var.parameters(), lr=learning_rate)
	
	for v in fairll_g.parameters():
		print(v)

	# build discriminator class F in |E| environments
	fairll_ds = []
	optimizer_ds = []
	for e in range(num_envs):
		print('======== environment {e} ========')
		ones = torch.tensor(np.array([1, 1, 1, 1, 1, 1])).float()
		fairll_d = FairLinearDiscriminator(dim_x, ones)#fairll_g.variable_weight())
		for v in fairll_d.parameters():
			print(v)
		print('\n')
		optimizer_d = optim.Adam(fairll_d.parameters(), lr=learning_rate)
		fairll_ds.append(fairll_d)
		optimizer_ds.append(optimizer_d)

	# construct dataset from numpy array
	train_datasets = []
	train_loaders = []
	for e in range(num_envs):
		train_dataset = torch.utils.data.TensorDataset(torch.tensor(features[e]).float(), torch.tensor(responses[e]).float())
		train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
		train_loaders.append(train_loader)
		train_datasets.append(train_dataset)

	gate_rec = []
	weight_rec = []
	# start training
	for it in tqdm(range(niters)):
		tau = max(0.5, np.exp(-(5/niters) * it))
		with torch.no_grad():
			weight = fairll_g.linear.weight.detach().cpu()
			logits = fairll_var.logits.detach().cpu()
			gate_rec.append(sigmoid(logits.numpy() / tau))
			ww = np.squeeze(weight.numpy() + 0.0)
			weight_rec.append(ww)

		for i in range(niters_d):
			for e in range(num_envs):
				optimizer_ds[e].zero_grad()
				feature_e, label_e = next(iter(train_loaders[e]))

				gate = fairll_var.generate_mask(tau)
				out_g = fairll_g(gate * feature_e)
				out_de = fairll_ds[e](gate * feature_e)
				loss_de = - torch.mean((label_e - out_g) * out_de - 0.5 * out_de * out_de)
				loss_de.backward()
				optimizer_ds[e].step()

		for i in range(niters_g):
			loss = 0
			optimizer_g.zero_grad()
			optimizer_var.zero_grad()
			gate = fairll_var.generate_mask(tau)

			for e in range(num_envs):
				optimizer_ds[e].zero_grad()
				feature_e, label_e = next(iter(train_loaders[e]))
				out_g = fairll_g(gate * feature_e)
				out_de = fairll_ds[e](gate * feature_e)
				residual = (out_g - label_e)

				loss += 0.5 * torch.mean(residual ** 2)
				loss += hyper_gamma * torch.mean((label_e - out_g) * out_de - 0.5 * out_de * out_de)
			loss.backward()

			optimizer_g.step()
			optimizer_var.step()

	return {'weight': ww * sigmoid(logits.numpy() / tau), 'weight_rec': np.array(weight_rec), 'gate_rec': np.array(gate_rec)}
