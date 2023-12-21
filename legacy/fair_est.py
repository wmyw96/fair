import numpy as np
import torch
import torch.optim as optim 


class FairLinearPredictor(torch.nn.Module):
	def __init__(self, input_dim):
		super(FairLinearPredictor, self).__init__()
		self.linear = torch.nn.Linear(in_features=input_dim, out_features=1, bias=False)

	def forward(self, x):
		y = self.linear(x)
		return y

	def variable_weight(self):
		return self.linear.weight > 0


class FairLinearDiscriminator(torch.nn.Module):
	def __init__(self, input_dim, variable_weight):
		super(FairLinearDiscriminator, self).__init__()
		self.linear = torch.nn.Linear(in_features=input_dim, out_features=1, bias=False)
		#self.variable_weight = variable_weight

	def forward(self, x):
		#mx = x * self.variable_weight
		y = self.linear(x)
		return y


def fair_ll_sgd(features, responses, hyper_gamma=10, learning_rate=1e-3, niters=10000, niters_d=3, niters_g=2, batch_size=32):
	num_envs = len(features)
	dim_x = np.shape(features[0])[1]

	# build predictor class G
	fairll_g = FairLinearPredictor(dim_x)
	optimizer_g = optim.Adam(fairll_g.parameters(), lr=learning_rate)

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

	# start training
	for it in range(niters):
		
		for i in range(niters_d):
			for e in range(num_envs):
				optimizer_ds[e].zero_grad()
				feature_e, label_e = next(iter(train_loaders[e]))

				out_g = fairll_g(feature_e)
				out_de = fairll_ds[e](feature_e * fairll_g.variable_weight())
				loss_de = - torch.mean((label_e - out_g) * out_de - 0.5 * out_de * out_de)
				loss_de.backward()

				optimizer_ds[e].step()
				#print(f'updated d: {fairll_ds[e].linear.weight.detach()}')
				#print('===============================================\n')

		for i in range(niters_g):
			loss = 0
			optimizer_g.zero_grad()
			for e in range(num_envs):
				optimizer_ds[e].zero_grad()
				feature_e, label_e = next(iter(train_loaders[e]))
				out_g = fairll_g(feature_e)
				out_de = fairll_ds[e](feature_e * fairll_g.variable_weight())
				residual = (out_g - label_e)
				#print(residual.shape)
				loss += 0.5 * torch.mean(residual ** 2)
				loss += hyper_gamma * torch.sqrt(torch.sum(torch.square(fairll_g.linear.weight * torch.mean(residual * feature_e, 0, keepdim=True))))
				#loss += hyper_gamma * torch.mean((label_e - out_g) * out_de - 0.5 * out_de * out_de)
			loss.backward()
			print(f'updated g: {fairll_g.linear.weight.grad.detach().cpu()}')
			print('===============================================\n')
			
			optimizer_g.step()
			with torch.no_grad():
				fairll_g.linear.weight.copy_(fairll_g.linear.weight * (torch.abs(fairll_g.linear.weight) > 1e-5))

		with torch.no_grad():
			weight = fairll_g.linear.weight.detach().cpu()
			out_info = f'g weight = {weight}'
			for e in range(num_envs):
				out_info += f', d[{e}] weight = {fairll_ds[e].linear.weight.detach().cpu()}'
			print(out_info)

	return weight
