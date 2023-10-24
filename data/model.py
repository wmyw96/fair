import numpy as np


def generate_exogeneous_variables(n, p, cov_sqrt, dist):
	u = np.zeros((n, p))
	if dist == 'normal':
		u = np.random.normal(0, 1, (n, p))
		u = np.matmul(u, cov_sqrt)
	elif dist == 'uniform':
		u = np.random.uniform(-1, 1, (n, p))
		u = np.matmul(u, cov_sqrt)
	else:
		raise ValueError(f"Error: exogenous variable distribution {dist} not defined.")
	return u


class StructuralCausalModel1:
	def __init__(self, p, exogenous_cov=None, exogenous_dist='normal'):
		self.p = p
		if exogenous_cov is None:
			exogenous_cov = np.eye(p)
		exogenous_cov = np.diag(np.diag(exogenous_cov))
		self.exogenous_cov = exogenous_cov
		self.exogenous_cov_sqrt = np.sqrt(exogenous_cov)
		self.exogenous_dist = exogenous_dist

	def sample(self, n, split=True):
		u = generate_exogeneous_variables(n, self.p, self.exogenous_cov_sqrt, self.exogenous_dist)
		x = np.copy(u)
		# x4
		x[:, 4] = u[:, 4]
		x[:, 1] = u[:, 1]
		x[:, 2] = np.sin(x[:, 4]) * 1 + u[:, 2]
		x[:, 3] = np.cos(x[:, 4]) * 1 + u[:, 3]
		x[:, 5] = np.sin(x[:, 3] + u[:, 5])
		x[:, 10] = x[:, 1] * 2.5 + x[:, 2] * 1.5 + u[:, 10]
		x[:, 0] = x[:, 1] * 3 + x[:, 2] * 2 + x[:, 3] * (-0.5) + u[:, 0]
		x[:, 6] = 0.8 * x[:, 0] * u[:, 6]
		x[:, 7] = x[:, 3] * 0.5 + x[:, 0] + u[:, 7]
		x[:, 8] = 0.5 * x[:, 7] + x[:, 0] * (-1) + x[:, 10] + u[:, 8]
		x[:, 9] = np.tanh(x[:, 7]) + 0.1 * np.cos(x[:, 8]) + u[:, 9]
		x[:, 11] = 0.4 * (x[:, 7] + x[:, 8]) * u[:, 11]
		if split:
			return x[:, 1:], x[:, :1], x[:, :1] - u[:, :1]
		else:
			return x


class StructuralCausalModel2:
	def __init__(self, p, exogenous_cov=None, exogenous_dist='normal'):
		self.p = p
		if exogenous_cov is None:
			exogenous_cov = np.eye(p)
		exogenous_cov = np.diag(np.diag(exogenous_cov))
		self.exogenous_cov = exogenous_cov
		self.exogenous_cov_sqrt = np.sqrt(exogenous_cov)
		self.exogenous_dist = exogenous_dist

	def sample(self, n, split=True):
		u = generate_exogeneous_variables(n, self.p, self.exogenous_cov_sqrt, self.exogenous_dist)
		x = np.copy(u)
		# print(f'E[x eps] = {np.mean(u[:, 0] * u[:, 1])}')

		# x4
		x[:, 4] = u[:, 4] ** 2 - 1
		x[:, 1] = u[:, 1]
		x[:, 2] = np.sin(x[:, 4]) * 1 + u[:, 2]
		x[:, 3] = np.cos(x[:, 4]) * 1 + u[:, 3]
		x[:, 5] = np.sin(x[:, 3] + u[:, 5])
		x[:, 10] = x[:, 1] * 2.5 + x[:, 2] * 1.5 + u[:, 10]
		x[:, 0] = x[:, 1] * 3 + x[:, 2] * 2 + x[:, 3] * (-0.5) + u[:, 0]
		x[:, 6] = 0.8 * x[:, 0] * u[:, 6]
		x[:, 7] = x[:, 3] * 4 + np.tanh(x[:, 0]) + u[:, 7]
		x[:, 8] = 0.5 * x[:, 7] + x[:, 0] * (-1) + x[:, 10] + u[:, 8]
		x[:, 9] = np.tanh(x[:, 7]) + 0.1 * np.cos(x[:, 8]) + u[:, 9]
		x[:, 11] = 0.4 * (x[:, 7] + x[:, 8]) * u[:, 11]
		if split:
			return x[:, 1:], x[:, :1], x[:, :1] - u[:, :1]
		else:
			return x

class AdditiveStructuralCausalModel:
	'''
		Consider the simple SCM that the strctural assignments admits additive form, that is 

			x_j <- sum_{k in pa(j)} coeff_{j,k} f_{j,k} (x_k) + coeff_{j,j} u_j
	
	'''
	def __init__(self, num_vars, coefficients_matrix, assignments_matrix, y_index, randtype='gaussian'):
		self.num_vars = num_vars
		self.y_index = y_index
		self.coefficients = coefficients_matrix
		self.assignments = assignments_matrix
		self.randtype = randtype

	def sample(self, n, split=True):
		z = np.zeros((n, self.num_vars))
		if self.randtype == 'gaussian':
			u = np.random.normal(0, 1, (n * self.num_vars))
		else:
			# use unit variance
			u = np.random.uniform(-np.sqrt(1.5), np.sqrt(1.5), (n * self.num_vars))
		u = np.reshape(u, (n, self.num_vars))
		for i in range(self.num_vars):
			z[:, i] = u[:, i] * self.coefficients[i, i]
			for j in range(i):
				function_id = self.assignments[i, j]
				pre_factor = self.coefficients[i, j]
				if function_id == 1:
					# linear function
					z[:, i] += pre_factor * (z[:, j])
				elif function_id == 2:
					# sin function
					z[:, i] += pre_factor * np.sin(z[:, j])
				elif function_id == 3:
					z[:, i] += pre_factor * np.cos(z[:, j])
				elif function_id == 4:
					z[:, i] += pre_factor * np.sin(np.pi * z[:, j])
				elif function_id == 5:
					z[:, i] += pre_factor * np.sqrt(np.abs(z[:, j]))
		if split:
			x = np.concatenate([z[:, :self.y_index], z[:, (self.y_index+1):]], 1)
			y = z[:, self.y_index:(self.y_index+1)]
			y_gt = z[:, self.y_index:(self.y_index+1)] - \
					self.coefficients[self.y_index, self.y_index] * u[:, self.y_index:(self.y_index+1)]
			return (x, y, y_gt)
		else:
			return z


def random_assignment_matrix(num_vars, ratio, function_id_max, coefficient_max, degree_max, reference_g=None):
	function_matrix = np.zeros((num_vars, num_vars), dtype=np.int)
	coefficient_matrix = np.zeros((num_vars, num_vars), dtype=np.float)
	if reference_g is None:
		for i in range(num_vars):
			cnt = 0
			idx = np.random.permutation(i)
			for j in range(i):
				if np.random.uniform(0, 1) < ratio:
					function_matrix[i, idx[j]] = np.random.randint(function_id_max) + 1
					coefficient_matrix[i, idx[j]] = np.random.uniform(-coefficient_max, coefficient_max)
					cnt += 1
				if cnt >= degree_max:
					break
			coefficient_matrix[i, i] = np.abs(np.random.uniform(-coefficient_max, coefficient_max))
	else:
		for i in range(num_vars):
			for j in range(i):
				if reference_g[i, j] > 0:
					function_matrix[i, j] = np.random.randint(function_id_max) + 1
					coefficient_matrix[i, j] = np.random.uniform(-coefficient_max, coefficient_max)
			coefficient_matrix[i, i] = np.abs(np.random.uniform(-coefficient_max, coefficient_max))

	return function_matrix, coefficient_matrix


def linear_SCM(num_vars, y_index=None, min_child=0, min_parent=0, num_envs=2, nonlinear_id=5):
	if y_index is None:
		y_index = np.random.randint(num_vars - 1) + 1

	models = []
	func_mat0, coeff_mat0 = random_assignment_matrix(num_vars, 0.4, nonlinear_id, 1, 3)
	func_mat0[y_index, :] = np.minimum(func_mat0[y_index, :], 1)

	num_child = np.sum(func_mat0 > 0, 0)[y_index]
	if num_child < min_child:
		remain_child = min_child - num_child
		for i in range(y_index+1, num_vars):
			if func_mat0[i, y_index] == 0:
				remain_child -= 1
				func_mat0[i, y_index] = 1
				print(f'assign child {i}')
			if remain_child == 0:
				break

	num_parent = np.sum(func_mat0 > 0, 1)[y_index]
	if num_parent < min_parent:
		remain_parent = min_parent - num_parent
		for i in range(y_index):
			if func_mat0[y_index, i] == 0:
				remain_parent -= 1
				func_mat0[y_index, i] = 1
			if remain_parent == 0:
				break

	parent_set = []
	# enforce large signal
	for i in range(y_index):
		if func_mat0[y_index, i] > 0:
			coeff_mat0[y_index, i] = (np.abs(np.random.uniform(0, 0.5)) + 1) * (2*np.random.randint(2)-1)
			parent_set.append(i)
	coeff_mat0[y_index, y_index] = 1

	# enforce large bias
	child_set = []
	for i in range(y_index+1, num_vars):
		if func_mat0[i, y_index] > 0:
			coeff_mat0[i, y_index] = (np.abs(np.random.uniform(0, 1)) + 0.5) * (2*np.random.randint(2)-1)
			child_set.append(i)

	print(f'Linear SCM: parents = {parent_set}, children = {[(i-1) for i in child_set]}')
	print(func_mat0, y_index)
	for i in range(num_envs):
		func_mat, coeff_mat = random_assignment_matrix(num_vars, 0.4, nonlinear_id, 1.5, 3, func_mat0)
		func_mat[y_index, :] = func_mat0[y_index, :]
		coeff_mat[y_index, :] = coeff_mat0[y_index, :]
		for child in child_set:
			while True:
				coeff_mat[child, y_index] = (np.abs(np.random.uniform(0, 1)) + 0.5) * (2*np.random.randint(2)-1)
				#print(coeff_mat0[child, y_index])
				if np.abs(coeff_mat[child, y_index] - coeff_mat0[child, y_index]) > 0.5 and np.abs(coeff_mat[child, y_index] + coeff_mat0[child, y_index]) > 0.5:
					break
		model = AdditiveStructuralCausalModel(num_vars, coeff_mat, func_mat, y_index)
		models.append(model)
	true = coeff_mat0[y_index, :]
	true[y_index] = 0.0
	return models, func_mat0[y_index, :-1], coeff_mat0[y_index, :-1], parent_set


def SCM_ex1():
	num_vars = 3
	y_index = 1
	func_mat = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.int)
	coeff1 = np.array([[1, 0, 0], [2, 1, 0], [1, -0.3, 1]], dtype=np.float)
	coeff2 = np.array([[1.5, 0, 0], [2, 1, 0], [0.3, 0.7, 0.3]], dtype=np.float)
	models = [
		AdditiveStructuralCausalModel(num_vars, coeff1, func_mat, y_index),
		AdditiveStructuralCausalModel(num_vars, coeff2, func_mat, y_index)
	]
	return models, np.array([1, 0], dtype=np.int), np.array([2, 0], dtype=np.float)

