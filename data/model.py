import numpy as np

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
			for j in range(i):
				if np.random.uniform(0, 1) < ratio:
					function_matrix[i, j] = np.random.randint(function_id_max) + 1
					coefficient_matrix[i, j] = np.random.uniform(-coefficient_max, coefficient_max)
					cnt += 1
				if cnt >= degree_max:
					break
			coefficient_matrix[i, i] = np.abs(np.random.uniform(-coefficient_max, coefficient_max)) #+ 0.5
	else:
		for i in range(num_vars):
			for j in range(i):
				if reference_g[i, j] > 0:
					function_matrix[i, j] = np.random.randint(function_id_max) + 1
					coefficient_matrix[i, j] = np.random.uniform(-coefficient_max, coefficient_max)
			coefficient_matrix[i, i] = np.abs(np.random.uniform(-coefficient_max, coefficient_max)) #+ 0.5

	return function_matrix, coefficient_matrix


def linear_SCM(num_vars, num_envs=2, nonlinear_id=5):
	y_index = np.random.randint(num_vars - 1) + 1

	models = []
	func_mat0, coeff_mat0 = random_assignment_matrix(num_vars, 0.4, nonlinear_id, 1.5, 3)
	#print(func_mat0, y_index)
	for i in range(num_envs):
		func_mat, coeff_mat = random_assignment_matrix(num_vars, 0.4, nonlinear_id, 1.5, 3, func_mat0)
		func_mat[y_index, :] = np.minimum(func_mat0[y_index, :], 1)
		coeff_mat[y_index, :] = coeff_mat0[y_index, :]
		#print(func_mat)
		model = AdditiveStructuralCausalModel(num_vars, coeff_mat, func_mat, y_index)
		models.append(model)
	true = coeff_mat0[y_index, :]
	true[y_index] = 0.0
	return models, func_mat0[y_index, :], coeff_mat0[y_index, :]


