import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt
from data.model import *
from eills_demo_script.demo_wrapper import *
from utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)

TEST_ID = 2

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#05348b',  # dark blue
	'#9acdc4',  # pain blue
	'#6bb392',  # green
	'#e5a84b',   # yellow
]

if TEST_ID == 1:
	results = np.load('unit_test_1.npy')
	dim_x = 12

	env1_model = StructuralCausalModel1(dim_x + 1)
	env2_model = StructuralCausalModel2(dim_x + 1)
	X1_test, _1, _2 = env1_model.sample(10000)
	X2_test, _1, _2 = env2_model.sample(10000)
	X_cov = np.matmul(X1_test.T, X1_test) / 20000 + np.matmul(X2_test.T, X2_test) / 20000

	num_n = results.shape[0]
	num_sml = results.shape[1]

	vec_n = [100, 300, 700, 1000, 2000]
	method_name = ['EILLS', "FAIR-BF", "FAIR-Gumbel", r"FAIR-Gumbel-RefitLS", r"FAIR-Gumbel-RefitAdv", r"LS$_{S^\star}$", r"LS$_{G^c}$", "ERM"]
	method_idx = [0, 1, 2, 9, 8, 3, 4, 7]

	lines = [
		'dashed',
		'solid',
		'solid',
		'solid',
		'solid',
		'dotted',
		'dotted',
		'dashed'
	]

	markers = [
		'P',
		'o',
		'D',
		'P',
		'o',
		's',
		'x',
		'<'
	]

	colors = [
		'#05348b',
		'#6bb392',
		'#ae1908',
		'#ae1908',
		'#ae1908',
		'#ec813b',
		'#e5a84b',
		'#9acdc4'
	]

	fig = plt.figure(figsize=(8, 6))
	ax1 = fig.add_subplot(111)
	plt.subplots_adjust(top=0.98, bottom=0.1, left=0.17, right=0.98)
	ax1.set_ylabel(r"$\|\bar{\Sigma}^{1/2}(\hat{\beta} - \beta^*)\|_2^2$")

	dim_x = 12
	true_coeff = np.array([3, 2, -0.5] + [0] * 9)
	#true_coeff = np.reshape(true_coeff, (1, 1, dim_x))

	for (j, mid) in enumerate(method_idx):
		metric = []
		for i in range(len(vec_n)):
			measures = []
			for k in range(num_sml):
				if i == 3 and mid == 2:
					#print(results[i, k, mid, :])
					print(np.sum(np.square(results[i, k, mid, :] - true_coeff)), mydist(X_cov, results[i, k, mid, :] - true_coeff))
				measures.append(np.sum(np.square(results[i, k, mid, :] - true_coeff)))
				#measures.append(mydist(X_cov, results[i, k, mid, :] - true_coeff))
			metric.append(np.mean(measures))
		ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	ax1.set_xlabel('$n$')
	ax1.set_yscale("log")
	ax1.set_xscale("log")

	ax1.legend(loc='best')
	#plt.show()
	plt.savefig("l2_error.pdf")

if TEST_ID == 2:
	results = np.load('unit_test_2.npy')

	num_n = results.shape[0]
	num_sml = results.shape[1]

	vec_n = [100, 300, 700, 1000, 2000]
	method_name = ['EILLS', "FAIR-BF", "FAIR-Gumbel", r"FAIR-Gumbel-RefitLS", r"FAIR-Gumbel-RefitAdv", "Oracle", r"IRM", r"Anchor", "ERM"]
	method_idx = [0, 1, 2, 8, 7, 3, 4, 5, 6]

	lines = [
		'dashed',
		'solid',
		'solid',
		'solid',
		'solid',
		'dashed',
		'dotted',
		'dotted',
		'dashed'
	]

	markers = [
		'P',
		'o',
		'D',
		'P',
		'+',
		'*',
		's',
		'x',
		'<'
	]

	colors = [
		'#6bb392',
		'#05348b',
		'#05348b',
		'#05348b',
		'#05348b',
		'#ae1908',
		'#ec813b',
		'#e5a84b',
		'#9acdc4'
	]

	fig = plt.figure(figsize=(8, 6))
	ax1 = fig.add_subplot(111)
	plt.subplots_adjust(top=0.98, bottom=0.1, left=0.17, right=0.98)
	ax1.set_ylabel(r"$\|\hat{\beta} - \beta^\star\|_2^2$")

	for (j, mid) in enumerate(method_idx):
		metric = []
		for i in range(len(vec_n)):
			measures = []
			for k in range(num_sml):
				measures.append(np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :])))
			metric.append(np.mean(measures))
		ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	ax1.set_xlabel('$n$')
	ax1.set_yscale("log")
	ax1.set_xscale("log")

	ax1.legend(loc='best')
	#plt.show()
	plt.savefig("l2_error.pdf")


if TEST_ID == 3:
	results = np.load('unit_test_3.npy')

	num_n = results.shape[0]
	num_sml = results.shape[1]

	vec_n = [500, 1000, 2000, 5000, 10000]
	method_name = ["FAIR-Gumbel", "FAIR-G-RefitLS", "FAIR-G-RefitAdv", "Oracle", r"Semi-Oracle", "ERM"]
	method_idx = [0, 5, 4, 1, 2, 3]

	lines = [
		'solid',
		'solid',
		'solid',
		'dashed',
		'dashed',
		'dotted'
	]

	markers = [
		'D',
		'+',
		'o',
		'P',
		'*',
		'x',
	]

	colors = [
		'#05348b',
		'#9acdc4',
		'#6bb392',
		'#ae1908',
		'#ec813b',
		'#e5a84b'
	]

	fig = plt.figure(figsize=(8, 6))
	ax1 = fig.add_subplot(111)
	plt.subplots_adjust(top=0.98, bottom=0.1, left=0.17, right=0.98)
	ax1.set_ylabel(r"$\|\hat{\beta} - \beta^\star\|_2^2$")

	'''true_coeffs = np.zeros((num_sml, 70))
	for k in range(num_sml):
		np.random.seed(k)
		print(f'graph {k}')
		#generate random graph with 20 nodes
		models, true_coeff, parent_set, child_set, offspring_set = \
			get_linear_SCM(num_vars=71, num_envs=2, y_index=35, 
							min_child=5, min_parent=5, nonlinear_id=5, 
							bias_greater_than=0.5, log=False)
		true_coeffs[k, :] = true_coeff'''

	for (j, mid) in enumerate(method_idx):
		metric = []
		for i in range(len(vec_n)):
			measures = []
			for k in range(num_sml):
				#if mid == 3 and i == 4:
				#	print(results[i, k, mid+1, :])
				measures.append(np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :])))
				if mid+1 == 0:
					print(vec_n[i], np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :])))
			metric.append(np.median(measures))
		ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	ax1.set_xlabel('$n$')
	ax1.set_yscale("log")
	ax1.set_xscale("log")

	ax1.legend(loc='best')
	plt.show()
