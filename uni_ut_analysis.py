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


if TEST_ID == 2:
	results = np.load('uni_unit_test_2.npy')

	num_n = results.shape[0]
	num_sml = results.shape[1]

	vec_n = [100, 300, 700, 1000, 2000]
	method_name = ['EILLS', "FAIR-BF", "FAIR-GB", r"FAIR-RF", "Oracle", r"IRM", r"Anchor", "ERM"]
	method_idx = [0, 1, 2, 7, 3, 4, 5, 6]

	lines = [
		'dashed',
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
				error = np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :]))
				if error > 0.2 and mid < 3:
					print(f'method = {mid}, n = {vec_n[i]}, seed = {k}, error = {error}')
				measures.append(np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :])))
			metric.append(np.mean(measures))
		ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	ax1.set_xlabel('$n$')
	#ax1.set_yscale("log")
	ax1.set_xscale("log")

	ax1.legend(loc='best')
	plt.show()
	#plt.savefig("l2_error.pdf")


if TEST_ID == 3:
	results = np.load('uni_unit_test_3.npy')

	num_n = results.shape[0]
	num_sml = results.shape[1]

	vec_n = [500, 1000, 2000, 5000, 10000]
	method_name = ["FAIR-GB", "FAIR-RF", "Oracle", r"Semi-Oracle", "ERM"]
	method_idx = [0, 4, 1, 2, 3]

	lines = [
		'solid',
		'solid',
		'dashed',
		'dashed',
		'dashed'
	]

	markers = [
		'P',
		'D',
		'o',
		'*',
		'x',
	]

	colors = [
		'#6bb392',
		'#05348b',
		'#ae1908',
		'#ec813b',
		'#e5a84b',
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
				error = np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :]))
				if error > 0.2 and mid != 3:
					print(f'method = {mid}, n = {vec_n[i]}, seed = {k}, error = {error}')
				measures.append(np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :])))
			metric.append(np.median(measures))
		ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	ax1.set_xlabel('$n$')
	ax1.set_yscale("log")
	ax1.set_xscale("log")

	ax1.legend(loc='best')
	plt.show()
