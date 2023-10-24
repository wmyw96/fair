import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#05348b',  # dark blue
	'#9acdc4',  # pain blue
	'#6bb392'
]
results = np.load('eills_demo_small.npy')

num_n = results.shape[0]
num_sml = results.shape[1]

vec_n = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
method_name = ['EILLS', 'EILLS+refit', "LS $S^*$", r"LS $G^c$", "Anchor", "PLS"]
method_idx = [0, 2, 3, 4, 5, 6]

lines = [
	'solid',
	'solid',
	'dotted',
	'dotted',
	'dashed',
	'dashed'
]

markers = [
	'D',
	'o',
	'x',
	'x',
	'x',
	'x',
	'x'
]

colors = [
	'#05348b',
	'#05348b',
	'#ae1908',
	'#ec813b',
	'#6bb392',
	'#6bb392',
]

fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)
plt.subplots_adjust(top=0.98, bottom=0.1, left=0.15, right=0.98)
ax1.set_ylabel(r"$\|\hat{\beta} - \beta^*\|_2^2$")

for (j, mid) in enumerate(method_idx):
	ax1.plot(vec_n, np.mean(results[:, :, mid, 0], axis=1), linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.set_xlabel('$n$')
ax1.set_yscale("log")
ax1.set_xscale("log")

ax1.legend(loc='upper right')
plt.show()
#plt.savefig("l2error_n_sigma.pdf")
