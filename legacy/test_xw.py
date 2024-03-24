from data.model import *
from utils import *
import numpy as np
import os
import argparse
import time

from methods.gumbel2 import *
from methods.tools import *

np.random.seed(0)

TEST_MODE = 1

# Set data generating process
if TEST_MODE == 1:
	dim_x = 2
	models, true_coeff = SCM_ex1()
	parent_set, child_set, offspring_set = [0], [1], [1]
elif TEST_MODE == 2:
	dim_x = 12
	models, true_coeff = [StructuralCausalModel1(13), StructuralCausalModel2(13)], np.array([3, 2, -0.5] + [0] * (13 - 4))
	parent_set, child_set, offspring_set = [0, 1, 2], [6, 7], [6, 7, 8]

xs, ys, yts = sample_from_SCM(models, 64)

print(pooled_least_squares(xs, ys))
model, results = train(xs, ys, tau=0.5, gamma=30, lr=1e-3, lr_lam=1e-4, step_d=2, num_epochs=50000, print_every=500, ini_value=0)