from data.model import *
from methods.brute_force import brute_force
from methods.fair_gumbel import fair_nn_gumbel_regression
from methods.tools import pooled_least_squares
from utils import get_linear_SCM
import numpy as np
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", type=int, default=1234)
parser.add_argument("--n", help="number of samples", type=int, default=1000)
parser.add_argument("--batch_size", help="batch size", type=int, default=36)
parser.add_argument("--num_envs", help="number of environments", type=int, default=2)
parser.add_argument("--dim_x", help="number of explanatory vars", type=int, default=60)
parser.add_argument("--niters", help="number of iterations", type=int, default=50000)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
parser.add_argument("--min_child", help="min number of children", type=int, default=5)
parser.add_argument("--min_parent", help="min number of parents", type=int, default=5)
parser.add_argument("--lsbias", help="least square bias", type=float, default=0.5)
parser.add_argument('--init_temp', help='initial temperature', type=float, default=5)
parser.add_argument('--final_temp', help='final temperature', type=float, default=0.5)
parser.add_argument('--gamma', help='hyper parameter gamma', type=float, default=36)
parser.add_argument("--record_dir", help="record directory", type=str, default="logs/")
parser.add_argument("--log", help="show log", type=bool, default=True)

args = parser.parse_args()

exp_name = f"n{args.n}_nenvs{args.num_envs}_dimx{args.dim_x}_niters{args.niters}_mch_{args.min_child}_mpa{args.min_parent}_lr{args.lr}"
exp_name += f"_lsbias{args.lsbias}_itemp{args.init_temp}_ftemp{args.final_temp}_gamma{args.gamma}_bz{args.batch_size}_seed{args.seed}"

np.random.seed(args.seed)

test_mode = 2

# Unit test 1: two node graph
if test_mode == 1:
	models, true_coeff = SCM_ex1()

# Unit test 2: EILLS unit test
if test_mode == 2:
	dim_z = 13
	models = [StructuralCausalModel1(dim_z), StructuralCausalModel2(dim_z)]
	true_coeff = np.array([3, 2, -0.5] + [0] * (dim_z - 4))

# Unit test 3: random generated SCM
if test_mode == 3:
	models, true_coeff, parent_set, child_set, offspring_set = \
		get_linear_SCM(num_vars=args.dim_x + 1, num_envs=args.num_envs, y_index=args.dim_x // 2, 
						min_child=args.min_child, min_parent=args.min_parent, nonlinear_id=5, 
						bias_greater_than=args.lsbias, log=args.log)


xs, ys, yts = sample_from_SCM(models, args.n)

xvs, yvs, yvts = sample_from_SCM(models, args.n // 7 * 3, index=args.dim_x // 2, shuffle=True)
xts, yts, ytts = sample_from_SCM(models, args.n)

valid_x, valid_y = np.concatenate(xvs, 0), np.concatenate(yvs, 0)
test_x, test_y = np.concatenate(xts, 0), np.concatenate(ytts, 0)

niters = args.niters
packs = fair_nn_gumbel_regression(xs, ys, (valid_x, valid_y), (test_x, test_y),
							hyper_gamma=args.gamma, learning_rate=args.lr, 
							niters=niters, batch_size=args.batch_size, init_temp=args.init_temp,
							final_temp=args.final_temp, log=args.log)
