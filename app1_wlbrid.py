from data.model import *
from methods.brute_force import brute_force
from methods.fair_gumbel_one import fair_ll_classification_sgd_gumbel_uni
from methods.tools import pooled_least_squares
from utils import *
import numpy as np
import os
import argparse
import time
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", type=int, default=1234)
parser.add_argument("--n", help="number of samples", type=int, default=1000)
parser.add_argument("--batch_size", help="batch size", type=int, default=36)
parser.add_argument("--dim_x", help="number of explanatory vars", type=int, default=60)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
parser.add_argument('--init_temp', help='initial temperature', type=float, default=5)
parser.add_argument('--final_temp', help='final temperature', type=float, default=0.1)
parser.add_argument('--gamma', help='hyper parameter gamma', type=float, default=36)
parser.add_argument("--log", help="show log", type=bool, default=True)
parser.add_argument("--diter", help="discriminator iters", type=int, default=3)
parser.add_argument("--niter", help="number of interations", type=int, default=50000)
parser.add_argument("--mode", help="mode", type=int, default=1)
parser.add_argument("--s1", help="spurious strength 1", type=float, default=0.95)
parser.add_argument("--s2", help="spurious strength 2", type=float, default=0.75)
parser.add_argument("--st", help="test spurious strength", type=float, default=0.1)
parser.add_argument("--signal", help="singla strength", type=float, default=1)

args = parser.parse_args()

np.random.seed(args.seed)


def logistic_regression(xs, ys, pooled=True):
	if pooled:
		x = np.concatenate(xs, 0)
		y = np.squeeze(np.concatenate(ys, 0))
	else:
		x, y = xs, ys
	model = LogisticRegression(random_state=0).fit(x, np.squeeze(y))
	return model


def accuracy(model, test_data):
	x, y = test_data
	pred = model.predict(x)
	ys = np.squeeze(y)
	return np.mean(pred * ys + (1 - pred) * (1 - ys))

if args.mode == 1:
	models = SCM_class(args.signal, args.s1, args.s2)
	xs, ys = [], []
	for me in models:
		x, y = me.sample(args.n)
		xs.append(x)
		ys.append(y)
	test = ClassificationSCM(args.signal, 0.1)
	xt, yt = test.sample(args.n * 10)
	eval_data = (xt, yt)

	plr = logistic_regression(xs, ys, True)

	rande = ClassificationSCM(args.signal, 0.5)
	xr, yr = rande.sample(args.n)
	rlr = logistic_regression(xr, yr, False)

	print(f'ERM prediction error = {accuracy(plr, eval_data)}')
	print(f'Oracle prediction error = {accuracy(rlr, eval_data)}')

	pack = fair_ll_classification_sgd_gumbel_uni(xs, ys, eval_data, hyper_gamma=args.gamma, learning_rate=args.lr, niters=args.niter, niters_d=args.diter, niters_g=1, 
						anneal_rate=0.993, offset=0, batch_size=args.batch_size, init_temp=args.init_temp, final_temp=args.final_temp, log=args.log)



'''
# load data
dat = np.load('dataset/embeddings.npz')
x1, x2, xt = dat['embeddings_e1'], dat['embeddings_e2'], dat['embeddings_t']
y1, y2, yt = dat['labels_e1'], dat['labels_e2'], dat['labels_t']

n1, n2, nt = np.shape(x1)[0], np.shape(x2)[0], np.shape(xt)[0]
y1, y2, yt = np.reshape(y1, (n1, 1)), np.reshape(y2, (n2, 1)), np.reshape(yt, (nt, 1))

#print(np.shape(y1), np.shape(y2))
xs = [x1, x2]
ys = [y1, y2]

#print(np.mean(y1 == 0), np.mean(y2), np.mean(yt))

nt1 = nt * 3 // 10
nt2 = nt - nt1
#xt1, yt1 = xt[:nt1, :], yt[:nt1,:]
#xt2, yt2 = xt[nt1:, :], yt[nt1:,:]
#eval_data = (xt1, yt1, xt2, yt2)
eval_data = (xt, yt)

#print(np.shape(xt1), np.shape(xt2))

#print(np.shape(y1), np.shape(y2))
from sklearn.linear_model import LogisticRegression

# env1 -> env2 acc:
clf1 = LogisticRegression(random_state=0).fit(x1, np.squeeze(y1))
print(f'env1 -> test acc = {np.mean(clf1.predict(xt) * np.squeeze(yt) + (1-clf1.predict(xt)) * (1 - np.squeeze(yt)))}')

# env1 -> env2 acc:
clf2 = LogisticRegression(random_state=0).fit(x2, np.squeeze(y2))
print(f'env2 -> test acc = {np.mean(clf2.predict(xt) * np.squeeze(yt) + (1-clf2.predict(xt)) * (1 - np.squeeze(yt)))}')

pack = fair_ll_classification_sgd_gumbel_uni(xs, ys, eval_data, hyper_gamma=args.gamma, learning_rate=args.lr, niters=args.niter, niters_d=args.diter, niters_g=1, 
						anneal_rate=0.993, offset=-3, batch_size=args.batch_size, init_temp=args.init_temp, final_temp=args.final_temp, log=args.log)
'''