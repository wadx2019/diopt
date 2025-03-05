import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import PowerAllocationOptimizationProblem

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def G_ineq(n_var):
    array = np.zeros((2*n_var+1, n_var))
    array[:n_var, :n_var] = np.eye(n_var)
    array[n_var:2*n_var, :n_var] = -np.eye(n_var)
    array[-1, :] = 1
    return array

def h_ineq(P_t,n_var):
    array = np.zeros(2*n_var+1)
    array[:n_var] = P_t
    array[-1] = P_t
    return array

torch.set_default_dtype(torch.float64)
n_var = 20
n_example = 10000
P_t = 5
G_m = np.random.uniform(0, 50, size=(n_example, n_var))
# G_m[:, :n_var//2] += 13
# G_m[:, n_var//2:] *= 0.1
G = G_ineq(n_var)
h = h_ineq(P_t, n_var)

# water filling(choice)
lambda_m = np.random.uniform(0, 1, size=(n_example, n_var))
# P_t = np.random.uniform(0, 10, n_example)
# lagrange(choice)
mu = np.random.uniform(0, 1, size=(n_example, n_var))
# X = np.column_stack((G_m, P_t))
# G = np.ones(n_var)

problem = PowerAllocationOptimizationProblem(G_m, P_t, G, h, lambda_m, mu)
_,t=problem.calc_Y()
print(t)
print(len(problem.Y))

with open("./random_power_dataset_var{}_ex{}".format(n_var, n_example), 'wb') as f:
    pickle.dump(problem, f)
