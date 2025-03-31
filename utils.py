import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float64)
import torch.functional as F

import numpy as np
import osqp
from qpth.qp import QPFunction
import ipopt
from scipy.linalg import svd
from scipy.sparse import csc_matrix

import hashlib
from copy import deepcopy
import scipy.io as spio
import time

from pypower.api import case57
from pypower.api import opf, makeYbus
from pypower import idx_bus, idx_gen, ppoption

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{value} is not a valid boolean value')

def my_hash(string):
    return hashlib.sha1(bytes(string, 'utf-8')).hexdigest()


###################################################################
# SIMPLE PROBLEM
###################################################################

class SimpleProblem:
    """ 
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
    """
    def __init__(self, Q, p, A, G, h, X, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._G = torch.tensor(G)
        self._h = torch.tensor(h)
        self._X = torch.tensor(X)
        self._Y = None
        self._xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = G.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        det = 0
        i = 0
        while abs(det) < 0.0001 and i < 100:
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
            det = torch.det(self._A[:, self._other_vars])
            i += 1
        if i == 100:
            raise Exception
        else:
            self._A_partial = self._A[:, self._partial_vars]
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])

        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'SimpleProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def set_w(self, wc, wo):
        self.wc = wc
        self.wo = wo

    def set_buffer(self, buffer):
        self.buffer = buffer

    @torch.no_grad()
    def _eval_func(self, X, Y_partial, Y_best=None, obj_best=None, idx=None, extra=False, extra2=False,
                   return_com=False):
        bonus0 = (Y_partial - self.buffer.get(idx)).norm(dim=1, keepdim=True)
        # bonus = (Y_partial - Y_best).norm(dim=1, keepdim=True)
        Y_partial = Y_partial * 0.5 + 0.5
        Y = self.complete_partial(X, Y_partial)
        resids = self.ineq_resid(X, Y)
        # obj_best = torch.min(obj_best,
        #                      torch.where(self.buffer.viol[idx] >= 0, self.obj_fn(self.buffer.Y_com[idx]).view(-1, 1),
        #                                  obj_best))
        gap = obj_best - self.obj_fn(Y).view(-1, 1)
        dist = torch.clamp(resids, -0.00).sum(dim=1, keepdim=True)
        # dist = resids.max(dim=1, keepdim=True)[0]
        judge = (resids.max(dim=1, keepdim=True)[0] <= 1e-5)
        if return_com:
            return 0.0 * bonus0 * judge * (not extra) * extra2 + 0.1 * torch.exp(
                torch.clamp(gap, -np.infty, 2)) * judge * extra - dist, Y
        else:
            return 0.0 * bonus0 * judge * (not extra) * extra2 + 0.1 * torch.exp(
                torch.clamp(gap, -np.infty,
                            2)) * judge * extra - dist  # + self.wc * judge + self.wo * judge * torch.sigmoid(-self.obj_fn(Y)).view(-1,

    # + self.wc * judge + self.wo * judge * torch.sigmoid(-self.obj_fn(Y)).view(-1,

    @torch.no_grad()  # 1)  # self.wc * torch.log(-resids.sum(dim=1, keepdim=True) * judge + 1) - self.wo * judge * self.obj_fn(Y).view(-1,1)
    def _eval_func_eval(self, X, Y_partial, return_com=False):
        if not return_com:

            Y_partial = Y_partial * 0.5 + 0.5
            Y = self.complete_partial(X, Y_partial)
        else:
            Y = Y_partial
        resids = self.ineq_resid(X, Y)
        dist = torch.clamp(resids, -0.0).sum(dim=1, keepdim=True)
        judge = (dist <= 1e-5)

        return -dist + judge * torch.sigmoid(-self.obj_fn(Y).view(-1,
                                                                  1))  # + self.wc * torch.log(-resids.sum(dim=1, keepdim=True) * judge + 1) - self.wo * judge * self.obj_fn(Y).view(-1,1)

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def h(self):
        return self._h

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num*self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def obj_fn(self, Y):
        return (0.5*(Y@self.Q)*Y + self.p*Y).sum(dim=1)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_dist@self.G

    def ineq_partial_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T
        grad = 2 * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    def unnorm(self, Y):
        return Y
    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Z = 2 * Z -1
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    def opt_solve(self, X, solver_type='osqp', tol=1e-4):

        if solver_type == 'qpth':
            print('running qpth')
            start_time = time.time()
            res = QPFunction(eps=tol, verbose=False)(self.Q, self.p, self.G, self.h, self.A, X)
            end_time = time.time()

            sols = np.array(res.detach().cpu().numpy())
            total_time = end_time - start_time
            parallel_time = total_time
        
        elif solver_type == 'osqp':
            print('running osqp')
            Q, p, A, G, h = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            for Xi in X_np:
                solver = osqp.OSQP()
                my_A = np.vstack([A, G])
                my_l = np.hstack([Xi, -np.ones(h.shape[0]) * np.inf])
                my_u = np.hstack([Xi, h])
                solver.setup(P=csc_matrix(Q), q=p, A=csc_matrix(my_A), l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += (end_time - start_time)
                if results.info.status == 'solved':
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time/len(X_np)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)  
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y


###################################################################
# NONCONVEX PROBLEM
###################################################################

class NonconvexProblem:
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
    """
    def __init__(self, Q, p, A, G, h, X, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._G = torch.tensor(G)
        self._h = torch.tensor(h)
        self._X = torch.tensor(X)
        self._Y = None
        self._xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = G.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        det = 0
        i = 0
        while abs(det) < 0.0001 and i < 100:
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
            det = torch.det(self._A[:, self._other_vars])
            i += 1
        if i == 100:
            raise Exception
        else:
            self._A_partial = self._A[:, self._partial_vars]
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])
            self._M = 2 * (self.G[:, self.partial_vars] -
                            self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial))

        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'NonconvexProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def set_w(self, wc, wo):
        self.wc = wc
        self.wo = wo

    def set_buffer(self, buffer):
        self.buffer = buffer

    @torch.no_grad()
    def _eval_func(self, X, Y_partial, Y_best=None, obj_best=None, idx=None, extra=False, extra2=False,
                   return_com=False):
        bonus0 = (Y_partial - self.buffer.get(idx)).norm(dim=1, keepdim=True)
        # bonus = (Y_partial - Y_best).norm(dim=1, keepdim=True)
        Y_partial = Y_partial * 0.5 + 0.5
        Y = self.complete_partial(X, Y_partial)
        resids = self.ineq_resid(X, Y)
        obj_best = torch.min(obj_best,
                             torch.where(self.buffer.viol[idx] >= 0, self.obj_fn(self.buffer.Y_com[idx]).view(-1, 1),
                                         obj_best))
        gap = obj_best - self.obj_fn(Y).view(-1, 1)
        dist = torch.clamp(resids, -0.00).sum(dim=1, keepdim=True)
        # dist = resids.max(dim=1, keepdim=True)[0]
        judge = (resids.max(dim=1, keepdim=True)[0] <= 1e-5)
        if return_com:
            return 0.0 * bonus0 * judge * (not extra) * extra2 + 0.1 * torch.exp(
                torch.clamp(gap, -np.infty, 2)) * judge * extra - dist, Y
        else:
            return 0.0 * bonus0 * judge * (not extra) * extra2 + 0.1 * torch.exp(
                torch.clamp(gap, -np.infty,
                            2)) * judge * extra - dist  # + self.wc * judge + self.wo * judge * torch.sigmoid(-self.obj_fn(Y)).view(-1,

    # + self.wc * judge + self.wo * judge * torch.sigmoid(-self.obj_fn(Y)).view(-1,

    @torch.no_grad()  # 1)  # self.wc * torch.log(-resids.sum(dim=1, keepdim=True) * judge + 1) - self.wo * judge * self.obj_fn(Y).view(-1,1)
    def _eval_func_eval(self, X, Y_partial, completed=False):
        if not completed:
            Y_partial = Y_partial * 0.5 + 0.5
            Y = self.complete_partial(X, Y_partial)
        else:
            Y = Y_partial
        resids = self.ineq_resid(X, Y)
        dist = torch.clamp(resids, -0.0).sum(dim=1, keepdim=True)
        judge = (dist <= 1e-5)

        return -dist + judge * torch.sigmoid(-self.obj_fn(Y).view(-1,
                                                                  1))  # + self.wc * torch.log(-resids.sum(dim=1, keepdim=True) * judge + 1) - self.wo * judge * self.obj_fn(Y).view(-1,1)


    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def h(self):
        return self._h

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num*self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def obj_fn(self, Y):
        return (0.5*(Y@self.Q)*Y + 10* self.p*torch.sin(Y)).sum(dim=1)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        return 2 * torch.clamp(Y@self.G.T - self.h, 0) @ self.G

    def ineq_partial_grad(self, X, Y):
        grad = torch.clamp(Y@self.G.T - self.h, 0) @ self._M
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    def unnorm(self, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Z = 2*Z-1
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    def opt_solve(self, X, solver_type='ipopt', tol=1e-4):
        Q, p, A, G, h = self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np
        X_np = X.detach().cpu().numpy()
        Y = []
        total_time = 0
        for Xi in X_np:
            if solver_type == 'ipopt':
                y0 = np.linalg.pinv(A)@Xi  # feasible initial point

                # upper and lower bounds on variables
                lb = -np.infty * np.ones(y0.shape)
                ub = np.infty * np.ones(y0.shape)

                # upper and lower bounds on constraints
                cl = np.hstack([Xi, -np.inf * np.ones(G.shape[0])])
                cu = np.hstack([Xi, h])

                nlp = ipopt.problem(
                            n=len(y0),
                            m=len(cl),
                            problem_obj=nonconvex_ipopt(Q, p, A, G),
                            lb=lb,
                            ub=ub,
                            cl=cl,
                            cu=cu
                            )

                nlp.addOption('tol', tol)
                nlp.addOption('print_level', 0) # 3)

                start_time = time.time()
                y, info = nlp.solve(y0)
                end_time = time.time()
                Y.append(y)
                total_time += (end_time - start_time)
            else:
                raise NotImplementedError

        return np.array(Y), total_time, total_time/len(X_np)

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y

class nonconvex_ipopt(object):
    def __init__(self, Q, p, A, G):
        self.Q = Q
        self.p = p
        self.A = A
        self.G = G
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p@np.sin(y)

    def gradient(self, y):
        return self.Q@y + (self.p * np.cos(y))

    def constraints(self, y):
        return np.hstack([self.A@y, self.G@y])

    def jacobian(self, y):
        return np.concatenate([self.A.flatten(), self.G.flatten()])

    # # Don't use: In general, more efficient with numerical approx
    # def hessian(self, y, lagrange, obj_factor):
    #     H = obj_factor * (self.Q - np.diag(self.p * np.sin(y)) )
    #     return H[self.tril_indices]

    # def intermediate(self, alg_mod, iter_count, obj_value,
    #         inf_pr, inf_du, mu, d_norm, regularization_size,
    #         alpha_du, alpha_pr, ls_trials):
    #     print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))



###################################################################
# ACOPF
###################################################################


CASE_FNS = dict([(57, case57)])

class ACOPFProblem:
    """
        minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
        s.t.                  p_g min   <= p_g  <= p_g max
                              q_g min   <= q_g  <= q_g max
                              vmag min  <= vmag <= vmag max
                              vang_slack = \theta_slack   # voltage angle     
                              (p_g - p_d) + (q_g - q_d)i = diag(vmag e^{i*vang}) conj(Y) (vmag e^{-i*vang})
    """

    def __init__(self, filename, valid_frac=0.0833, test_frac=0.0833):
        data = spio.loadmat(filename)
        self.nbus = int(filename.split('_')[-1][4:-4])

        ## Define useful power network quantities and indices
        ppc = CASE_FNS[self.nbus]()
        self.ppc = ppc

        self.genbase = ppc['gen'][:, idx_gen.MBASE]
        self.baseMVA = ppc['baseMVA']

        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))

        # indices within gens
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])

        self.ng = ppc['gen'].shape[0]
        self.nslack = len(self.slack)
        self.npv = len(self.pv)

        self.quad_costs = torch.tensor(ppc['gencost'][:,4], dtype=torch.get_default_dtype())
        self.lin_costs  = torch.tensor(ppc['gencost'][:,5], dtype=torch.get_default_dtype())
        self.const_cost = ppc['gencost'][:,6].sum()

        self.pmax = torch.tensor(ppc['gen'][:,idx_gen.PMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.pmin = torch.tensor(ppc['gen'][:,idx_gen.PMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.qmax = torch.tensor(ppc['gen'][:,idx_gen.QMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.qmin = torch.tensor(ppc['gen'][:,idx_gen.QMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.vmax = torch.tensor(ppc['bus'][:,idx_bus.VMAX], dtype=torch.get_default_dtype())
        self.vmin = torch.tensor(ppc['bus'][:,idx_bus.VMIN], dtype=torch.get_default_dtype())
        self.slackva = torch.tensor([np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])], 
            dtype=torch.get_default_dtype()).squeeze(-1)

        ppc2 = deepcopy(ppc)
        ppc2['bus'][:,0] -= 1
        ppc2['branch'][:,[0,1]] -= 1
        Ybus, _, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus.todense()
        self.Ybusr = torch.tensor(np.real(Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(Ybus), dtype=torch.get_default_dtype())

        ## Define optimization problem input and output variables
        demand = data['Dem'].T / self.baseMVA
        gen =  data['Gen'].T / self.genbase
        voltage = data['Vol'].T

        X = np.concatenate([np.real(demand), np.imag(demand)], axis=1)
        Y = np.concatenate([np.real(gen), np.imag(gen), np.abs(voltage), np.angle(voltage)], axis=1)
        feas_mask =  ~np.isnan(Y).any(axis=1)

        self._X = torch.tensor(X[feas_mask], dtype=torch.get_default_dtype())
        self._Y = torch.tensor(Y[feas_mask], dtype=torch.get_default_dtype())
        self._xdim = X.shape[1]
        self._ydim = Y.shape[1]
        self._num = feas_mask.sum()

        self._neq = 2*self.nbus
        self._nineq = 4*self.ng + 2*self.nbus
        self._nknowns = self.nslack

        # indices of useful quantities in full solution
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2*self.ng
        self.va_start_yidx = 2*self.ng + self.nbus


        ## Keep parameters indicating how data was generated
        self.EPS_INTERIOR = data['EPS_INTERIOR'][0][0]
        self.CorrCoeff = data['CorrCoeff'][0][0]
        self.MaxChangeLoad = data['MaxChangeLoad'][0][0]


        ## Define train/valid/test split
        self._valid_frac = valid_frac
        self._test_frac = test_frac


        ## Define variables and indices for "partial completion" neural network

        # pg (non-slack) and |v|_g (including slack)
        self._partial_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv, self.va_start_yidx + self.slack])
        self._other_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)
        self._partial_unknown_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv])

        # initial values for solver
        self.vm_init = ppc['bus'][:, idx_bus.VM]
        self.va_init = np.deg2rad(ppc['bus'][:, idx_bus.VA])
        self.pg_init = ppc['gen'][:, idx_gen.PG] / self.genbase
        self.qg_init = ppc['gen'][:, idx_gen.QG] / self.genbase

        # voltage angle at slack buses (known)
        self.slack_va = self.va_init[self.slack]

        # indices of useful quantities in partial solution
        self.pg_pv_zidx = np.arange(self.npv)
        self.vm_spv_zidx = np.arange(self.npv, 2*self.npv + self.nslack)

        # useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nbus

        ### For Pytorch
        self._device = None

    def set_w(self, wc, wo):
        self.wc = wc
        self.wo = wo

    def set_buffer(self, buffer):
        self.buffer = buffer

    @torch.no_grad()
    def _eval_func(self, X, Y_partial, Y_best=None, obj_best=None, idx=None, extra=False, extra2=False,
                   return_com=False):
        bonus0 = (Y_partial - self.buffer.get(idx)).norm(dim=1, keepdim=True)
        # bonus = (Y_partial - Y_best).norm(dim=1, keepdim=True)
        Y_partial = Y_partial * 0.5 + 0.5
        Y = self.complete_partial(X, Y_partial)
        resids = self.ineq_resid(X, Y)
        # obj_best = torch.min(obj_best,
        #                      torch.where(self.buffer.viol[idx] >= 0, self.obj_fn(self.buffer.Y_com[idx]).view(-1, 1),
        #                                  obj_best))
        gap = obj_best - self.obj_fn(Y).view(-1, 1)
        dist = torch.clamp(resids, -0.00).sum(dim=1, keepdim=True)
        # dist = resids.max(dim=1, keepdim=True)[0]
        judge = (resids.max(dim=1, keepdim=True)[0] <= 1e-5)
        if return_com:
            return 0.0 * bonus0 * judge * (not extra) * extra2 + 0.1 * torch.exp(
                torch.clamp(gap, -np.infty, 2)) * judge * extra - dist, Y
        else:
            return 0.0 * bonus0 * judge * (not extra) * extra2 + 0.1 * torch.exp(
                torch.clamp(gap, -np.infty,
                            2)) * judge * extra - dist  # + self.wc * judge + self.wo * judge * torch.sigmoid(-self.obj_fn(Y)).view(-1,

    @torch.no_grad()  # 1)  # self.wc * torch.log(-resids.sum(dim=1, keepdim=True) * judge + 1) - self.wo * judge * self.obj_fn(Y).view(-1,1)
    def _eval_func_eval(self, X, Y_partial, return_com=False):
        if not return_com:

            Y_partial = Y_partial * 0.5 + 0.5
            Y = self.complete_partial(X, Y_partial)
        else:
            Y = Y_partial
        resids = self.ineq_resid(X, Y)
        dist = torch.clamp(resids, -0.0).sum(dim=1, keepdim=True)
        judge = (dist <= 1e-5)

        return -dist + judge * torch.sigmoid(-self.obj_fn(Y).view(-1,
                                                                  1))  # + self.wc * torch.log(-resids.sum(dim=1, keepdim=True) * judge + 1) - self.wo * judge * self.obj_fn(Y).view(-1,1)

    def _cons_region(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return resids <= 0

    def __str__(self):
        return 'ACOPF-{}-{}-{}-{}-{}-{}'.format(
            self.nbus,
            self.EPS_INTERIOR, self.CorrCoeff, self.MaxChangeLoad,
            self.valid_frac, self.test_frac)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_unknown_vars

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num * self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num * self.train_frac):int(self.num * (self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num * (self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def get_yvars(self, Y):
        pg = Y[:, :self.ng]
        qg = Y[:, self.ng:2*self.ng]
        vm = Y[:, -2*self.nbus:-self.nbus]
        va = Y[:, -self.nbus:]
        return pg, qg, vm, va

    def obj_fn(self, Y):
        pg, _, _, _ = self.get_yvars(Y)
        pg_mw = pg * torch.tensor(self.genbase).to(self.device)
        cost = (self.quad_costs * pg_mw**2).sum(axis=1) + \
            (self.lin_costs * pg_mw).sum(axis=1) + \
            self.const_cost
        return cost / (self.genbase.mean() ** 2)

    def eq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        ## power balance equations
        tmp1 = vr@self.Ybusr - vi@self.Ybusi
        tmp2 = -vr@self.Ybusi - vi@self.Ybusr

        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=self.device)
        pg_expand[:, self.spv] = pg
        real_resid = (pg_expand - X[:, :self.nbus]) - (vr*tmp1 - vi*tmp2)

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=self.device)
        qg_expand[:, self.spv] = qg
        react_resid = (qg_expand - X[:, self.nbus:]) - (vr*tmp2 + vi*tmp1)

        ## all residuals
        resids = torch.cat([
            real_resid,
            react_resid
        ], dim=1)
        
        return resids

    def ineq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        resids = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            vm - self.vmax,
            self.vmin - vm
        ], dim=1)
        return resids

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        eq_resid = self.eq_resid(X,Y)
        return 2*eq_jac.transpose(1,2).bmm(eq_resid.unsqueeze(-1)).squeeze(-1)

    def ineq_grad(self, X, Y):
        ineq_jac = self.ineq_jac(Y)
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_jac.transpose(1,2).bmm(ineq_dist.unsqueeze(-1)).squeeze(-1)

    def ineq_partial_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]).bmm(eq_jac[:, :, self.partial_vars])

        direct_grad = self.ineq_grad(X, Y)
        indirect_partial_grad = dynz_dz.transpose(1,2).bmm(
            direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

        full_partial_grad = indirect_partial_grad + direct_grad[:, self.partial_vars]

        full_grad = torch.zeros(X.shape[0], self.ydim, device=self.device)
        full_grad[:, self.partial_vars] = full_partial_grad
        full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_grad.unsqueeze(-1)).squeeze(-1)

        return full_grad

    def eq_jac(self, Y):
        _, _, vm, va = self.get_yvars(Y)

        # helper functions
        mdiag = lambda v1, v2: torch.diag_embed(v1).bmm(torch.diag_embed(v2))
        Ydiagv = lambda Y, v: Y.unsqueeze(0).expand(v.shape[0], *Y.shape).bmm(torch.diag_embed(v))
        dtm = lambda v, M: torch.diag_embed(v).bmm(M)

        # helper quantities
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        Yr = self.Ybusr
        Yi = self.Ybusi
        YrvrYivi = vr@Yr - vi@Yi
        YivrYrvi = vr@Yi + vi@Yr

        # real power equations
        dreal_dpg = torch.zeros(self.nbus, self.ng, device=self.device) 
        dreal_dpg[self.spv, :] = torch.eye(self.ng, device=self.device)
        dreal_dvm = -mdiag(cosva, YrvrYivi) - dtm(vr, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva)) \
            -mdiag(sinva, YivrYrvi) - dtm(vi, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva))
        dreal_dva = -mdiag(-vi, YrvrYivi) - dtm(vr, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr)) \
            -mdiag(vr, YivrYrvi) - dtm(vi, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr))
        
        # reactive power equations
        dreact_dqg = torch.zeros(self.nbus, self.ng, device=self.device)
        dreact_dqg[self.spv, :] = torch.eye(self.ng, device=self.device)
        dreact_dvm = mdiag(cosva, YivrYrvi) + dtm(vr, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva)) \
            -mdiag(sinva, YrvrYivi) - dtm(vi, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva))
        dreact_dva = mdiag(-vi, YivrYrvi) + dtm(vr, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr)) \
            -mdiag(vr, YrvrYivi) - dtm(vi, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr))

        jac = torch.cat([
            torch.cat([dreal_dpg.unsqueeze(0).expand(vr.shape[0], *dreal_dpg.shape), 
                torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreal_dvm, dreal_dva], dim=2),
            torch.cat([torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreact_dqg.unsqueeze(0).expand(vr.shape[0], *dreact_dqg.shape),
                dreact_dvm, dreact_dva], dim=2)],
            dim=1)

        return jac


    def ineq_jac(self, Y):
        jac = torch.cat([
            torch.cat([torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([-torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device),
                torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device), 
                -torch.eye(self.ng, device=self.device),
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=self.device),
                torch.zeros(self.nbus, self.ng, device=self.device), 
                torch.eye(self.nbus, device=self.device), 
                torch.zeros(self.nbus, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=self.device), 
                torch.zeros(self.nbus, self.ng, device=self.device),
                -torch.eye(self.nbus, device=self.device), 
                torch.zeros(self.nbus, self.nbus, device=self.device)], dim=1)
            ], dim=0)
        return jac.unsqueeze(0).expand(Y.shape[0], *jac.shape)

    # Processes intermediate neural network output
    def process_output(self, X, out):
        out2 = nn.Sigmoid()(out[:, :-self.nbus+self.nslack])
        pg = out2[:, :self.qg_start_yidx] * self.pmax + (1-out2[:, :self.qg_start_yidx]) * self.pmin
        qg = out2[:, self.qg_start_yidx:self.vm_start_yidx] * self.qmax + \
            (1-out2[:, self.qg_start_yidx:self.vm_start_yidx]) * self.qmin
        vm = out2[:, self.vm_start_yidx:] * self.vmax + (1- out2[:, self.vm_start_yidx:]) * self.vmin

        va = torch.zeros(X.shape[0], self.nbus, device=self.device)
        va[:, self.nonslack_idxes] = out[:, self.va_start_yidx:]
        va[:, self.slack] = torch.tensor(self.slack_va, device=self.device).unsqueeze(0).expand(X.shape[0], self.nslack)

        return torch.cat([pg, qg, vm, va], dim=1)

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y_partial = torch.zeros(Z.shape, device=self.device)

        # Re-scale real powers
        Y_partial[:, self.pg_pv_zidx] = Z[:, self.pg_pv_zidx] * self.pmax[1:] + \
             (1-Z[:, self.pg_pv_zidx]) * self.pmin[1:]
        
        # Re-scale real parts of voltages
        Y_partial[:, self.vm_spv_zidx] = Z[:, self.vm_spv_zidx] * self.vmax[self.spv] + \
            (1-Z[:, self.vm_spv_zidx]) * self.vmin[self.spv]

        return PFFunction(self)(X, Y_partial)

    def unnorm(self, Y_partial):
        Y_partial[:, self.pg_pv_zidx] = (Y_partial[:, self.pg_pv_zidx] - self.pmin[1:]) / (self.pmax[1:] - self.pmin[1:])
        Y_partial[:, self.vm_spv_zidx] = (Y_partial[:, self.vm_spv_zidx] - self.vmin[self.spv]) / (self.vmax[self.spv] - self.vmin[self.spv])
        Y_partial = (Y_partial - 0.5) * 2
        return Y_partial

    def opt_solve(self, X, solver_type='pypower', tol=1e-4):
        X_np = X.detach().cpu().numpy()

        ppc = self.ppc

        # Set reduced voltage bounds if applicable
        ppc['bus'][:,idx_bus.VMIN] = ppc['bus'][:,idx_bus.VMIN] + self.EPS_INTERIOR
        ppc['bus'][:,idx_bus.VMAX] = ppc['bus'][:,idx_bus.VMAX] - self.EPS_INTERIOR

        # Solver options
        ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol)  # MIPS PDIPM

        Y = []
        total_time = 0
        for i in range(X_np.shape[0]):
            print(i)
            ppc['bus'][:, idx_bus.PD] = X_np[i, :self.nbus] * self.baseMVA
            ppc['bus'][:, idx_bus.QD] = X_np[i, self.nbus:] * self.baseMVA

            start_time = time.time()
            my_result = opf(ppc, ppopt)
            end_time = time.time()
            total_time += (end_time - start_time)

            pg = my_result['gen'][:, idx_gen.PG] / self.genbase
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))

        return np.array(Y), total_time, total_time/len(X_np)


def PFFunction(data, tol=1e-3, bsz=200, max_iters=5):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):

            ## Step 1: Newton's method
            Y = torch.zeros(X.shape[0], data.ydim, device=DEVICE)
            
            # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
            Y[:, data.pg_start_yidx + data.pv_] = Z[:, data.pg_pv_zidx]    # pg at non-slack gens
            Y[:, data.vm_start_yidx + data.spv] = Z[:, data.vm_spv_zidx]   # vm at gens
            Y[:, data.va_start_yidx + data.slack] = torch.tensor(data.slack_va, device=DEVICE)  # va at slack bus

            # init guesses for remaining values
            Y[:, data.vm_start_yidx + data.pq] = torch.tensor(data.vm_init[data.pq], device=DEVICE)  # vm at load buses
            Y[:, data.va_start_yidx + data.pv] = torch.tensor(data.va_init[data.pv], device=DEVICE)  # va at non-slack gens 
            Y[:, data.va_start_yidx + data.pq] = torch.tensor(data.va_init[data.pq], device=DEVICE)  # va at load buses
            Y[:, data.qg_start_yidx:data.qg_start_yidx+data.ng] = 0    # qg at gens (not used in Newton upd)
            Y[:, data.pg_start_yidx+data.slack_] = 0                   # pg at slack (not used in Newton upd)

            keep_constr = np.concatenate([
                data.pflow_start_eqidx + data.pv,     # real power flow at non-slack gens
                data.pflow_start_eqidx + data.pq,     # real power flow at load buses
                data.qflow_start_eqidx + data.pq])    # reactive power flow at load buses
            newton_guess_inds = np.concatenate([             
                data.vm_start_yidx + data.pq,         # vm at load buses
                data.va_start_yidx + data.pv,         # va at non-slack gens
                data.va_start_yidx + data.pq])        # va at load buses

            converged = torch.zeros(X.shape[0])
            jacs = []
            newton_jacs_inv = []
            for b in range(0, X.shape[0], bsz):
                # print('batch: {}'.format(b))
                X_b = X[b:b+bsz]
                Y_b = Y[b:b+bsz]

                for i in range(max_iters):
                    # print(i)
                    gy = data.eq_resid(X_b, Y_b)[:, keep_constr]
                    jac_full = data.eq_jac(Y_b)
                    jac = jac_full[:, keep_constr, :]
                    newton_jac_inv = torch.inverse(jac[:, :, newton_guess_inds])
                    delta = newton_jac_inv.bmm(gy.unsqueeze(-1)).squeeze(-1)
                    Y_b[:, newton_guess_inds] -= delta
                    if torch.norm(delta, dim=1).abs().max() < tol:
                        break

                converged[b:b+bsz] = (delta.abs() < tol).all(dim=1)
                jacs.append(jac_full)
                newton_jacs_inv.append(newton_jac_inv)


            ## Step 2: Solve for remaining variables

            # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.qflow_start_eqidx + data.spv]
            # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
            Y[:, data.pg_start_yidx + data.slack_] = \
                -data.eq_resid(X, Y)[:, data.pflow_start_eqidx + data.slack]

            ctx.data = data
            ctx.save_for_backward(torch.cat(jacs), torch.cat(newton_jacs_inv),
                torch.tensor(newton_guess_inds, device=DEVICE),
                torch.tensor(keep_constr, device=DEVICE))

            return Y

        @staticmethod
        def backward(ctx, dl_dy):

            data = ctx.data
            jac, newton_jac_inv, newton_guess_inds, keep_constr = ctx.saved_tensors

            ## Step 2 (calc pg at slack and qg at gens)

            # gradient of all voltages through step 3 outputs
            last_eqs = np.concatenate([data.pflow_start_eqidx + data.slack, data.qflow_start_eqidx + data.spv])
            last_vars = np.concatenate([
                data.pg_start_yidx + data.slack_, np.arange(data.qg_start_yidx, data.qg_start_yidx + data.ng)])
            jac3 = jac[:, last_eqs, :]
            dl_dvmva_3 = -jac3[:, :, data.vm_start_yidx:].transpose(1,2).bmm(
                dl_dy[:, last_vars].unsqueeze(-1)).squeeze(-1)

            # gradient of pd at slack and qd at gens through step 3 outputs
            dl_dpdqd_3 = dl_dy[:, last_vars]

            # insert into correct places in x and y loss vectors
            dl_dy_3 = torch.zeros(dl_dy.shape, device=DEVICE)
            dl_dy_3[:, data.vm_start_yidx:] = dl_dvmva_3

            dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=DEVICE)
            dl_dx_3[:, np.concatenate([data.slack, data.nbus + data.spv])] = dl_dpdqd_3


            ## Step 1
            dl_dy_total = dl_dy_3 + dl_dy  # Backward pass vector including result of last step

            # Use precomputed inverse jacobian
            jac2 = jac[:, keep_constr, :]
            d_int = newton_jac_inv.transpose(1,2).bmm(
                            dl_dy_total[:,newton_guess_inds].unsqueeze(-1)).squeeze(-1)

            dl_dz_2 = torch.zeros(dl_dy.shape[0], data.npv + data.ng, device=DEVICE)
            dl_dz_2[:, data.pg_pv_zidx] = -d_int[:, :data.npv]  # dl_dpg at pv buses
            dl_dz_2[:, data.vm_spv_zidx] = -jac2[:, :, data.vm_start_yidx + data.spv].transpose(1,2).bmm(
                d_int.unsqueeze(-1)).squeeze(-1)

            dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=DEVICE)
            dl_dx_2[:, data.pv] = d_int[:, :data.npv]                       # dl_dpd at pv buses
            dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv+len(data.pq)]  # dl_dpd at pq buses
            dl_dx_2[:, data.nbus + data.pq] = d_int[:, -len(data.pq):]      # dl_dqd at pq buses


            # Final quantities
            dl_dx_total = dl_dx_3 + dl_dx_2
            dl_dz_total = dl_dz_2 + dl_dy_total[:, np.concatenate([
                data.pg_start_yidx + data.pv_, data.vm_start_yidx + data.spv])]

            return dl_dx_total, dl_dz_total


    return PFFunctionFn.apply

###############################################
#####   RETARGETING PRO########################
###############################################

import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float64)
import torch.functional as F

import numpy as np
import osqp
from qpth.qp import QPFunction
import ipopt
import cyipopt
from scipy.linalg import svd
from scipy.sparse import csc_matrix

import hashlib
from copy import deepcopy
import scipy.io as spio
import time

from pypower.api import case57
from pypower.api import opf, makeYbus
from pypower import idx_bus, idx_gen, ppoption

import pandas as pd
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")




def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{value} is not a valid boolean value')

def my_hash(string):
    return hashlib.sha1(bytes(string, 'utf-8')).hexdigest()

###############################################
#####   RETARGETING PRO########################
###############################################

import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float64)

import numpy as np
import osqp
from qpth.qp import QPFunction
import ipopt
import cyipopt
from scipy.sparse import csc_matrix

import hashlib
from copy import deepcopy
import scipy.io as spio
import time

from pypower.api import opf, makeYbus
from pypower import idx_bus, idx_gen, ppoption

from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Retargeting_h1:
    def __init__(self, R, R_root, R_root_trans, h1_joint_pick_idx):
        self.R = torch.tensor(R)
        self.R_root = torch.tensor(R_root)
        self.R_root_trans = torch.tensor(R_root_trans)
        self.h1_joint_pick_idx = h1_joint_pick_idx
        self.frames = R_root.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward_kinematics(self, Y):
        h1_fk = Humanoid_Batch(device=self.device)
        # print('1', self.R_root[None, :, None].shape)
        # print('2', (self.R * Y).shape)
        # print('4', torch.zeros((1, self.frames, 2, 3)).shape)
        pose_aa_h1 = torch.cat([ self.R_root[None, :, None].to(self.device), (self.R * Y).to(self.device), torch.zeros((1, self.frames, 2, 3), device=self.device),], axis=2,).to(self.device)

        return h1_fk.fk_batch(pose_aa_h1.to(torch.float32), self.R_root_trans[None,].to(torch.float32).to(self.device))

    # def loss(self, X, Y):
    #     loss = (self.forward_kinematics(Y)["global_translation_extend"][:, :,
    #             self.h1_joint_pick_idx] - X.to(self.device))
    #     loss = loss.norm(dim=-1).sum(dim=-1).view(-1,1)
    #     num = loss.numel()
    #     if num == 1:
    #         loss.backward()
    #         return loss, Y.grad
    #     else:
    #         return loss, None

    def loss(self, X, Y):
        loss = (self.forward_kinematics(Y)["global_translation_extend"][:, :,
                self.h1_joint_pick_idx] - X.to(self.device))
        loss = loss.norm(dim=-1).sum(dim=-1).view(-1,1)

        return loss

    def grad(self, X, Y):
        if Y.grad is not None:
            Y.grad.zero_()
        loss = self.loss(X, Y)
        loss.backward()
        return Y.grad


class RetargetingProblem:
    def __init__(self, filename, valid_frac=0.0833, test_frac=0.0833):
        # self._X = torch.tensor(X)
        # self._G = torch.tensor(G)
        # self._h = torch.tensor(h)
        # self._R = torch.tensor(R)
        # self._R_root = torch.tensor(R_root)
        # self._R_root_trans = torch.tensor(R_root_trans)

        data = torch.load(filename)
        ones_vector = torch.vstack([-torch.ones(19, 1), torch.ones(19, 1)])
        self._X = data['X']
        self._G = data['G']
        self._h = data['h']*ones_vector
        self._R = data['R']
        self._Y = data['Y']
        self._R_root = data['R_root']
        self._R_root_trans = data['R_root_trans']
        self.idx = [0, 4, 5, 9, 10, 13, 15, 20, 17, 19, 21]

        # self.idx = idx
        # self._Y = None
        # self._xdim = self._X.shape[1]
        self._xdim =  torch.cat((self._X, self._R_root, self._R_root_trans), dim=1).shape[1]
        self._ydim = self._G.shape[1]
        self._num = self._X.shape[0]
        self._neq = 0
        self._nineq = self._G.shape[0] + 1

        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        self.solver_type = 'ipopt'

        self.total_cost = 2.0

        self._partial_vars = np.arange(self.ydim)
        self._other_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)
        self._partial_unknown_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)


        det = 0
        # self.solver_type = 'lagrange'
        # self.solver_type = 'waterfilling'

        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'RetargetingProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )


    @property
    def X(self):
        return self._X

    @property
    def G(self):
        return self._G.to(torch.float64)

    @property
    def h(self):
        return self._h


    @property
    def R(self):
        return self._R

    @property
    def R_root(self):
        return self._R_root

    @property
    def R_root_trans(self):
        return self._R_root_trans

    @property
    def merge(self):
        return torch.cat((self.X, self.R_root, self.R_root_trans), dim=1)

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()


    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def R_np(self):
        return self.R.detach().cpu().numpy()

    @property
    def R_root_np(self):
        return self.R_root.detach().cpu().numpy()

    @property
    def R_root_trans_np(self):
        return self.R_root_trans.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def shuffle_idx(self):
        np.random.seed(3)
        indices = np.arange(self.num)
        np.random.shuffle(indices)
        return indices

    @property
    def neq(self):
        return self._neq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def nineq(self):
        return self._nineq

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        # X = self.merge
        trainX = torch.cat((self.X[self.shuffle_idx][:int(self.num * self.train_frac)], self.R_root[self.shuffle_idx][:int(self.num * self.train_frac)], self.R_root_trans[self.shuffle_idx][:int(self.num * self.train_frac)]), dim=1)
        return trainX
        # return self.X[self.shuffle_idx][:int(self.num * self.train_frac)]

    @property
    def validX(self):
        return torch.cat((self.X[self.shuffle_idx][int(self.num * self.train_frac):int(self.num * (self.train_frac + self.valid_frac))], self.R_root[self.shuffle_idx][int(self.num * self.train_frac):int(self.num * (self.train_frac + self.valid_frac))], self.R_root_trans[self.shuffle_idx][int(self.num * self.train_frac):int(self.num * (self.train_frac + self.valid_frac))]), dim=1)

    @property
    def testX(self):
        return torch.cat((self.X[self.shuffle_idx][int(self.num * (self.train_frac + self.valid_frac)):], self.R_root[self.shuffle_idx][int(self.num * (self.train_frac + self.valid_frac)):], self.R_root_trans[self.shuffle_idx][int(self.num * (self.train_frac + self.valid_frac)):]),dim = 1)

    @property
    def trainY(self):
        return self.Y[self.shuffle_idx][:int(self.num * self.train_frac)]

    @property
    def validY(self):
        return self.Y[self.shuffle_idx][int(self.num * self.train_frac):int(self.num * (self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[self.shuffle_idx][int(self.num * (self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def obj_fn(self, X , Y):
        # R_root = self.R_root[idx,:]
        # R_root_trans = self.R_root_trans[idx,:]

        R_root = X[:,33:36]
        R_root_trans = X[:,36:39]
        X = X[:, :33]

        self.ret_pro = Retargeting_h1(self.R, R_root, R_root_trans, self.idx)
        X = X.view(X.shape[0], 11 , 3)
        Y = Y[None,:, :, None]
        loss = self.ret_pro.loss(X,Y)

        return loss

    def eq_resid(self, X, Y):
        return torch.zeros_like(Y)
        # raise NotImplementedError

    # def ineq_resid(self, X, Y):
    #     return Y @ self.G.T - self.h.T

    def ineq_dist(self, X, Y):
        X = X[:, :33]
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return torch.zeros_like(Y)
        # raise NotImplementedError
    def ineq_resid(self, X, Y):
        return torch.cat((Y@self.G.T - self.h.T, (torch.sum(Y*Y, dim = 1)-4).reshape(-1,1)), dim=1)
    def ineq_grad(self, X, Y):
        return 2 * (torch.clamp(Y @ (self.G.T) - self.h.T, 0) @ self.G+Y)

    def ineq_partial_grad(self, X, Y):
        # grad = torch.clamp(Y @ self.G.T - self.h, 0) @ self._M
        # Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        # Y[:, self.partial_vars] = grad
        # Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T

        # return Y
        raise NotImplementedError

    def process_output(self, X, Y):

        lower_bounds = -1*self.h[:19,:].T
        upper_bounds = self.h[19:,:].T
        Y = Y * (upper_bounds - lower_bounds) + lower_bounds
        return Y

    def complete_partial(self, X, Y):
        # Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        # Y[:, self.partial_vars] = Z
        # Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        # return Y
        lower_bounds = -1 * self.h[:19, :].T
        upper_bounds = self.h[19:, :].T
        Y = Y * (upper_bounds - lower_bounds) + lower_bounds
        return Y

    def unnorm(self, Y):
        lower_bounds = -1 * self.h[:19, :].T
        upper_bounds = self.h[19:, :].T

        return ((Y - lower_bounds) / (upper_bounds - lower_bounds) - 0.5) *2

    def set_w(self, wc, wo):
        self.wc = wc
        self.wo = wo

    def set_buffer(self, buffer):
        self.buffer = buffer

    @torch.no_grad()
    def _eval_func(self, X, Y_partial, Y_best=None, obj_best=None, idx=None, extra=False, extra2=False):
        bonus0 = (Y_partial - self.buffer.get(idx)).norm(dim=1, keepdim=True)
        # bonus = (Y_partial - Y_best).norm(dim=1, keepdim=True)
        Y_partial = Y_partial * 0.5 + 0.5
        Y = self.complete_partial(X, Y_partial)
        resids = self.ineq_resid(X, Y)
        gap = obj_best - self.obj_fn(X, Y).view(-1, 1)
        dist = torch.clamp(resids, -0.00).sum(dim=1, keepdim=True)
        # dist = resids.max(dim=1, keepdim=True)[0]
        judge = (resids.max(dim=1, keepdim=True)[0] <= 1e-5)
        return 0.1 * bonus0 * judge * (not extra) * extra2 + 1.0 * torch.exp(
            gap) * judge * extra - dist  # + self.wc * judge + self.wo * judge * torch.sigmoid(-self.obj_fn(Y)).view(-1,

    @torch.no_grad()  # 1)  # self.wc * torch.log(-resids.sum(dim=1, keepdim=True) * judge + 1) - self.wo * judge * self.obj_fn(Y).view(-1,1)
    def _eval_func_eval(self, X, Y_partial):
        Y_partial = Y_partial * 0.5 + 0.5
        Y = self.complete_partial(X, Y_partial)
        resids = self.ineq_resid(X, Y)
        dist = torch.clamp(resids, -0.0).sum(dim=1, keepdim=True)
        judge = (dist <= 1e-5)

        return -dist + judge * torch.sigmoid(-self.obj_fn(X, Y).view(-1,
                                                                  1))  # + self.wc * torch.log(-resids.sum(dim=1, keepdim=True) * judge + 1) - self.wo * judge * self.obj_fn(Y).view(-1,1)

    def _cons_region(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return resids <= 0

    def opt_solve(self, X, tol=1e-4, max_iter=1000):

        if self.solver_type == 'ipopt':
            # G, P_t, h = self.G_np, self.P_np, self.h_np
            G, h, R, R_root, R_root_trans, idx = self.G_np, self.h_np, self.R_np, self.R_root_np, self.R_root_trans_np, self.idx
            X_np = X.detach().cpu().numpy()

            p, total_time, parallel_time = self.ipopt_solver( G,h, R, R_root, R_root_trans, idx, X_np)

        else:
            raise NotImplementedError

        if isinstance(p, np.ndarray):
            sols = p
        else:
            sols = p.detach().cpu().numpy()

        # sols = np.array(p.detach().cpu().numpy())

        return sols, total_time, parallel_time

    def ipopt_solver(self, G,h, R, R_root, R_root_trans, idx, X_np, tol=1e-4):
        P = []
        total_time = 0
        n = 0
        p_final = None
        for X_i, R_root_i, R_root_trans_i in zip(X_np, R_root, R_root_trans):
            pos = torch.tensor(X_i)
            R_root_i = torch.tensor(R_root_i).unsqueeze(0)
            R_root_trans_i = torch.tensor(R_root_trans_i).unsqueeze(0)
            # N_0 = 1.0
            dim = 19

            # initial p_0
            # p_0 = np.full(dim, P_t / dim)
            # p_0 = np.random.rand(19)*0.4
            if p_final is None:
                p_0 = np.zeros(19)
            else:
                p_0 = p_final

            # p_0 = np.zeros(19)




            # lb = np.zeros(dim)
            # ub = np.full(dim, np.inf)

            # lb = -np.infty * np.ones(p_0.shape)
            # ub = np.infty * np.ones(p_0.shape)

            lb = h[:19,:].squeeze()
            ub = h[19:,:].squeeze()

            # print(G.shape[0])
            # cl = -np.inf * np.ones(G.shape[0])
            # cu = h


            nlp = cyipopt.Problem(
                n=dim,
                m=0,
                problem_obj=Retargeting_ipopt(G,R, R_root_i, R_root_trans_i, idx, pos),
                lb=lb,
                ub=ub,
            )
            nlp.add_option('tol', tol)
            nlp.add_option('max_iter', 200)
            nlp.add_option('print_level', 5)
            # nlp.add_option('mu_init', 0.01)
            nlp.add_option('max_soc',8)
            nlp.add_option('alpha_red_factor', 0.5)
            nlp.add_option('mu_strategy', 'adaptive')
            nlp.add_option('acceptable_tol',0.1)
            nlp.add_option('acceptable_obj_change_tol',0.1)
            # nlp.addOption()
            # nlp.add_option('nlp_scaling_method', 'gradient-based')
            # nlp.add_option('linear_solver', 'ma57')
            # nlp.add_option('hessian_approximation', 'limited-memory')

            start_time = time.time()
            p_final, info = nlp.solve(p_0)
            print(p_final)
            end_time = time.time()
            P.append(p_final)
            print(end_time - start_time)
            total_time += (end_time - start_time)
            n += 1

        return np.array(P), total_time, total_time / n

    def calc_Y(self):
        Y, t, _ = self.opt_solve(self.X)
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y, t


class Retargeting_ipopt(object):
    def __init__(self, G,  R, R_root, R_root_trans, idx, pos):
        self.pos = pos.reshape(1, 11, 3)
        self.G = G
        self.R = R
        self.R_root = R_root
        self.R_root_trans = R_root_trans
        self.idx = idx
        self.var_dim = G.shape[1]
        self.ret_pro = Retargeting_h1(self.R, self.R_root, self.R_root_trans, self.idx)

        # self.loss = None
        # self.grad = None

    def objective(self, p):
        # input:  p   d*1 array
        # output: loss  int
        p = torch.from_numpy(p).unsqueeze(0)[:,None,:, None]
        p.requires_grad_(True)

        loss = self.ret_pro.loss(self.pos, p)
        # self.loss = loss
        # self.grad = grad
        return loss.item()

    # def gradient_check(self, p, epsilon=1e-5):
    #     grad_numerical = np.zeros_like(p)
    #     for i in range(len(p)):
    #         p_plus = p.copy()
    #         p_minus = p.copy()
    #         p_plus[i] += epsilon
    #         p_minus[i] -= epsilon
    #
    #
    #         # p_plus = p_plus.
    #         # p_minus = p_minus.unsqueeze(0)[:, None, :, None]
    #         loss_plus = self.objective(p_plus)
    #         loss_minus = self.objective(p_minus)
    #
    #
    #         grad_numerical[i] = (loss_plus - loss_minus) / (2 * epsilon)
    #
    #     # print('grad_numerical', grad_numerical)
    #
    #     # grad_autodiff = self.gradient(p)
    #
    #
    #     # diff = np.linalg.norm(grad_numerical - grad_autodiff)
    #     # print("Gradient difference:", diff)
    #     return grad_numerical

    def gradient(self, p):
        # input:  p   d*1 array
        # output: grad d*1 array

        # grad_check = self.gradient_check(p)

        p = torch.from_numpy(p).unsqueeze(0)[:,None,:, None]
        p.requires_grad_(True)
        # if p.grad is not None:
        #     p.grad.zero_()

        grad = self.ret_pro.grad(self.pos, p)
        grad = grad.squeeze().detach().cpu().numpy()
        # print('grad',grad)

        # print('grad_check',grad_check)
        return grad

        # return self.grad.squeeze().numpy()

    def constraints(self, p):
        # return self.G @ p
        return np.array[[]]

    def jacobian(self, p):
        # return self.G.flatten()
        return np.array[[]]

#############################################
##   Power allocation optimization problem  #
#############################################
class PowerAllocationOptimizationProblem:
    """
    \max_{\{p_{1},\ldots,p_{M}\}} C = \sum_{m=1}^{M} \log_{2}\left(1 + g_{m}p_{m}\right)

    s.t. p_{m} \geq 0, & \forall m = 1, 2, \ldots, M \\
            \sum_{m=1}^{M} p_{m} \leq P_{T}

    """

    def __init__(self, X, P_t, G, h, lambda_m, mu, valid_frac=0.0833, test_frac=0.0833):
        self._X = torch.tensor(X)
        self._P_t = torch.tensor(P_t)
        self._G = torch.tensor(G)
        self._h = torch.tensor(h)
        self._Y = None
        self._xdim = X.shape[1]
        self._ydim = X.shape[1]
        self._num = X.shape[0]
        self._neq = 0
        self._nineq = G.shape[0]

        self._lambda_m = torch.tensor(lambda_m)
        self._mu = torch.tensor(mu)
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        det = 0
        # self.solver_type = 'lagrange'
        # self.solver_type = 'waterfilling'
        self.solver_type = 'ipopt'

        self._partial_vars = np.arange(self._ydim)
        self._other_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)
        self._partial_unknown_vars = self._partial_vars

        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'PowerProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )



    @property
    def P_t(self):
        return self._P_t

    @property
    def X(self):
        return self._X

    @property
    def G(self):
        return self._G

    @property
    def h(self):
        return self._h

    @property
    def lambda_m(self):
        return self._lambda_m

    @property
    def mu(self):
        return self._mu

    @property
    def Y(self):
        return self._Y

    # @property
    # def partial_vars(self):
    #     return self._partial_vars
    #
    # @property
    # def other_vars(self):
    #     return self._other_vars
    #
    # @property
    # def partial_unknown_vars(self):
    #     return self._partial_vars

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def P_np(self):
        return self.P_t.detach().cpu().numpy()

    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_unknown_vars

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def nineq(self):
        return self._nineq

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num * self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def obj_fn(self,X, Y):
        # Y = torch.clamp(Y,  1e-6)
        return -((1 / torch.log(torch.tensor(2.0))) * torch.log(1 + X*Y)).sum(dim=1)

    def set_w(self, wc, wo):
        self.wc = wc
        self.wo = wo

    def set_buffer(self, buffer):
        self.buffer = buffer

    @torch.no_grad()
    def _eval_func(self, X, Y_partial, Y_best=None, obj_best=None, idx=None, extra=False, extra2=False):
        bonus0 = (Y_partial - self.buffer.get(idx)).norm(dim=1, keepdim=True)
        # bonus = (Y_partial - Y_best).norm(dim=1, keepdim=True)
        Y_partial = Y_partial * 0.5 + 0.5
        Y = self.complete_partial(X, Y_partial)
        resids = self.ineq_resid(X, Y)
        gap = obj_best - self.obj_fn(X, Y).view(-1, 1)
        dist = torch.clamp(resids, -0.00).sum(dim=1, keepdim=True)
        # dist = resids.max(dim=1, keepdim=True)[0]
        judge = (resids.max(dim=1, keepdim=True)[0] <= 1e-5)
        return 0.1 * bonus0 * judge * (not extra) * extra2 + 1.0 * torch.exp(
            gap) * judge * extra - dist  # + self.wc * judge + self.wo * judge * torch.sigmoid(-self.obj_fn(Y)).view(-1,

    @torch.no_grad()  # 1)  # self.wc * torch.log(-resids.sum(dim=1, keepdim=True) * judge + 1) - self.wo * judge * self.obj_fn(Y).view(-1,1)
    def _eval_func_eval(self, X, Y_partial):
        Y_partial = Y_partial * 0.5 + 0.5
        Y = self.complete_partial(X, Y_partial)
        resids = self.ineq_resid(X, Y)
        dist = torch.clamp(resids, -0.0).sum(dim=1, keepdim=True)
        judge = (dist <= 1e-5)

        return -dist + judge * torch.sigmoid(-self.obj_fn(X, Y).view(-1,
                                                                  1))  # + self.wc * torch.log(-resids.sum(dim=1, keepdim=True) * judge + 1) - self.wo * judge * self.obj_fn(Y).view(-1,1)

    def _cons_region(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return resids <= 0

    def eq_resid(self, X, Y):
        return torch.zeros_like(Y)
        # raise NotImplementedError

    def ineq_resid(self, X, Y):
        return Y @ self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return torch.zeros_like(Y)
        # raise NotImplementedError

    def ineq_grad(self, X, Y):

        # return torch.ger(2 * torch.clamp(Y @ self.G.T - (X[:,self.ydim:]).squeeze(), 0), self.G)
        return 2 * torch.clamp(Y @ self.G.T - self.h, 0) @ self.G

    def ineq_partial_grad(self, X, Y):
        # grad = torch.clamp(Y @ self.G.T - self.h, 0) @ self._M
        # Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        # Y[:, self.partial_vars] = grad
        # Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T

        # return Y
        raise NotImplementedError


    def process_output(self, X, Y):
        return Y*self.P_t

    def complete_partial(self, X, Y):
        # Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        # Y[:, self.partial_vars] = Z
        # Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        # return Y
        return Y * self.P_t

    def unnorm(self, Y):
        return (Y / self.P_t - 0.5) * 2


    @property
    def lambda_m_np(self):
        return self.lambda_m.detach().cpu().numpy()

    def water_batch(self, G_m, P_t):
        p = None
        experts = []
        sumdata_rates = []
        subexperts = []

        for s, total_power in zip(G_m, P_t):
            a = total_power
            g_n = s
            N_0 = 1.0

            L = torch.tensor(0.0, dtype=torch.float32)
            U = a + N_0 * torch.sum(1 / (g_n + 1e-6))  # Initial upper bound

            precision = 1e-6
            # error = 1e6
            while U - L > precision:
                alpha_bar = (L + U) / 2
                p_n = torch.maximum(alpha_bar - N_0 / (g_n + 1e-6), torch.tensor(0.0))
                P = torch.sum(p_n)

                if P > a:
                    U = alpha_bar
                else:
                    L = alpha_bar
                # error = U-L
            # Final power allocation
            p_n_final = torch.maximum(alpha_bar - N_0 / (g_n + 1e-6), torch.tensor(0.0))
            p_n_final = p_n_final.unsqueeze(0)
            # Calculate data rate
            SNR = g_n * p_n_final / N_0
            data_rate = torch.log2(1 + SNR)
            sumdata_rate = torch.sum(data_rate)

            # Expert and suboptimal power allocation
            expert = p_n_final / total_power
            subexpert = p_n_final / total_power + torch.normal(0, 0.1, size=p_n_final.shape)

            if p is None:
                p = p_n_final
            else:
                p = torch.cat((p, p_n_final), dim=0)  # Use torch.cat instead of torch.stack

            experts.append(expert)
            sumdata_rates.append(sumdata_rate)
            subexperts.append(subexpert)

        return p, experts, sumdata_rates, subexperts

    def opt_solve(self,X, tol=1e-4, max_iter=1000):

        if self.solver_type == 'lagrange':
            lambda_m = self.lambda_m
            # p_list = np.zeros(self.Gdim)

            start_time = time.time()
            for _ in range(max_iter):
                p = torch.maximum(1 / (lambda_m * torch.log(torch.tensor(2.0))) - 1 / self.G_m, torch.tensor(0.0))
                total_power = torch.sum(p)
                if abs(total_power - self.P_t) < tol:
                    break
                if total_power < self.P_t:
                    lambda_m *= 0.9
                else:
                    lambda_m *= 1.1
            end_time = time.time()
            total_time = end_time - start_time
            parallel_time = total_time

        elif self.solver_type == 'waterfilling':
            mu = self.mu
            # p_list = np.zeros(self.Gdim)
            start_time = time.time()
            p, _, _, _ = self.water_batch(self.G_m, self.P_t)
            end_time = time.time()
            total_time = end_time - start_time
            parallel_time = total_time

        elif self.solver_type == 'ipopt':
            G, P_t, h = self.G_np[-1,:].reshape(1,-1), self.P_np, self.h_np[-1]
            X_np = X.detach().cpu().numpy()

            p, total_time, parallel_time = self.ipopt_solver(G, P_t, h, X_np)

        else:
            raise NotImplementedError

        if isinstance(p, np.ndarray):
            sols = p
        else:
            sols = p.detach().cpu().numpy()

        # sols = np.array(p.detach().cpu().numpy())

        return sols, total_time, parallel_time

    def ipopt_solver(self, G, P_t, h, X_np, tol=1e-7):
        P = []
        total_time = 0
        n = 0
        for X_i in X_np:
            g_m = X_i
            N_0 = 1.0
            dim = g_m.shape[0]

            # initial p_0
            p_0 = np.full(dim, P_t / dim)

            # lb = np.zeros(dim)
            # ub = np.full(dim, np.inf)

            lb = np.zeros(p_0.shape)
            ub = P_t * np.ones(p_0.shape)

            # print(G.shape[0])
            cl = -np.inf * np.ones(G.shape[0])
            cu = np.array(h).reshape(1,-1)

            nlp = cyipopt.Problem(
                n=dim,
                m=1,
                problem_obj=PowerAllocation_ipopt(g_m, G),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
            )
            nlp.addOption('tol', tol)
            nlp.addOption('print_level', 0)

            start_time = time.time()
            p_final, info = nlp.solve(p_0)
            end_time = time.time()
            P.append(p_final)
            total_time += (end_time - start_time)
            n += 1

        return np.array(P), total_time, total_time / n

    def calc_Y(self):
        Y, t, _ = self.opt_solve(self.X)
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y, t


class PowerAllocation_ipopt(object):
    def __init__(self, g_m, G):
        self.g_m = g_m
        self.G = G
        self.var_dim = g_m.shape[0]

    def objective(self, p):
        return -np.sum(1 / np.log(2) * np.log(1 + self.g_m * p))

    def gradient(self, p):
        return -1 / np.log(2) * self.g_m / (1 + self.g_m * p)

    def constraints(self, p):
        return self.G@p

    def jacobian(self, p):
        return self.G.flatten()