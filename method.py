import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
try:
    import waitGPU
    #
    # waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)

import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse
import copy
from utils import my_hash, str_to_bool
import default_args

from diffopt import DiffOpt

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description='DC3')
    parser.add_argument('--probType', type=str, default='acopf57',
                        choices=['simple', 'nonconvex', 'acopf57', 'acopf118'], help='problem type')
    parser.add_argument('--simpleVar', type=int,
                        help='number of decision vars for simple problem')
    parser.add_argument('--simpleIneq', type=int,
                        help='number of inequality constraints for simple problem')
    parser.add_argument('--simpleEq', type=int,
                        help='number of equality constraints for simple problem')
    parser.add_argument('--simpleEx', type=int,
                        help='total number of datapoints for simple problem')
    parser.add_argument('--nonconvexVar', type=int,
                        help='number of decision vars for nonconvex problem')
    parser.add_argument('--nonconvexIneq', type=int,
                        help='number of inequality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEq', type=int,
                        help='number of equality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEx', type=int,
                        help='total number of datapoints for nonconvex problem')
    parser.add_argument('--epochs', type=int,
                        help='number of neural network epochs')
    parser.add_argument('--batchSize', type=int,
                        help='training batch size')
    parser.add_argument('--lr', type=float,
                        help='neural network learning rate')
    parser.add_argument('--hiddenSize', type=int,
                        help='hidden layer size for neural network')
    parser.add_argument('--softWeight', type=float,
                        help='total weight given to constraint violations in loss')
    parser.add_argument('--softWeightEqFrac', type=float,
                        help='fraction of weight given to equality constraints (vs. inequality constraints) in loss')
    parser.add_argument('--useCompl', type=str_to_bool,
                        help='whether to use completion')
    parser.add_argument('--useTrainCorr', type=str_to_bool,
                        help='whether to use correction during training')
    parser.add_argument('--useTestCorr', type=str_to_bool,
                        help='whether to use correction during testing')
    parser.add_argument('--corrMode', choices=['partial', 'full'],
                        help='employ DC3 correction (partial) or naive correction (full)')
    parser.add_argument('--corrTrainSteps', type=int,
                        help='number of correction steps during training')
    parser.add_argument('--corrTestMaxSteps', type=int,
                        help='max number of correction steps during testing')
    parser.add_argument('--corrEps', type=float,
                        help='correction procedure tolerance')
    parser.add_argument('--corrLr', type=float,
                        help='learning rate for correction procedure')
    parser.add_argument('--corrMomentum', type=float,
                        help='momentum for correction procedure')
    parser.add_argument('--saveAllStats', type=str_to_bool,
                        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int,
                        help='how frequently (in terms of number of epochs) to save stats to file')

    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')

    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='env timesteps (default: 1000000)')

    parser.add_argument("--policy_type", type=str, default="Diffusion", metavar='S',
                        help="Diffusion, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", metavar='S',
                        help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=5, metavar='N',
                        help='diffusion timesteps (default: 20)')
    parser.add_argument('--diffusion_lr', type=float, default=0.0003, metavar='G',
                        help='diffusion learning rate (default: 0.0001)')

    parser.add_argument('--action_lr', type=float, default=0.03, metavar='G',
                        help='diffusion learning rate (default: 0.03)')
    parser.add_argument('--noise_ratio', type=float, default=1.0, metavar='G',
                        help='noise ratio in sample process (default: 1.0)')

    parser.add_argument('--action_gradient_steps', type=int, default=20, metavar='N',
                        help='action gradient steps (default: 20)')
    parser.add_argument('--ratio', type=float, default=0.1, metavar='G',
                        help='the ratio of action grad norm to action_dim (default: 0.1)')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, metavar='G',
                        help='actor and critic grad norm (default: 1.0)')

    parser.add_argument('--weighted', action="store_true", help="weighted training")

    parser.add_argument('--aug', action="store_true", help="augmentation")

    parser.add_argument('--train_sample', type=int, default=64, metavar='N',
                        help='train_sample (default: 64)')

    parser.add_argument('--chosen', type=int, default=1, metavar='N', help="chosen actions (default:1)")

    parser.add_argument('--behavior_sample', type=int, default=4, metavar='N', help="behavior_sample (default: 1)")
    parser.add_argument('--target_sample', type=int, default=4, metavar='N',
                        help="target_sample (default: behavior sample)")

    parser.add_argument('--eval_sample', type=int, default=32, metavar='N', help="eval_sample (default: 512)")

    parser.add_argument('--deterministic', action="store_true", help="deterministic mode")

    parser.add_argument('--q_transform', type=str, default='qadv', metavar='S', help="q_transform (default: qrelu)")

    parser.add_argument('--gradient', action="store_true", help="aug gradient")

    parser.add_argument('--entropy_alpha', type=float, default=0.0, metavar='G', help="entropy_alpha (default: 0.02)")

    parser.add_argument('--wo', type=float, default=0.0, metavar='G', help="wo")
    parser.add_argument('--wc', type=float, default=1.0, metavar='G', help="wc")

    args_v = parser.parse_args()
    args = vars(args_v)  # change to dictionary
    defaults = default_args.method_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)

    setproctitle('DC3-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        filepath = os.path.join('datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif prob_type == 'nonconvex':
        filepath = os.path.join('datasets', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    elif prob_type == 'acopf57':
        filepath = os.path.join('datasets', 'acopf', 'acopf57_dataset')
    elif prob_type == 'acopf118':
        filepath = os.path.join('datasets', 'acopf', 'acopf118_dataset')
    else:
        raise NotImplementedError

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass
    data._device = DEVICE

    save_dir = os.path.join('results', str(data), 'method', my_hash(str(sorted(list(args.items())))),
                            str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)

    data.set_w(args_v.wc, args_v.wo)
    buffer = Buffer(data, len(data.trainX))
    buffer.reset(data.unnorm(copy.deepcopy(data.trainY[:, data.partial_unknown_vars])), copy.deepcopy(data.trainY))
    data.set_buffer(buffer)
    print(data._xdim, data._ydim - data.nknowns - data.neq)
    optimizer = DiffOpt(args_v, buffer, data, data._xdim, data._ydim - data.nknowns - data.neq, data._eval_func, DEVICE)
    # Run method
    train_diff(optimizer, data, args, save_dir)


def train_diff(optimizer, data, args, save_dir):

    solver_step = args['lr']
    nepochs = args['epochs']
    batch_size = args['batchSize']

    train_dataset = TensorDataset(data.trainX, data.trainY, torch.arange(len(data.trainX)))
    valid_dataset = TensorDataset(data.validX, data.validY)
    test_dataset = TensorDataset(data.testX)



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    print(len(valid_dataset), len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    train_sample = optimizer.train_sample
    optimizer.train_sample = 1
    viol = float("inf")
    val = float("inf")
    stats = {}
    for i in range(nepochs*10):
        if i==1000:
            break
            optimizer.train_sample = train_sample
            data.buffer.viol = -float('inf') * torch.ones_like(data.buffer.viol)
            data.buffer.Y = torch.zeros_like(data.buffer.Y)
            data.buffer.Y_com = torch.zeros_like(data.buffer.Y_com)
        epoch_stats = {}
        print("op:", (optimizer.buffer.viol>=1e-5).sum().item()/len(optimizer.buffer))

        # Get valid loss
        if i % 5 == 0:
            for Xvalid, Yvalid in valid_loader:
                Xvalid = Xvalid.to(DEVICE)
                Yvalid = Yvalid.to(DEVICE)
                # print(data.obj_fn(Yvalid).mean(), data.ineq_resid(Xvalid, Yvalid).max(), data.eq_resid(Xvalid, Yvalid).abs().max())
                #
                optimizer.eval(Xvalid, Yvalid, data, "valid", epoch_stats, args)

        # Get valid loss
            for Xtrain, Ytrain, idx in train_loader:
                Xtrain = Xtrain.to(DEVICE)
                optimizer.eval(Xtrain, Ytrain, data, "train2", epoch_stats, args)
                break

        # # Get test loss
        # for Xtest in test_loader:
        #     Xtest = Xtest[0].to(DEVICE)
        #     optimizer.eval(Xtest, data, "test", epoch_stats, args)

        # Get train loss
        for Xtrain, Ytrain, idx in train_loader:
            Xtrain = Xtrain.to(DEVICE)
            obj_best = data.obj_fn(Ytrain.to(DEVICE)).view(-1,1)
            Ytrain_partial = Ytrain.to(DEVICE)[:, data.partial_unknown_vars]
            start_time = time.time()
            train_loss = optimizer.train(Xtrain, Ytrain_partial, obj_best, idx=idx, extra=True, extra2=False)
            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', np.array(train_loss).reshape(1))
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')

        if i % 5 == 0:
            print(
                'Epoch {}: train loss {:.4f}, eval {:.4f}({:.4f}), dist {:.4f}, ineq max {:.4f}({:.4f}), ineq mean {:.4f}({:.4f}), ineq num viol {:.4f}({:.4f}), eq max {:.4f}({:.4f}), eq mean {:.4f}({:.4f}), time {:.4f}'.format(
                    i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                    np.std(epoch_stats['valid_eval']),
                    np.mean(epoch_stats['valid_dist']), np.mean(epoch_stats['valid_ineq_max']),
                    np.std(epoch_stats['valid_ineq_max']),
                    np.mean(epoch_stats['valid_ineq_mean']), np.std(epoch_stats['valid_ineq_mean']),
                    np.mean(epoch_stats['valid_ineq_num_viol_0']), np.std(epoch_stats['valid_ineq_num_viol_0']),
                    np.mean(epoch_stats['valid_eq_max']), np.std(epoch_stats['valid_eq_max']),
                    np.mean(epoch_stats['valid_eq_mean']),
                    np.std(epoch_stats['valid_eq_mean']),
                    np.mean(epoch_stats['valid_time'])))
            print(
                'Epoch {}: train loss {:.4f}, eval {:.4f}, dist {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, time {:.4f}'.format(
                    i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['train2_eval']),
                    np.mean(epoch_stats['train2_dist']), np.mean(epoch_stats['train2_ineq_max']),
                    np.mean(epoch_stats['train2_ineq_mean']), np.mean(epoch_stats['train2_ineq_num_viol_0']),
                    np.mean(epoch_stats['train2_eq_max']),
                    np.mean(epoch_stats['train2_time'])))

        if args['saveAllStats']:
            if i == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        else:
            stats = epoch_stats

        if (i % args['resultsSaveFreq'] == 0):
            if np.mean(epoch_stats['valid_ineq_max'])<viol:
                viol = np.mean(epoch_stats['valid_ineq_max'])
                val = np.mean(epoch_stats['valid_eval'])
                with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                    pickle.dump(stats, f)
                optimizer.save_model(save_dir)
            elif abs(np.mean(epoch_stats['valid_ineq_max'])-viol)<=1e-3 and np.mean(epoch_stats['valid_eval']) < val:
                viol = np.mean(epoch_stats['valid_ineq_max'])
                val = np.mean(epoch_stats['valid_eval'])
                with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                    pickle.dump(stats, f)
                optimizer.save_model(save_dir)

    return stats


# Modifies stats in place
def dict_agg(stats, key, value, op='concat'):
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value

class Buffer(object):
    def __init__(self, data, size):
        self.data = data
        self.viol = -float("inf") * torch.ones(size, 1, device=DEVICE)
        self.Y = torch.zeros(size, data._ydim-data.nknowns-data.neq, device=DEVICE)
        self.Y_pre= torch.zeros(size, data._ydim-data.nknowns-data.neq, device=DEVICE)
        self.size = size
        self.Y_com = torch.zeros(size, data._ydim, device=DEVICE)

    def replace(self, idx, Y_new, viol):
        viol = self.data._eval_func_eval(self.data.trainX[idx], self.tmp, True)
        self.Y_com[idx] = torch.where(viol>self.viol[idx], self.tmp, self.Y_com[idx])
        self.Y[idx] = torch.where(viol>self.viol[idx], Y_new, self.Y[idx])
        self.viol[idx] = torch.where(viol>self.viol[idx], viol, self.viol[idx])
    def __getitem__(self, idx):
        return self.Y[idx], self.viol[idx]

    def store(self, tmp):
        self.tmp = tmp.detach().clone()

    def new(self, idx, Y_new):
        self.Y_pre[idx] = Y_new

    def get(self, idx):
        return self.Y_pre[idx]
    def __len__(self):
        return self.size

    def reset(self, trainY, full):
        self.Y_com.copy_(full)
        self.viol.zero_()
        self.Y.copy_(trainY)



if __name__ == '__main__':
    main()
