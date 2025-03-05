import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusion import Diffusion
from q_transform import *
import os
import time
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
class DiffOpt(object):
    def __init__(self,
                 args,
                 buffer,
                 data,
                 state_dim,
                 action_dim,
                 eval_func,
                 device,
                 ):
        self.buffer = buffer
        self.data = data
        self.policy_type = args.policy_type
        if self.policy_type == 'Diffusion':
            self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, noise_ratio=args.noise_ratio,
                                   beta_schedule=args.beta_schedule, n_timesteps=args.n_timesteps, behavior_sample=args.behavior_sample,
                                   eval_sample=args.eval_sample, deterministic=args.deterministic).to(device)
            self.running_q_std = 1.0
            self.running_q_mean = 0.0

            self.chosen = args.chosen
            self.eval_func = eval_func

            self.weighted = args.weighted
            self.aug = args.aug
            self.train_sample = args.train_sample

            self.q_transform = args.q_transform
            self.gradient = args.gradient


            self.entropy_alpha = args.entropy_alpha

        else:
            raise NotImplementedError

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.diffusion_lr, eps=1e-5)

        self.action_gradient_steps = args.action_gradient_steps

        self.action_grad_norm = action_dim * args.ratio
        self.ac_grad_norm = args.ac_grad_norm

        self.step = 0

        # self.running_v = args.running_v
        # if args.running_v:
        #     self.critic_v = RunningV(state_dim).to(device)
        #     self.critic_v_optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=args.critic_lr, eps=1e-5)

        self.action_dim = action_dim

        self.action_lr = args.action_lr

        self.device = device

        self.action_scale = 1.
        self.action_bias = 0.

    def sample_action(self, state, eval=False):
        state = state
        normal = False
        if not eval and torch.rand(1).item() <= self.epsilon:
            normal = True

        action = self.actor(state, eval, q_func=self.data._eval_func_eval, normal=normal)
        action = action.clip(-1, 1)
        action = action * self.action_scale + self.action_bias
        return action

    def action_aug(self, states, Y, obj_best, idx, extra, log_writer, return_mean_std=False):

        # states, actions, rewards, next_states, masks = self.memory.sample(batch_size)
        old_states = states
        states, best_actions, v_target, (mean, std) = self.actor.sample_n(states, Y=Y, obj_best=obj_best, times=self.train_sample, chosen=self.chosen, q_func=self.eval_func, extra=extra, idx=idx, buffer=self.buffer)
        # if self.running_v:
        #     v = self.critic_v(old_states)
        #     v_loss = F.mse_loss(v, v_target[1].squeeze(1))
        #     self.critic_v_optimizer.zero_grad()
        #     v_loss.backward()
        #     self.critic_v_optimizer.step()
        #     v = v.detach().unsqueeze(1)
        # else:
        v = v_target[1]

        if return_mean_std:
            return states, best_actions, (v_target[0], v), (mean, std)
        else:
            return states, best_actions, (v_target[0], v)

    def action_gradient(self, states, log_writer, return_mean_std=False):
        # states, best_actions, idxs = self.diffusion_memory.sample(batch_size)
        q = self.eval_func(states, best_actions)
        mean = q.mean()
        std = q.std()

        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            best_actions.requires_grad_(True)
            q = self.eval_func(states, best_actions)
            loss = -q

            actions_optim.zero_grad()

            loss.backward(torch.ones_like(loss))
            if self.action_grad_norm > 0:
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm,
                                                              norm_type=2)

            actions_optim.step()

            best_actions.requires_grad_(False)
            best_actions.clamp_(-1., 1.)

        # if self.step % 10 == 0:
        #     log_writer.add_scalar('Action Grad Norm', actions_grad_norms.max().item(), self.step)

        best_actions = best_actions.detach()

        self.diffusion_memory.replace(idxs, best_actions.cpu().numpy())

        if return_mean_std:
            return states, best_actions, (mean, std)
        else:
            return states, best_actions

    def train(self, states, Y, obj_best=None, idx=None, extra=False, extra2=False, log_writer=None):

        extra = (self.step % 2 == 0) and extra
        batch_size = states.shape[0]
        """ Policy Training """
        if self.aug:
            # if self.gradient:
            #     states, best_actions, qv, (mean, std) = self.aug_gradient(states, log_writer, return_mean_std=True)
            states, best_actions, qv, (mean, std) = self.action_aug(states, Y, obj_best , idx, (extra,extra2), log_writer, return_mean_std=True)
        else:
            states, best_actions, (mean, std) = self.action_gradient(states, log_writer, return_mean_std=True)

        if self.policy_type == 'Diffusion' and self.weighted:
            if self.aug:
                q, v = qv
                self.buffer.replace(idx, best_actions[:batch_size], q[:batch_size])
            else:
                v = None
                with torch.no_grad():
                    q = self.eval_func(states, best_actions)

            # q.clamp_(-self.q_neg).add_(self.q_neg)
            q = eval(self.q_transform)(q, v=v, batch_size=batch_size, chosen=self.chosen)
            if self.entropy_alpha > 0.0:
                rand_states = states.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*self.chosen*10, -1)
                rand_policy_actions = torch.empty(batch_size * self.chosen * 10, best_actions.shape[-1], device=self.device).uniform_(
                    -1, 1)
                rand_q = q.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*self.chosen*10, -1) * self.entropy_alpha

                best_actions = torch.cat([best_actions, rand_policy_actions], dim=0)
                states = torch.cat([states, rand_states], dim=0)
                q = torch.cat([q, rand_q], dim=0)
            # q[q<1.0] = 1.0
            # q = torch.clip(q / self.running_avg_qnorm, -6 ,6)
            # expq = torch.exp(self.beta * q)
            # expq[expq<=expq.quantile(0.95)] = 0.0
            # if itr % 10000 == 0 : print(expq, itr)
            # print("expq", expq.shape)
            actor_loss = self.actor.loss(best_actions, states, weights=q)
        else:
            actor_loss = self.actor.loss(best_actions, states)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.ac_grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm,
                                                        norm_type=2)
            # if self.step % 10 == 0:
            #     log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
        self.actor_optimizer.step()


        self.step += 1
        return actor_loss.item()

    def eval(self, X, Y_BEST, data, prefix, stats, args):
        eps_converge = args['corrEps']
        make_prefix = lambda x: "{}_{}".format(prefix, x)
        start_time = time.time()
        Y_partial = self.sample_action(X, eval=True)
        Y_partial_mod = Y_partial * 0.5 + 0.5
        base_end_time = time.time()
        Y = self.data.complete_partial(X, Y_partial_mod)
        end_time = time.time()
        Ycorr = Ynew = Y
        raw_end_time = time.time()

        print((self.data.ineq_dist(X,Y).max(dim=1, keepdim=True)[0]<=1e-5).sum().item()/X.shape[0], X.shape)

        dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
        # dict_agg(stats, make_prefix('loss'), self.eval_func(X, Y_partial).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy())
        dict_agg(stats, make_prefix('dist'), torch.norm(data.unnorm(Ycorr[:, data.partial_unknown_vars]) - data.unnorm(Y_BEST[:, data.partial_unknown_vars]), dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_num_viol_0'),
                 torch.sum(data.ineq_dist(X, Ycorr) > eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_num_viol_1'),
                 torch.sum(data.ineq_dist(X, Ycorr) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_num_viol_2'),
                 torch.sum(data.ineq_dist(X, Ycorr) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_max'),
                 torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_mean'),
                 torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_num_viol_0'),
                 torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_num_viol_1'),
                 torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_num_viol_2'),
                 torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_time'), (raw_end_time - end_time) + (base_end_time - start_time), op='sum')
        dict_agg(stats, make_prefix('raw_eval'), data.obj_fn(Ynew).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_ineq_max'),
                 torch.max(data.ineq_dist(X, Ynew), dim=1)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(data.ineq_dist(X, Ynew), dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_ineq_num_viol_0'),
                 torch.sum(data.ineq_dist(X, Ynew) > eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_ineq_num_viol_1'),
                 torch.sum(data.ineq_dist(X, Ynew) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_ineq_num_viol_2'),
                 torch.sum(data.ineq_dist(X, Ynew) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_eq_max'),
                 torch.max(torch.abs(data.eq_resid(X, Ynew)), dim=1)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_eq_mean'),
                 torch.mean(torch.abs(data.eq_resid(X, Ynew)), dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_eq_num_viol_0'),
                 torch.sum(torch.abs(data.eq_resid(X, Ynew)) > eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_eq_num_viol_1'),
                 torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_eq_num_viol_2'),
                 torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        return stats

    def save_model(self, dir, id=None):
        if not os.path.exists(dir):
            os.mkdir(dir)
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth', map_location=self.device))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth', map_location=self.device))

    def aug_gradient(self, states, log_writer, return_mean_std=False):

        # states, best_actions, v, (mean, std) = self.action_aug(batch_size, log_writer, return_mean_std=True)


        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            best_actions.requires_grad_(True)
            q = self.eval_func(states, best_actions)
            loss = -q

            actions_optim.zero_grad()

            loss.backward(torch.ones_like(loss))
            if self.action_grad_norm > 0:
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm,
                                                              norm_type=2)

            actions_optim.step()

            best_actions.requires_grad_(False)
            best_actions.clamp_(-1., 1.)

        # if self.step % 10 == 0:
        #     log_writer.add_scalar('Action Grad Norm', actions_grad_norms.max().item(), self.step)

        best_actions = best_actions.detach()

        _, v = v
        with torch.no_grad():
            q = self.eval_func(states, best_actions)

        if return_mean_std:
            return states, best_actions, (q, v), (mean, std)
        else:
            return states, best_actions, (q, v)

    def get_policy(self, states, times):
        batch_size = states.shape[0]
        states = states.unsqueeze(1).repeat(1, times, 1).view(times*batch_size, -1)
        actions = self.actor(states, eval=False, normal=True)
        return actions

    def get_value(self, states, actions):
        action_shape = actions.shape[0]
        state_shape = states.shape[0]
        rep = int(action_shape / state_shape)
        states = states.unsqueeze(1).repeat(1, rep, 1).view(rep*state_shape, -1)
        q = self.eval_func(states, actions)
        return q.view(state_shape, rep, 1)
