import copy
from itertools import count, permutations
import math 
import numpy as np

from einops import rearrange, reduce, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


tf = lambda x: torch.FloatTensor(x)
tl = lambda x: torch.LongTensor(x)

def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    net = nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))
    return net


class IQN(nn.Module):
    def __init__(self, args):
        super().__init__()
        # input: state (bs, args.horizon * args.ndim)
        # output: flow values (bs, n_quantiles, args.ndim+1), quantiles (bs, n_quantiles)
        self.feature = make_mlp([args.horizon * args.ndim] + \
                                [args.n_hid] * (args.n_layers-1))

        self.quantile_embed_dim = args.quantile_dim
        self.phi = make_mlp([self.quantile_embed_dim,] + [args.n_hid]*2)
        self.register_buffer("feature_id", torch.arange(1, 1+self.quantile_embed_dim))

        self.last = make_mlp([args.n_hid]*2 + [args.ndim+1])
    
    def forward(self, state, quantiles):
        batch_size, n_quantiles = quantiles.shape
        assert batch_size == state.shape[0]

        feature_id = repeat(self.feature_id, "d -> b n d", b=batch_size, n=n_quantiles)
        quantiles_rep = repeat(quantiles, "b n -> b n d", d=self.quantile_embed_dim)
        cos = torch.cos(math.pi * feature_id * quantiles_rep) # (bs, n_quantiles, d)
        x = self.feature(state).unsqueeze(1) * F.relu(self.phi(cos)) # (bs, n_quantiles, n_hid)
        logflow_vals = self.last(x) # (bs, n_quantiles, ndim+1)
        return logflow_vals


def normal_cdf(value, loc=0., scale=1.):
    return 0.5 * (1 + torch.erf((value - loc) / (scale * math.sqrt(2))))

def normal_invcdf(value, loc=0., scale=1.):
    return loc + scale * torch.erfinv(2 * value - 1) * math.sqrt(2)


class DistFlowNetAgentIQN:
    def __init__(self, args, envs):
        assert args.n_layers >= 1
        self.model = IQN(args)
        self.model.to(args.dev)
        self.target = copy.deepcopy(self.model)
        self.envs = envs
        self.ndim = args.ndim
        self.tau = args.bootstrap_tau
        self.args = args
        self.device = args.device

        self.N = args.N
        self.n_quantile_in = args.N
        self.n_quantile_out = args.N
        self.thompson_sampling = args.ts

        self.in_distort = self.out_distort = False
        if args.indist:
            assert args.beta != "neutral"
            self.in_distort = args.indist
        if args.outdist:
            assert args.beta != "neutral"
            self.out_distort = args.outdist

        # \int_0^1 Z(t) dg(t) = \int_0^1 Z(g^{-1}(t)) dt
        if args.beta in ["neutral"]:
            assert args.eta == 0.
            self.g_inv = lambda tau: tau
        elif args.beta in ["cvar"]:
            assert args.eta <= 1 and args.eta >= 0
            self.g_inv = lambda tau: tau * args.eta
        elif args.beta in ["cpw"]:
            assert args.beta != 0. # preferably eta=0.71
            self.g_inv = lambda tau: (tau ** args.eta) / ((tau ** args.eta + (1 - tau) ** args.eta) ** (1 / args.eta))
        elif args.beta in ["wang"]:
            # eta < 0: risk averse
            self.g_inv = lambda tau: normal_cdf(normal_invcdf(tau) + args.eta)
        elif args.beta in ["pow"]:
            if args.eta >= 0.: # risk seeking
                self.g_inv = lambda tau: tau ** (1 / (1 + args.eta))
            else: # risk averse
                self.g_inv = lambda tau: 1 - (1 - tau) ** (1 / (1 - args.eta))
        else:
            raise NotImplementedError
        
    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, all_visited, eval=False):
        batch = []
        s = tf([i.reset()[0] for i in self.envs])
        done = [False] * mbsize
        while not all(done):
            with torch.no_grad():
                if self.thompson_sampling and eval is False:
                    tau = torch.rand(s.shape[0], self.N).to(self.device)
                    preds = self.model(s, self.g_inv(tau))
                    aug_bs = preds.shape[0]
                    pred = preds[torch.arange(aug_bs), torch.randint(self.N, size=(aug_bs,))]
                else:
                    quantiles = torch.rand(s.shape[0], self.N).to(s.device)
                    quantiles = self.g_inv(quantiles)
                    pred = self.model(s, quantiles).logsumexp(dim=1)  # (bs, ndim+1

                acts = Categorical(logits=pred).sample()
                
            step = [e.step(a) for e, a in zip([e for d, e in zip(done, self.envs) if not d], acts)]
            p_a = [self.envs[0].parent_transitions(sp_state, a == self.ndim)
                        for a, (sp, r, done, sp_state) in zip(acts, step)]
            batch += [
                        [tf(i) for i in (p, a, [r], [sp], [d])]
                            for (p, a), (sp, r, d, _) in zip(p_a, step)
                    ]
            c = count(0)
            m = {j:next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])
            for (_, r, d, sp) in step:
                if d:
                    all_visited.append(tuple(sp))
        return batch

    def learn_from(self, it, batch):
        loginf = tf([1000])
        
        # batch_idxs.shape[0] = parents_Qsa.shape[0] > sp.shape[0] = in_flow.shape[0]
        batch_idxs = tl(
            sum([[i] * len(parents) for i, (parents,_,_,_,_) in enumerate(batch)], [])
        )
        parents, actions, r, sp, done = map(torch.cat, zip(*batch))

        # shud be called percentage rather than quantiles
        quantiles = torch.rand(sp.shape[0], self.n_quantile_in).to(self.device)
        if self.in_distort:
            quantiles = self.g_inv(quantiles)

        # inflow
        in_quantiles = torch.gather(quantiles, 0, repeat(batch_idxs, "npar -> npar nq_in", nq_in=self.n_quantile_in))
        logflows = self.model(parents, in_quantiles)
        parents_Qsa = logflows[torch.arange(parents.shape[0]), ..., actions.long()] # (num, n_quantile_in)
        in_flow = torch.zeros((sp.shape[0], self.n_quantile_in)).to(self.device)\
                    .index_add_(0, batch_idxs, torch.exp(parents_Qsa)) # (bs, n_quantile_in)

        # outflow
        out_quantiles = torch.rand(sp.shape[0], self.n_quantile_out).to(self.device)
        if self.out_distort:
            out_quantiles = self.g_inv(out_quantiles)

        if self.tau > 0:
            with torch.no_grad(): 
                next_q = self.target(sp, out_quantiles)
        else:
            next_q = self.model(sp, out_quantiles) # (bs, n_quantile_out, n_child)

        next_qd = next_q * (1-done)[..., None, None] + done[..., None, None] * (-loginf)
        log_out_flow = torch.where(repeat(done.bool(), "b -> b nq", nq=self.n_quantile_out),
                                    repeat(torch.log(r), "b -> b nq", nq=self.n_quantile_out),
                                    next_qd.logsumexp(dim=-1)) # (bs, n_quantile_out)
        
        diff = repeat(log_out_flow, "b n_out -> b 1 n_out") - repeat(in_flow.log(), "b n_in -> b n_in 1")
        abs_weight = torch.abs(repeat(quantiles, "b n_in -> b n_in n_out", n_out=self.n_quantile_out) \
            - diff.le(0).float())
        losses = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
        losses = (abs_weight * losses).sum(dim=-2).mean(dim=-1) # sum over qunatile_in, mean over quantile_out
        loss = (losses * done * self.args.leaf_coef + losses * (1 - done)).sum() / len(losses)

        with torch.no_grad():
            term_loss = (losses * done).pow(2).sum() / (done.sum() + 1e-20)
            flow_loss = (losses * (1-done)).pow(2).sum() / ((1-done).sum() + 1e-20)

        if self.tau > 0:
            for a,b in zip(self.model.parameters(), self.target.parameters()):
                b.data.mul_(1-self.tau).add_(self.tau*a)

        return loss, term_loss.detach(), flow_loss.detach()
