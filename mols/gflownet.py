import argparse
import os
import gzip
import pdb
import pickle
import threading
import time
from copy import deepcopy
import wandb
import random
import warnings
warnings.filterwarnings('ignore')

from einops import rearrange, reduce, repeat
import networkx as nx
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch_scatter

import model_atom, model_block
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from mol_dataset import Dataset, DatasetDirect
from metrics import eval_mols


def detailed_balance_loss(P_F, P_B, F, R, traj_lengths):
    cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=traj_lengths.device),
                                traj_lengths]), 0).long()
    total_loss = torch.zeros(1, device=traj_lengths.device)
    
    for ep in range(traj_lengths.shape[0]):  # batch size
        offset = cumul_lens[ep]
        T = int(traj_lengths[ep])
        for i in range(T):
            # This flag is False if the endpoint flow of this trajectory is R == F(s_T)
            flag = float(i + 1 < T)
            acc = (F[offset + i] - F[offset + min(i + 1, T - 1)] * flag - R[ep] * (1 - flag)
                   + P_F[offset + i] - P_B[offset + i])
            total_loss += acc.pow(2)

    return total_loss

def trajectory_balance_loss(P_F, P_B, F, R, traj_lengths):
    cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=traj_lengths.device), traj_lengths]), 0).long()
    total_loss = torch.zeros(1, device=traj_lengths.device)
    for ep in range(traj_lengths.shape[0]):
        offset = cumul_lens[ep]
        T = int(traj_lengths[ep])
        total_loss += (F[offset] - R[ep] + P_F[offset:offset+T].sum() - P_B[offset:offset+T].sum()).pow(2)
    return total_loss / float(traj_lengths.shape[0])

def tb_lambda_loss(P_F, P_B, F, R, traj_lengths, Lambda):
    cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=traj_lengths.device), traj_lengths]), 0).long()
    total_loss = torch.zeros(1, device=traj_lengths.device)
    total_Lambda = torch.zeros(1, device=traj_lengths.device)
    for ep in range(traj_lengths.shape[0]):
        offset = cumul_lens[ep]
        T = int(traj_lengths[ep])
        for i in range(T):
            for j in range(i, T):
                # This flag is False if the endpoint flow of this subtrajectory is R == F(s_T)
                flag = float(j + 1 < T)
                acc = F[offset + i] - F[offset + min(j + 1, T - 1)] * flag - R[ep] * (1 - flag)
                for k in range(i, j + 1):
                    acc += P_F[offset + k] - P_B[offset + k]
                total_loss += acc.pow(2) * Lambda ** (j - i + 1)
                total_Lambda += Lambda ** (j - i + 1)
    return total_loss / total_Lambda


def make_model(args, mdp, out_per_mol=1):
    if args.repr_type == 'block_graph':
        if args.obj == "qm":
            if args.model_version == 'v4':
                print(f"Warning: args.model_version={args.model_version}, not v3")
            model = model_block.DistGraphAgent(nemb=args.nemb,
                                nvec= args.nvec,
                                       out_per_stem=mdp.num_blocks,
                                       out_per_mol=out_per_mol,
                                       num_conv_steps=args.num_conv_steps,
                                       mdp_cfg=mdp,
                                       version=args.model_version,
                                quantile_dim=args.quantile_dim,  
                                n_quantiles=args.N,
                                thompson_sampling=args.ts)
        else:
            model = model_block.GraphAgent(nemb=args.nemb,
                                       nvec=0,
                                       out_per_stem=mdp.num_blocks,
                                       out_per_mol=out_per_mol,
                                       num_conv_steps=args.num_conv_steps,
                                       mdp_cfg=mdp,
                                       version=args.model_version)

    elif args.repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=args.nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=out_per_mol,
                                     num_conv_steps=args.num_conv_steps,
                                     version=args.model_version,
                                     do_nblocks=(hasattr(args,'include_nblocks')
                                                 and args.include_nblocks), dropout_rate=0.1)
    elif args.repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
    return model


class Proxy:
    def __init__(self, args, bpath, device):
        home_path = os.path.expanduser("~")
        proxy_path = f"{home_path}/Distributional-GFlowNets/mols/data/pretrained_proxy"
        eargs = pickle.load(gzip.open(f'{proxy_path}/info.pkl.gz'))['args']
        params = pickle.load(gzip.open(f'{proxy_path}/best_params.pkl.gz'))
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, eargs.repr_type)
        self.mdp.floatX = args.floatX
        self.proxy = make_model(eargs, self.mdp)
        # If you get an error when loading the proxy parameters, it is probably due to a version
        # mismatch in torch geometric. Try uncommenting this code instead of using the
        # super_hackish_param_map
        # for a,b in zip(self.proxy.parameters(), params):
        #    a.data = torch.tensor(b, dtype=self.mdp.floatX)
        super_hackish_param_map = {
            'mpnn.lin0.weight': params[0],
            'mpnn.lin0.bias': params[1],
            'mpnn.conv.bias': params[3],
            'mpnn.conv.nn.0.weight': params[4],
            'mpnn.conv.nn.0.bias': params[5],
            'mpnn.conv.nn.2.weight': params[6],
            'mpnn.conv.nn.2.bias': params[7],
            'mpnn.conv.lin.weight': params[2],
            'mpnn.gru.weight_ih_l0': params[8],
            'mpnn.gru.weight_hh_l0': params[9],
            'mpnn.gru.bias_ih_l0': params[10],
            'mpnn.gru.bias_hh_l0': params[11],
            'mpnn.lin1.weight': params[12],
            'mpnn.lin1.bias': params[13],
            'mpnn.lin2.weight': params[14],
            'mpnn.lin2.bias': params[15],
            'mpnn.set2set.lstm.weight_ih_l0': params[16],
            'mpnn.set2set.lstm.weight_hh_l0': params[17],
            'mpnn.set2set.lstm.bias_ih_l0': params[18],
            'mpnn.set2set.lstm.bias_hh_l0': params[19],
            'mpnn.lin3.weight': params[20],
            'mpnn.lin3.bias': params[21],
        }
        for k, v in super_hackish_param_map.items():
            self.proxy.get_parameter(k).data = torch.tensor(v, dtype=self.mdp.floatX)
        self.proxy.to(device)

    def __call__(self, m):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        return self.proxy(m, do_stems=False)[1].item()

_stop = [None]


def train_model_with_proxy(args, model, proxy, dataset, num_steps=None, do_save=True):
    debug_no_threads = False
    device = torch.device('cuda')

    if num_steps is None:
        num_steps = args.num_iterations + 1

    tau = args.bootstrap_tau
    if args.bootstrap_tau > 0:
        target_model = deepcopy(model)

    dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)

    if do_save:
        exp_dir = f'{args.save_path}'
        os.makedirs(exp_dir, exist_ok=True)

    def save_stuff(iter):
        pickle.dump(dataset.sampled_mols,
            gzip.open(f'{exp_dir}/' + str(iter) + '_sampled_mols.pkl.gz', 'wb'))
        pickle.dump(save_dict, gzip.open(os.path.join(exp_dir, "result.json"), 'wb'))
        print(f"Iter = {iter}. Saved at {exp_dir}.")

    save_dict = {
                "state_visited": {},
                "spearman_corr": {},

                "reward_top10_recent": {}, "reward_top100_recent": {}, "reward_top1000_recent": {},
                "tanimoto_top10_recent": {}, "tanimoto_top100_recent": {}, "tanimoto_top1000_recent": {},
                "num_modes R>7.5 recent": {}, "num_modes R>8.0 recent": {},

                "reward_top10": {}, "reward_top100": {}, "reward_top1000": {},
                "tanimoto_top10": {}, "tanimoto_top100": {}, "tanimoto_top1000": {},
                "num_modes R>7.5": {}, "num_modes R>8.0": {},
                }
    tf = lambda x: torch.tensor(x, device=device).to(args.floatX)
    tint = lambda x: torch.tensor(x, device=device).long()
    if args.obj == 'tb':
        model.logZ = nn.Parameter(tf(args.initial_log_Z))
    opt = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay,
                           betas=(args.opt_beta, args.opt_beta2), eps=args.opt_epsilon)
    mbsize = args.mbsize

    if not debug_no_threads:
        sampler = dataset.start_samplers(8, mbsize)

    def stop_everything():
        print('joining')
        dataset.stop_samplers_and_join()
    _stop[0] = stop_everything

    last_losses = []
    train_losses = []
    train_infos = []
    time_last_check = time.time()

    loginf = 1000 # to prevent nans
    log_reg_c = args.log_reg_c
    clip_loss = tf([args.clip_loss])
    balanced_loss = args.balanced_loss
    do_nblocks_reg = False
    max_blocks = args.max_blocks
    leaf_coef = args.leaf_coef
    
    Lambda = tf([args.subtb_lambda])

    for i in range(num_steps):
        if not debug_no_threads:
            r = sampler()
            for thread in dataset.sampler_threads:
                if thread.failed:
                    stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
                    return
            minibatch = r
        else:
            minibatch = dataset.sample2batch(dataset.sample(mbsize))
        
        if args.obj == 'fm':
            p, pb, a, r, s, d, mols = minibatch
            # Since we sampled 'mbsize' trajectories, we're going to get
            # roughly mbsize * H (H is variable) transitions
            ntransitions = r.shape[0]
            # state outputs
            if tau > 0:
                with torch.no_grad():
                    stem_out_s, mol_out_s = target_model(s, None)
            else:
                stem_out_s, mol_out_s = model(s, None)
            # mol_out_s[0] is stop action logits

            # parents of the state outputs
            stem_out_p, mol_out_p = model(p, None)
            # index parents by their corresponding actions
            qsa_p = model.index_output_by_action(p, stem_out_p, mol_out_p[:, 0], a)
            # then sum the parents' contribution, this is the inflow
            max_qsap = qsa_p.max() 
            exp_inflow = torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)\
                    .index_add_(0, pb, torch.exp(qsa_p - max_qsap)) # pb is the parents' batch index
            inflow = torch.logaddexp(exp_inflow.log(), np.log(log_reg_c) - max_qsap) + max_qsap

            assert torch.all(r[d == 1.] > 0.), "zero terminal reward!"
            assert torch.all(r[d == 0.] == 0.)  # when d=0, whether r is always 0!
            size = int(s.stems_batch.max().item() + 1)
            
            outflow = torch.logaddexp(
                torch_scatter.scatter_logsumexp(stem_out_s, s.stems_batch, dim=-2, dim_size=size)\
                    .logsumexp(dim=-1),
                mol_out_s[:, 0]
            ) # same as exp_outflow2.log()
            outflow = torch.logaddexp(outflow, np.log(log_reg_c)*torch.ones_like(outflow)) # care less about tiny flows
            outflow_plus_r = torch.where(d > 0, (r+log_reg_c).log(), outflow)
            exp_outflow = outflow.exp().detach()  # for logging

            if do_nblocks_reg:
                losses = _losses = ((inflow - outflow_plus_r) / (s.nblocks * max_blocks)).pow(2)
            else:
                losses = _losses = (inflow - outflow_plus_r).pow(2)
            if clip_loss > 0:
                ld = losses.detach()
                losses = losses / ld * torch.minimum(ld, clip_loss)
            term_loss = (losses * d).sum() / (d.sum() + 1e-20)
            flow_loss = (losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
            if balanced_loss:
                loss = term_loss * leaf_coef + flow_loss
            else:
                loss = losses.mean()
            opt.zero_grad()
            loss.backward(retain_graph=(not i % 50))

            _term_loss = (_losses * d).sum() / (d.sum() + 1e-20)
            _flow_loss = (_losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
            last_losses.append((loss.item(), term_loss.item(), flow_loss.item()))
            train_losses.append((loss.item(), _term_loss.item(), _flow_loss.item(),
                                 term_loss.item(), flow_loss.item()))
            if not i % 50:
                train_infos.append((
                    _term_loss.data.cpu().numpy(),
                    _flow_loss.data.cpu().numpy(),
                    exp_inflow.data.cpu().numpy(),
                    exp_outflow.data.cpu().numpy(),
                    r.data.cpu().numpy(),
                    mols[1],
                    [i.pow(2).sum().item() for i in model.parameters()],
                    torch.autograd.grad(loss, qsa_p, retain_graph=True)[0].data.cpu().numpy(),
                    torch.autograd.grad(loss, stem_out_s, retain_graph=True)[0].data.cpu().numpy(),
                    torch.autograd.grad(loss, stem_out_p, retain_graph=True)[0].data.cpu().numpy(),
                ))

        elif args.obj == 'qm':
            p, pb, a, r, s, d, mols = minibatch
            ntransitions = r.shape[0]
            assert ntransitions == int(s.batch.max() + 1) # num of mols "batch size"
            assert tau == 0.

            n_quantile_in = n_quantile_out = args.N
            quantiles = torch.rand((ntransitions, n_quantile_in), dtype=dataset.floatX).to(device)
            
            # inflow
            in_quantiles = torch.gather(quantiles, 0, repeat(pb, "npar -> npar nq_in", nq_in=n_quantile_in))
            stem_out_p, mol_out_p = model.forward_with_quantile(p, in_quantiles)
            qsa_p = model.index_output_by_action(p, stem_out_p, mol_out_p[..., 0], a)
            max_qsap = qsa_p.max(dim=0, keepdim=True)[0] # (1, nq)
            exp_inflow = torch.zeros((ntransitions, n_quantile_in), device=device, dtype=dataset.floatX).\
                index_add_(0, pb, torch.exp(qsa_p - max_qsap)) # pb is the parents' batch index
            inflow = torch.logaddexp(exp_inflow.log(), np.log(log_reg_c) - max_qsap) + max_qsap 

            # outflow
            out_quantiles = torch.rand((ntransitions, n_quantile_out), dtype=dataset.floatX).to(device)
            stem_out_s, mol_out_s = model.forward_with_quantile(s, out_quantiles)

            size = int(s.stems_batch.max().item() + 1)
            outflow = torch.logaddexp(
                torch_scatter.scatter_logsumexp(stem_out_s, s.stems_batch, dim=0, dim_size=size).\
                    logsumexp(dim=-1),
                mol_out_s[:, ..., 0]
            ) # same as exp_outflow2.log()
            outflow = torch.logaddexp(outflow, np.log(log_reg_c)*torch.ones_like(outflow)) # care less about tiny flows
            rep_d = repeat(d, "n_tran -> n_tran nq", nq=n_quantile_out)
            rep_r = repeat(r, "n_tran -> n_tran nq", nq=n_quantile_out)
            outflow_plus_r = torch.where(rep_d > 0, (rep_r+log_reg_c).log(), outflow)
            
            diff = repeat(outflow_plus_r, "b nq_out -> b 1 nq_out") - repeat(inflow, "b nq_in -> b nq_in 1")
            abs_weight = torch.abs(repeat(quantiles, "b nq_in -> b nq_in nq_out", nq_out=n_quantile_out) \
                - diff.le(0).float())
            losses = torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
            losses = (abs_weight * losses).sum(dim=-2).mean(dim=-1) # sum over qunatile_in, mean over quantile_out

            if clip_loss > 0:
                ld = losses.detach()
                losses = losses / ld * torch.minimum(ld, clip_loss)
            term_loss = (losses * d).sum() / (d.sum() + 1e-20)
            flow_loss = (losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
            if balanced_loss:
                loss = term_loss * leaf_coef + flow_loss
            else:
                loss = losses.mean()
            opt.zero_grad()
            loss.backward()
            last_losses.append((loss.item(), term_loss.item(), flow_loss.item()))

        else: # tb, db, subtb alg of gflownets
            s, a, r, d, n, mols, idc, lens, *o = minibatch
            # a: action ((blockidx, stemidx) or (-1, x) for ‘stop’)
            # mol_out_s[0] is stop action logits, mol_out_s[1] is flow function on state.
            stem_out_s, mol_out_s = model(s, None)
            # index parents by their corresponding actions
            logits = -model.action_negloglikelihood(s, a, 0, stem_out_s, mol_out_s)
            tzeros = torch.zeros(idc[-1]+1, device=device, dtype=args.floatX)
            traj_r = tzeros.index_add(0, idc, r)

            if args.obj == 'tb':
                uniform_log_PB = tzeros.index_add(0, idc, torch.log(1/n))
                traj_logits = tzeros.index_add(0, idc, logits)
                losses = ((model.logZ + traj_logits) - (torch.log(traj_r) + uniform_log_PB)).pow(2)
                loss = losses.mean()
            elif args.obj in ['db', 'detbal']:
                loss = detailed_balance_loss(logits, torch.log(1/n), mol_out_s[:, 1], torch.log(traj_r), lens)
            elif args.obj == 'subtb':
                loss = tb_lambda_loss(logits, torch.log(1/n), mol_out_s[:, 1], torch.log(traj_r), lens, Lambda)
            
            opt.zero_grad()
            loss.backward()
            last_losses.append((loss.item(),))
            train_losses.append((loss.item(),))
            if not i % 50:
                train_infos.append((
                    r.data.cpu().numpy(),
                    mols[1],
                    [i.pow(2).sum().item() for i in model.parameters()],
                ))
        
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)
        opt.step()
        model.training_steps = i + 1
        if tau > 0:
            for _a,b in zip(model.parameters(), target_model.parameters()):
                b.data.mul_(1-tau).add_(tau*_a)

        print_interval = 10 if args.debug else 1000
        if not i % print_interval:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            wandb_dict = {"loss/loss": last_losses[0]} 
            if args.obj in ['fm', 'qm']:
                wandb_dict.update({"loss/terminal": last_losses[1], "loss/flow": last_losses[2]})
            print(f"Iter={i}", " ".join([f"{k}={v:.2f}" for k, v in wandb_dict.items()]))
            time_used = time.time() - time_last_check
            print(f'Time: {time_used:.2f} sec; {time_used/print_interval:.3f} sec per step')
            time_last_check = time.time()
            last_losses = []
            
            save_interval = print_interval if args.debug else 5000
            if not i % save_interval and do_save:
                corr_logp, corr = compute_correlation(model, dataset.mdp, args)
                
                avg_topk_rs, avg_topk_tanimoto, num_modes_above_7_5, num_modes_above_8_0, \
                    num_mols_above_7_5, num_mols_above_8_0 = eval_mols(dataset.sampled_mols,
                        reward_norm=args.reward_norm, reward_exp=args.reward_exp, algo="gfn")
                avg_topk_rs_recent, avg_topk_tanimoto_recent, num_modes_above_7_5_recent, num_modes_above_8_0_recent, \
                    num_mols_above_7_5_recent, num_mols_above_8_0_recent = eval_mols(dataset.sampled_mols[-50000:],
                        reward_norm=args.reward_norm, reward_exp=args.reward_exp, algo="gfn")

                wandb_dict.update({
                    "spearman_corr": corr,
                    
                    "reward_top10_recent": avg_topk_rs_recent[10], "reward_top100_recent": avg_topk_rs_recent[100], "reward_top1000_recent": avg_topk_rs_recent[1000],
                    "tanimoto_top10_recent": avg_topk_tanimoto_recent[10], "tanimoto_top100_recent": avg_topk_tanimoto_recent[100], "tanimoto_top1000_recent": avg_topk_tanimoto_recent[1000],
                    "num_modes R>7.5 recent": num_modes_above_7_5_recent, "num_modes R>8.0 recent": num_modes_above_8_0_recent,

                    "reward_top10": avg_topk_rs[10], "reward_top100": avg_topk_rs[100], "reward_top1000": avg_topk_rs[1000],
                    "tanimoto_top10": avg_topk_tanimoto[10], "tanimoto_top100": avg_topk_tanimoto[100], "tanimoto_top1000": avg_topk_tanimoto[1000],
                    "num_modes R>7.5": num_modes_above_7_5, "num_modes R>8.0": num_modes_above_8_0,
                })
                
                save_dict["spearman_corr"][i] = corr
                save_dict["state_visited"][i] = len(dataset.sampled_mols)

                save_dict["reward_top10_recent"][i] = avg_topk_rs_recent[10]
                save_dict["reward_top100_recent"][i] = avg_topk_rs_recent[100]
                save_dict["reward_top1000_recent"][i] = avg_topk_rs_recent[1000]
                save_dict["tanimoto_top10_recent"][i] = avg_topk_tanimoto_recent[10]
                save_dict["tanimoto_top100_recent"][i] = avg_topk_tanimoto_recent[100]
                save_dict["tanimoto_top1000_recent"][i] = avg_topk_tanimoto_recent[1000]
                save_dict["num_modes R>7.5 recent"][i] = num_modes_above_7_5_recent
                save_dict["num_modes R>8.0 recent"][i] = num_modes_above_8_0_recent
                save_dict["reward_top10"][i] = avg_topk_rs[10]
                save_dict["reward_top100"][i] = avg_topk_rs[100]
                save_dict["reward_top1000"][i] = avg_topk_rs_recent[1000]
                save_dict["tanimoto_top10"][i] = avg_topk_tanimoto[10]
                save_dict["tanimoto_top100"][i] = avg_topk_tanimoto[100]
                save_dict["tanimoto_top1000"][i] = avg_topk_tanimoto[1000]
                save_dict["num_modes R>7.5"][i] = num_modes_above_7_5
                save_dict["num_modes R>8.0"][i] = num_modes_above_8_0

                print(f"Iter={i} state_visited={len(dataset.sampled_mols)}: spearman_corr={corr:.3f};"
                        f"reward_top100={avg_topk_rs[100]:.2f}; tanimoto_top100={avg_topk_tanimoto[100]:.3f}; "
                        f"num_modes R>7.5={num_modes_above_7_5};")
                print(f"                                    "
                        f"reward_top100_recent={avg_topk_rs_recent[100]:.2f}; tanimoto_top100_recent={avg_topk_tanimoto_recent[100]:.3f}; "
                        f"num_modes R>7.5 recent={num_modes_above_7_5_recent};")
                
                save_stuff(i, corr_logp)
            
            wandb_dict["state_visited"] = len(dataset.sampled_mols)
            if args.wandb:
                wandb.log(wandb_dict, step=i)

    stop_everything()
    if do_save:
        save_stuff(i, None)
    return model


def main_mols(args):
    assert args.model_version in ['v1', 'v2', 'v3', 'v4']
    seed_torch(args.seed)
    print("Args:", vars(args))

    if args.wandb:
        wandb.init(project="GFN-mol", config=args, save_code=True)

    bpath = "~/Distributional-GFlowNets/mols/data/blocks_PDB_105.json"
    device = torch.device('cuda')

    if args.floatX == 'float32':
        args.floatX = torch.float
    else:
        args.floatX = torch.double
    
    if args.obj in ['fm', 'qm']:
        dataset = Dataset(args, bpath, device, floatX=args.floatX)
    else:
        args.ignore_parents = True
        dataset = DatasetDirect(args, bpath, device, floatX=args.floatX)
    mdp = dataset.mdp

    model = make_model(args, mdp, 
        out_per_mol=1 + (1 if args.obj in ['subtb', 'subtbWS', 'detbal', "db"] else 0))
    model.to(args.floatX)
    model.to(device)

    proxy = Proxy(args, bpath, device)
    train_model_with_proxy(args, model, proxy, dataset, do_save=True)
    print('Done.')
    if args.wandb:
        wandb.finish()


def seed_torch(seed, verbose=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if verbose:
        print("==> Set seed to {:}".format(seed))

def get_mol_path_graph(mol):
    bpath = "~/Distributional-GFlowNets/mols/data/blocks_PDB_105.json"
    mdp = MolMDPExtended(bpath)
    mdp.post_init(torch.device('cpu'), 'block_graph')
    mdp.build_translation_table()
    mdp.floatX = torch.float
    agraph = nx.DiGraph()
    agraph.add_node(0)
    ancestors = [mol]
    ancestor_graphs = []

    par = mdp.parents(mol)
    mstack = [i[0] for i in par]
    pstack = [[0, a] for i,a in par]
    while len(mstack):
        m = mstack.pop() #pop = last item is default index
        p, pa = pstack.pop()
        match = False
        mgraph = mdp.get_nx_graph(m)
        for ai, a in enumerate(ancestor_graphs):
            if mdp.graphs_are_isomorphic(mgraph, a):
                agraph.add_edge(p, ai+1, action=pa)
                match = True
                break
        if not match:
            agraph.add_edge(p, len(ancestors), action=pa) #I assume the original molecule = 0, 1st ancestor = 1st parent = 1
            ancestors.append(m) #so now len(ancestors) will be 2 --> and the next edge will be to the ancestor labelled 2
            ancestor_graphs.append(mgraph)
            if len(m.blocks):
                par = mdp.parents(m)
                mstack += [i[0] for i in par]
                pstack += [(len(ancestors)-1, i[1]) for i in par]

    for u, v in agraph.edges:
        c = mdp.add_block_to(ancestors[v], *agraph.edges[(u,v)]['action'])
        geq = mdp.graphs_are_isomorphic(mdp.get_nx_graph(c, true_block=True),
                                        mdp.get_nx_graph(ancestors[u], true_block=True))
        if not geq: # try to fix the action
            block, stem = agraph.edges[(u,v)]['action']
            for i in range(len(ancestors[v].stems)):
                c = mdp.add_block_to(ancestors[v], block, i)
                geq = mdp.graphs_are_isomorphic(mdp.get_nx_graph(c, true_block=True),
                                                mdp.get_nx_graph(ancestors[u], true_block=True))
                if geq:
                    agraph.edges[(u,v)]['action'] = (block, i)
                    break
        if not geq:
            raise ValueError('could not fix action')
    for u in agraph.nodes:
        agraph.nodes[u]['mol'] = ancestors[u]
    return agraph
    

# calculate exact likelihood of GFN for given molecules
def compute_correlation(model, mdp, args):  
    device = torch.device('cuda')
    tf = lambda x: torch.tensor(x, device=device).to(args.floatX)
    tint = lambda x: torch.tensor(x, device=device).long()
    home_path = os.path.expanduser("~")
    test_mols = pickle.load(gzip.open(f'{home_path}/Distributional-GFlowNets/mols/data/some_mols_U_1k.pkl.gz'))

    logsoftmax = nn.LogSoftmax(0)
    logp = []
    reward = []
    numblocks = []

    num_test_mols = 100 if args.debug else 1000
    continue_count = 0
    for moli_idx, moli in enumerate(test_mols[:num_test_mols]):
        try:
            agraph = get_mol_path_graph(moli[1])
            reward.append(np.log(moli[0])) #
        except:
            continue_count += 1
            continue
        s = mdp.mols2batch([mdp.mol2repr(agraph.nodes[i]['mol']) for i in agraph.nodes])
        numblocks.append(len(moli[1].blocks))
        with torch.no_grad():
            stem_out_s, mol_out_s = model(s, None)  # get the mols_out_s for ALL molecules not just the end one.
        
        per_mol_out = []
        # Compute pi(a|s)
        for j in range(len(agraph.nodes)):
            a,b = s._slice_dict['stems'][j:j+2]

            stop_allowed = len(agraph.nodes[j]['mol'].blocks) >= args.min_blocks
            mp = logsoftmax(torch.cat([
                stem_out_s[a:b].reshape(-1),
                # If num_blocks < min_blocks, the model is not allowed to stop
                mol_out_s[j, :1] if stop_allowed else tf([-1000])]))
            per_mol_out.append((mp[:-1].reshape((-1, stem_out_s.shape[1])), mp[-1]))

        # When the model reaches 8 blocks, it is stopped automatically. If instead it stops before
        # that, we need to take into account the STOP action's logprob
        if len(moli[1].blocks) < 8:
            stem_out_last, mol_out_last = model(mdp.mols2batch([mdp.mol2repr(moli[1])]), None)
            mplast = logsoftmax(torch.cat([stem_out_last.reshape(-1), mol_out_last[0, :1]]))
            MSTOP = mplast[-1]
        
        # assign logprob to edges
        for u,v in agraph.edges:
            a = agraph.edges[u,v]['action']
            if a[0] == -1:
                agraph.edges[u,v]['logprob'] = per_mol_out[v][1]
            else:
                agraph.edges[u,v]['logprob'] = per_mol_out[v][0][a[1], a[0]]

        # propagate logprobs through the graph
        for n in list(nx.topological_sort(agraph))[::-1]: 
            for c in agraph.predecessors(n): 
                if len(moli[1].blocks) < 8 and c == 0:
                    agraph.nodes[c]['logprob'] = torch.logaddexp(
                        agraph.nodes[c].get('logprob', tf(-1000)),
                        agraph.edges[c, n]['logprob'] + agraph.nodes[n].get('logprob', 0) + MSTOP)
                else:
                    agraph.nodes[c]['logprob'] = torch.logaddexp(
                        agraph.nodes[c].get('logprob', tf(-1000)),
                        agraph.edges[c, n]['logprob'] + agraph.nodes[n].get('logprob',0))

        logp.append((moli, agraph.nodes[n]['logprob'].item()))  # add the first item
    
    print(f"In compute_correlation(): error-caused skipped: {continue_count} / {len(test_mols[:num_test_mols])} = {100. * float(continue_count) / len(test_mols[:num_test_mols]):.2f}% ")
    corr = stats.spearmanr([logprob for (moli, logprob) in logp], reward).correlation
    return logp, corr


import hydra
@hydra.main(config_path="configs", config_name="main_gfn") # use hydra==1.1
def main(cfg):
    class ARGS:
        pass
    args = ARGS()
    for k, v in cfg.items():
        setattr(args, k, v)

    main_mols(args)

if __name__ == '__main__':
    main()