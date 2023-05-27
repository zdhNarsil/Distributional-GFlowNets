import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn
import torch_scatter

import math
from einops import rearrange, reduce, repeat

def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    net = nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))
    return net

class GraphAgent(nn.Module):

    def __init__(self, nemb, nvec, out_per_stem, out_per_mol, num_conv_steps, mdp_cfg, version='v1'):
        super().__init__()
        print("GFN architecture version:", version) # "v4"
        if version == 'v5': 
            version = 'v4'
        self.version = version
        self.embeddings = nn.ModuleList([
            nn.Embedding(mdp_cfg.num_true_blocks + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types, nemb)])
        self.conv = gnn.NNConv(nemb, nemb, nn.Sequential(), aggr='mean')
        nvec_1 = nvec * (version == 'v1' or version == 'v3')
        nvec_2 = nvec * (version == 'v2' or version == 'v3')
        self.block2emb = nn.Sequential(nn.Linear(nemb + nvec_1, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb))

        self.gru = nn.GRU(nemb, nemb)
        self.stem2pred = nn.Sequential(nn.Linear(nemb * 2 + nvec_2, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, out_per_stem))
        self.global2pred = nn.Sequential(nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                         nn.Linear(nemb, out_per_mol))
        self.num_conv_steps = num_conv_steps
        self.nemb = nemb
        self.training_steps = 0
        self.categorical_style = 'softmax'
        self.escort_p = 6

    # vec_data is for conditioning input
    def forward(self, graph_data, vec_data=None, do_stems=True):
        blockemb, stememb, bondemb = self.embeddings
        graph_data.x = blockemb(graph_data.x)
        if do_stems:
            graph_data.stemtypes = stememb(graph_data.stemtypes)
        graph_data.edge_attr = bondemb(graph_data.edge_attr)
        graph_data.edge_attr = (
            graph_data.edge_attr[:, 0][:, :, None] * graph_data.edge_attr[:, 1][:, None, :]
        ).reshape((graph_data.edge_index.shape[1], self.nemb**2))

        out = graph_data.x
        if self.version == 'v1' or self.version == 'v3':
            batch_vec = vec_data[graph_data.batch]
            out = self.block2emb(torch.cat([out, batch_vec], 1))
        else:  # if self.version == 'v2' or self.version == 'v4':
            out = self.block2emb(out)

        h = out.unsqueeze(0)
        for i in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv(out, graph_data.edge_index, graph_data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        # Index of the origin block of each stem in the batch (each
        # stem is a pair [block idx, stem atom type], we need to
        # adjust for the batch packing)
        if do_stems:
            if hasattr(graph_data, '_slice_dict'):
                x_slices = torch.tensor(graph_data._slice_dict['x'], device=out.device)[graph_data.stems_batch]
            else:
                x_slices = torch.tensor(graph_data.__slices__['x'], device=out.device)[graph_data.stems_batch]
            stem_block_batch_idx = (x_slices + graph_data.stems[:, 0])
            if self.version == 'v1' or self.version == 'v4':
                stem_out_cat = torch.cat([out[stem_block_batch_idx], graph_data.stemtypes], 1)
            elif self.version == 'v2' or self.version == 'v3':
                stem_out_cat = torch.cat([out[stem_block_batch_idx],
                                          graph_data.stemtypes,
                                          vec_data[graph_data.stems_batch]], 1)

            stem_preds = self.stem2pred(stem_out_cat)
        else:
            stem_preds = None
        mol_preds = self.global2pred(gnn.global_mean_pool(out, graph_data.batch))
        return stem_preds, mol_preds  # per stem output, per molecule output

    def out_to_policy(self, s, stem_o, mol_o):
        assert self.categorical_style == 'softmax'
        size = int(s.stems_batch.max().item() + 1)
        Z_log = torch.logaddexp(
                    torch_scatter.scatter_logsumexp(
                        stem_o, s.stems_batch, dim=-2, dim_size=size)
                    .logsumexp(dim=-1),
                    mol_o[:, 0]
                )
        return mol_o[:, 0] - Z_log, stem_o - Z_log[s.stems_batch, None]

    def action_negloglikelihood(self, s, a, g, stem_o, mol_o):
        mol_lsm, stem_lsm = self.out_to_policy(s, stem_o, mol_o)
        return -self.index_output_by_action(s, stem_lsm, mol_lsm, a)

    def index_output_by_action(self, s, stem_o, mol_o, a):
        if hasattr(s, '_slice_dict'):
            stem_slices = torch.tensor(s._slice_dict['stems'][:-1], dtype=torch.long, device=stem_o.device)
        else:
            stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_o.device)
            
        return (
                stem_o[stem_slices + a[:, 1]][torch.arange(a.shape[0]), a[:, 0]] \
                    * (a[:, 0] >= 0)
                + mol_o * (a[:, 0] == -1)
            )

    def sum_output(self, s, stem_o, mol_o):
        return gnn.global_add_pool(stem_o, s.stems_batch).sum(1) + mol_o


class DistGraphAgent(nn.Module):

    def __init__(self, nemb, nvec, out_per_stem, out_per_mol, num_conv_steps, mdp_cfg, 
        version='v1', quantile_dim=-1, n_quantiles=-1, thompson_sampling=False):
        super().__init__()
        print("GFN architecture version:", version)
        if version == 'v5': 
            version = 'v4'
        self.version = version
        self.mdp = mdp_cfg
        self.embeddings = nn.ModuleList([
            nn.Embedding(mdp_cfg.num_true_blocks + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types, nemb)])
        
        self.conv_orig = gnn.NNConv(nemb, nemb, nn.Sequential(), aggr='mean')
        self.conv = gnn.NNConv(nemb, nemb, nn.Sequential(), aggr='mean')

        nvec_1 = nvec * (version == 'v1' or version == 'v3')
        nvec_2 = nvec * (version == 'v2' or version == 'v3')
        self.block2emb = nn.Sequential(nn.Linear(nemb + nvec_1, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb))
        self.block2emb_orig = nn.Sequential(nn.Linear(nemb, nemb), nn.LeakyReLU(), nn.Linear(nemb, nemb))

        self.gru = nn.GRU(nemb, nemb)
        self.stem2pred = nn.Sequential(nn.Linear(nemb * 2 + nvec_2, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, out_per_stem))
        self.stem2pred_orig = nn.Sequential(nn.Linear(nemb * 2, nemb), nn.LeakyReLU(),
            nn.Linear(nemb, nemb), nn.LeakyReLU(), nn.Linear(nemb, out_per_stem))                                       
        self.global2pred = nn.Sequential(nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                         nn.Linear(nemb, out_per_mol))
        self.num_conv_steps = num_conv_steps
        self.nemb = nemb
        self.training_steps = 0
        self.categorical_style = 'softmax'

        # assert version == 'v3' # ?
        # assert nvec > 0 #
        assert quantile_dim > 0 and n_quantiles > 0
        self.quantile_dim = quantile_dim
        self.iqn = True
        self.n_quantiles = n_quantiles
        self.nvec = nvec
        self.thompson_sampling = thompson_sampling
        self.register_buffer("feature_id", torch.arange(1, 1 + self.quantile_dim))
        self.phi = make_mlp([self.quantile_dim,] + [nvec]*2) 

    def forward_with_quantile(self, graph_data, quantiles, do_stems=True, pdb=False):
        batch_size, n_quantiles = quantiles.shape
        assert batch_size == graph_data.batch.max() + 1
        
        feature_id = repeat(self.feature_id, "d -> n b d", b=batch_size, n=n_quantiles)
        quantiles_rep = repeat(quantiles, "b n -> n b d", d=self.quantile_dim)
        cos = torch.cos(math.pi * feature_id * quantiles_rep) # (bs, n_quantiles, d)
        vec_data = F.relu(self.phi(cos)) # (n_mol, n_quantiles, d)
        # graph_data.batch: edge_idx -> mol_idx
        batch_vec = torch.gather(vec_data, 1, 
            repeat(graph_data.batch, "nnode -> nq nnode d", nq=n_quantiles, d=self.nvec))

        blockemb, stememb, bondemb = self.embeddings
        graph_data.x = blockemb(graph_data.x)
        if do_stems:
            graph_data.stemtypes = stememb(graph_data.stemtypes)
        graph_data.edge_attr = bondemb(graph_data.edge_attr)
        graph_data.edge_attr = (
            graph_data.edge_attr[:, 0][:, :, None] * graph_data.edge_attr[:, 1][:, None, :]
        ).reshape((graph_data.edge_index.shape[1], self.nemb**2))
        if pdb:
            print(graph_data)

        out_orig = graph_data.x
        out = repeat(out_orig, "nnode d -> nq nnode d", nq=n_quantiles)
        if self.version == 'v1' or self.version == 'v3':
            out = self.block2emb(torch.cat([out, batch_vec], dim=-1))
        elif self.version == 'v2' or self.version == 'v4':
            out = self.block2emb(out)
        else: raise NotImplementedError
        out = repeat(out, "nq nnode d -> (nq nnode) d")

        num_edge = graph_data.edge_index.shape[1]
        num_node = graph_data.x.shape[0]
        # don't know how to add additional dim for message passing layer
        rep_edge_index = repeat(graph_data.edge_index, "two ne -> two n ne", n=n_quantiles) + \
            repeat(torch.arange(n_quantiles).to(graph_data.x).long() * num_node, "n -> 2 n ne", ne=num_edge) 
        rep_edge_index = rearrange(rep_edge_index, "two n ne -> two (n ne)")
        rep_edge_attr = repeat(graph_data.edge_attr, "ne d -> (n ne) d", n=n_quantiles)

        # may need to rewrite the gnn.NNConv to enable more input dim?
        # self.conv is torch_geometric.nn.NNConv(256, 256, nn.Sequential(), aggr='mean')
        # out: (#node, 256)  edge_index: (2, #edge)  edge_attr: (#edge, 256^2)
        h = out.unsqueeze(0) 
        for i in range(self.num_conv_steps):
            # self.conv.bias + self.conv.lin(out) + self.conv.propagate(graph_data.edge_index, x=(out,out), edge_attr=graph_data.edge_attr)
            m = F.leaky_relu(self.conv(out, rep_edge_index, rep_edge_attr))
            # I think we shud actually use a gru cell here;
            # Currently since it is a gru, we have to to unsqueeze every time...
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        out = rearrange(out, "(nq nnode) d -> nnode nq d", nq=n_quantiles)
        batch_vec = rearrange(batch_vec, "nq idx d -> idx nq d")
        # Index of the origin block of each stem in the batch (each
        # stem is a pair [block idx, stem atom type], we need to
        # adjust for the batch packing)
        if do_stems:
            if hasattr(graph_data, '_slice_dict'):
                x_slices = torch.tensor(graph_data._slice_dict['x'], device=out.device)[graph_data.stems_batch]
            else:
                x_slices = torch.tensor(graph_data.__slices__['x'], device=out.device)[graph_data.stems_batch]
            stem_block_batch_idx = (x_slices + graph_data.stems[:, 0])

            rep_stemtypes = repeat(graph_data.stemtypes, "idk d -> idk nq d", nq=n_quantiles)
            if self.version == 'v1' or self.version == 'v4':
                stem_out_cat = torch.cat([out[stem_block_batch_idx], rep_stemtypes], dim=-1) # (?, n_quantile, 2*nemb+nvec_2)
            elif self.version == 'v2' or self.version == 'v3':
                stem_out_cat = torch.cat([out[stem_block_batch_idx], rep_stemtypes, 
                    batch_vec[stem_block_batch_idx]], dim=-1)
            stem_preds = self.stem2pred(stem_out_cat)
        else:
            stem_preds = None
        
        dim_size = int(graph_data.batch.max().item() + 1)
        mol_preds = torch_scatter.scatter(out, graph_data.batch, dim=0, dim_size=dim_size, reduce='mean')
        mol_preds = self.global2pred(mol_preds)

        # (n_stem nq mdp.num_blocks=105), (n_mols nq 1)
        return stem_preds, mol_preds  # per stem output, per molecule output
    
    def forward(self, graph_data, vec_data=None, do_stems=True, pdb=False):
        dtype = next(self.parameters()).dtype
        quantiles = torch.rand(int(graph_data.batch.max() + 1), self.n_quantiles,
            dtype=dtype, device=graph_data.x.device)
        stem_preds, mol_preds = self.forward_with_quantile(graph_data, quantiles)
        return stem_preds.mean(dim=1), mol_preds.mean(dim=1)
    
    def forward_orig(self, graph_data, vec_data=None, do_stems=True, pdb=False):
        # return self.forward(graph_data, pdb=pdb)

        assert vec_data is None
        blockemb, stememb, bondemb = self.embeddings
        graph_data.x = blockemb(graph_data.x)
        if do_stems:
            graph_data.stemtypes = stememb(graph_data.stemtypes)
        graph_data.edge_attr = bondemb(graph_data.edge_attr)
        graph_data.edge_attr = (
            graph_data.edge_attr[:, 0][:, :, None] * graph_data.edge_attr[:, 1][:, None, :]
        ).reshape((graph_data.edge_index.shape[1], self.nemb**2)) # ？

        out = graph_data.x
        # if self.version == 'v1' or self.version == 'v3':
        #     batch_vec = vec_data[graph_data.batch]
        #     out = self.block2emb(torch.cat([out, batch_vec], 1))
        # else:  # if self.version == 'v2' or self.version == 'v4':
        out = self.block2emb_orig(out)

        h = out.unsqueeze(0)
        for i in range(self.num_conv_steps):
            # m = F.leaky_relu(self.conv(out, graph_data.edge_index, graph_data.edge_attr))
            m = F.leaky_relu(self.conv_orig(out, graph_data.edge_index, graph_data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        # Index of the origin block of each stem in the batch (each
        # stem is a pair [block idx, stem atom type], we need to
        # adjust for the batch packing)
        if do_stems:
            if hasattr(graph_data, '_slice_dict'):
                x_slices = torch.tensor(graph_data._slice_dict['x'], device=out.device)[graph_data.stems_batch]
            else:
                x_slices = torch.tensor(graph_data.__slices__['x'], device=out.device)[graph_data.stems_batch]
            stem_block_batch_idx = (x_slices + graph_data.stems[:, 0])
            # if self.version == 'v1' or self.version == 'v4':
            stem_out_cat = torch.cat([out[stem_block_batch_idx], graph_data.stemtypes], 1)
            # elif self.version == 'v2' or self.version == 'v3':
            #     stem_out_cat = torch.cat([out[stem_block_batch_idx],
            #                               graph_data.stemtypes,
            #                               vec_data[graph_data.stems_batch]], 1)
            stem_preds = self.stem2pred_orig(stem_out_cat)
        else:
            stem_preds = None
        mol_preds = self.global2pred(gnn.global_mean_pool(out, graph_data.batch))
        return stem_preds, mol_preds  # per stem output, per molecule output

    def out_to_policy(self, s, stem_o, mol_o): # not used?
        assert self.categorical_style == 'softmax'
        size = int(s.stems_batch.max().item() + 1)
        Z_log = torch.logaddexp(
                    torch_scatter.scatter_logsumexp(
                        stem_o, s.stems_batch, dim=-2, dim_size=size)
                    .logsumexp(dim=-1),
                    mol_o[:, 0]
                )
        return mol_o[:, 0] - Z_log, stem_o - Z_log[s.stems_batch, None]

    def action_negloglikelihood(self, s, a, g, stem_o, mol_o): # for DB / TB / SubTB
        mol_lsm, stem_lsm = self.out_to_policy(s, stem_o, mol_o)
        return -self.index_output_by_action(s, stem_lsm, mol_lsm, a)

    def index_output_by_action(self, s, stem_o, mol_o, a, pdb=False):
        if hasattr(s, '_slice_dict'):
            stem_slices = torch.tensor(s._slice_dict['stems'][:-1], dtype=torch.long, device=stem_o.device)
        else:
            stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_o.device)
        
        if mol_o.ndim > a[:, 0].ndim: # with quantile, one additional dimension
            logp_nostop = stem_o[stem_slices + a[:, 1]][torch.arange(a.shape[0]), ..., a[:, 0]] *\
                repeat(a[:, 0] >= 0, "n_mols -> n_mols 1")
            logp_stop = mol_o * repeat(a[:, 0] == -1, "n_mols -> n_mols 1")
        else: # original version
            logp_nostop = stem_o[stem_slices + a[:, 1]][torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            logp_stop = mol_o * (a[:, 0] == -1)
        return logp_nostop + logp_stop
                
    def sum_output(self, s, stem_o, mol_o):
        return gnn.global_add_pool(stem_o, s.stems_batch).sum(1) + mol_o


def mol2graph(mol, mdp, floatX=torch.float, bonds=False, nblocks=False):
    f = lambda x: torch.tensor(x, dtype=torch.long, device=mdp.device)
    if len(mol.blockidxs) == 0:
        data = Data(  # There's an extra block embedding for the empty molecule
            x=f([mdp.num_true_blocks]),
            edge_index=f([[], []]),
            edge_attr=f([]).reshape((0, 2)),
            stems=f([(0, 0)]),
            stemtypes=f([mdp.num_stem_types]))  # also extra stem type embedding
        return data
    edges = [(i[0], i[1]) for i in mol.jbonds]
    t = mdp.true_blockidx
    edge_attrs = [(mdp.stem_type_offset[t[mol.blockidxs[i[0]]]] + i[2],
                   mdp.stem_type_offset[t[mol.blockidxs[i[1]]]] + i[3])
                  for i in mol.jbonds]
    # Here stem_type_offset is a list of offsets to know which
    # embedding to use for a particular stem. Each (blockidx, atom)
    # pair has its own embedding.
    stemtypes = [mdp.stem_type_offset[t[mol.blockidxs[i[0]]]] + i[1] for i in mol.stems]

    data = Data(x=f([t[i] for i in mol.blockidxs]),
                edge_index=f(edges).T if len(edges) else f([[],[]]),
                edge_attr=f(edge_attrs) if len(edges) else f([]).reshape((0,2)),
                stems=f(mol.stems) if len(mol.stems) else f([(0,0)]),
                stemtypes=f(stemtypes) if len(mol.stems) else f([mdp.num_stem_types]))
    data.to(mdp.device)
    assert not bonds and not nblocks
    return data


def mols2batch(mols, mdp):
    batch = Batch.from_data_list(mols, follow_batch=['stems'])
    batch.to(mdp.device)
    return batch