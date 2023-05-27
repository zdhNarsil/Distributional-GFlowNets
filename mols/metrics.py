import gzip
import pickle
import rdkit.DataStructs
from rdkit import Chem
import numpy as np


def get_tanimoto_pairwise(mols):
    fps = [Chem.RDKFingerprint(i.mol) for i in mols]
    pairwise_sim = []
    for i in range(len(mols)):
        pairwise_sim.extend(rdkit.DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    return pairwise_sim


class NumModes:
    def __init__(self, reward_exp, reward_norm, reward_thr=8, tanimoto_thr=0.7):
        self.reward_exp = reward_exp
        self.reward_norm = reward_norm
        self.reward_thr = reward_thr
        self.tanimoto_thr = tanimoto_thr
        self.modes = []
        self.max_reward = -1000
    def __call__(self, batch):
        candidates = []
        for some in batch:
            reward, mol = some[0], some[1]
            reward = (reward ** (1/self.reward_exp)) * self.reward_norm
            if reward > self.max_reward: 
                self.max_reward = reward
            if reward > self.reward_thr:
                candidates.append(mol)
        if len(candidates) > 0:
            # add one mode if needed
            if len(self.modes)==0: 
                self.modes.append(Chem.RDKFingerprint(candidates[0].mol))
            for mol in candidates:
                fp = Chem.RDKFingerprint(mol.mol)
                sims = np.asarray(rdkit.DataStructs.BulkTanimotoSimilarity(fp, self.modes))
                if all(sims < self.tanimoto_thr):
                    self.modes.append(fp)
        return self.max_reward, len(self.modes)


def eval_mols(mols, reward_norm=8, reward_exp=10, algo="gfn"):
    def r2r_back(r):
        return r ** (1. / reward_exp) * reward_norm
    
    numModes_above_7_5 = NumModes(reward_exp=reward_exp, reward_norm=reward_norm, reward_thr=7.5)
    _, num_modes_above_7_5 = numModes_above_7_5(mols)
    numModes_above_8_0 = NumModes(reward_exp=reward_exp, reward_norm=reward_norm, reward_thr=8.)
    _, num_modes_above_8_0 = numModes_above_8_0(mols)

    top_ks = [10, 100, 1000]
    avg_topk_rs = {}
    avg_topk_tanimoto = {}
    mol_r_map = {}

    for i in range(len(mols)):
        if algo == 'gfn':
            r, m, trajectory_stats, inflow = mols[i]
        else:
            r, m = mols[i]
        r = r2r_back(r)
        mol_r_map[m] = r
    
    unique_rs = list(mol_r_map.values())
    unique_rs = sorted(unique_rs, reverse=True)
    unique_rs = np.array(unique_rs)
    num_above_7_5 = np.sum(unique_rs > 7.5) # just a integer
    num_above_8_0 = np.sum(unique_rs > 8.0)

    sorted_mol_r_map = sorted(mol_r_map.items(), key=lambda kv: kv[1], reverse=True)
    for top_k_idx, top_k in enumerate(top_ks):
        avg_topk_rs[top_k] = np.mean(unique_rs[:top_k])
        
        topk_mols = [mol for (mol, r) in sorted_mol_r_map[:top_k]]
        avg_topk_tanimoto[top_k] = np.mean(get_tanimoto_pairwise(topk_mols))

    return avg_topk_rs, avg_topk_tanimoto, num_modes_above_7_5, num_modes_above_8_0, num_above_7_5, num_above_8_0