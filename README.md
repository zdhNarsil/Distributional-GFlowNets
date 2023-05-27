# Distributional GFlowNet - Quantile Matching

[//]: # (---)

Code for our paper [Distributional GFlowNets with Quantile Flows](https://arxiv.org/abs/2302.05793).

<!-- <p align="center"> -->
<img src="https://s1.ax1x.com/2023/05/27/p9Lfjn1.png" border="0" width=40% class="center" />
<!-- </p> -->


We think of each edge flow as a random variable, and parameterize its quantile function in a distributional way. 
We then propose "Quantile Matching" (QM) to train the GFlowNet model based on a distributional temporal-difference-like 🤖 flow constraint.
With such risk-sensitive probabilistic flows, GFlowNet now support risk-sensitive polices to deal with uncertainty in the reward models.
To make things better, Quantile Matching even outperforms previous methods in non-stochastic environments🔬 due to richer learning signals.

[//]: # (---)

## Hypergrid task
```
python run_hydra.py ndim=4 method=fm
python run_hydra.py ndim=4 method=tb
python run_hydra.py ndim=4 method=qm N=8 quantile_dim=256
```
The last one is the proposed QM algorithm.

## Molecule task
```
python gflownet.py obj=fm
python gflownet.py obj=tb reward_exp=4 random_action_prob=0.1
python gflownet.py obj=qm
```
The last one is the proposed QM algorithm.