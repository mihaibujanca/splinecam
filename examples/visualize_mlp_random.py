#!/usr/bin/env python
# coding: utf-8

import sys
import os
sys.path.append('..')

import splinecam as sc
import torch
import matplotlib.pyplot as plt

### model definition

in_shape = 10
out_shape = 1
width = 20
depth = 10
# act = torch.nn.ReLU(inplace=True)
act = torch.nn.LeakyReLU(0.03)

layers = []

layers.append(torch.nn.Linear(in_shape,width))
layers.append(act)


for i in range(depth-1):
    layers.append(torch.nn.Linear(width,width))
    layers.append(torch.nn.BatchNorm1d(width))
    layers.append(act)
    
layers.append(torch.nn.Linear(width,out_shape))

model = torch.nn.Sequential(*layers)
model.cuda()

model.eval()
model.type(torch.float64)

### define input domain to compute partitions

# Precompute random vectors z1 and z2
rng = torch.Generator()
seed = 42
rng.manual_seed(seed)  # Set seed for reproducibility

z1 = torch.randn(in_shape, generator=rng)
z2 = torch.randn(in_shape, generator=rng)

# Call the function with precomputed random vectors
domain = sc.utils.get_square_slice_from_one_anchor(
    anchors=torch.randn(1, in_shape),
    pad_dist=2,
    z1=z1,
    z2=z2
)

# compute linear projection from input space to target domain
T = sc.utils.get_proj_mat(domain)

# wrap model with splinecam library
NN = sc.wrappers.model_wrapper(
    model,
    input_shape=(in_shape,),
    T = T,
    dtype = torch.float64
)

print('forward and affine equivalency flag ', NN.verify())

### Compute regions and decision boundary

out_cyc,endpoints,Abw = sc.compute.get_partitions_with_db(domain,T,NN)

### Plot partitions

minval,_ = torch.vstack(out_cyc).min(0)
maxval,_ = torch.vstack(out_cyc).max(0)

sc.plot.plot_partition(out_cyc, xlims=[minval[0],maxval[0]],alpha=0.3,
                         edgecolor='#a70000',color_range=[.3,.8],
                         colors=['#469597', '#5BA199', '#BBC6C8', '#E5E3E4', '#DDBEAA'],
                         ylims=[minval[1],maxval[1]], linewidth=.5)

plt.savefig('../figures/mlp_visualize.jpg',transparent=True, bbox_inches=0, pad_inches=0)