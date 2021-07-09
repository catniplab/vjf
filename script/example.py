#%%

import matplotlib as mpl
# mpl.use("pgf")
# %matplotlib inline
# %matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%

import time

import numpy as np
import scipy as sp
from sklearn import decomposition
from sklearn.linear_model import LinearRegression
from functools import partial

#%% md

# Let's load the packages

#%%

import torch
import vjf
from vjf import online

#%% md

# We have synthesized some data from Lorenz attractor, 3D state, 200D observation, 216 realizations, each lasts 1500 steps. Let's load the data first. ([download here](https://doi.org/10.6084/m9.figshare.14588469))

#%%

data = np.load('../notebook/lorenz_216_1500_10_200_gaussian_s0.npz')

#%%

xs = data['x']  # state
ys = data['y']  # observation
us = data['u']  # control input
xdim = xs.shape[-1]
ydim = ys.shape[-1]
udim = us.shape[-1]

#%% md

# Firstly we draw some of the latent trajectories.

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')
for x in xs[::50, ...]:
    ax.plot(*x.T, color='b', alpha=0.1, zorder=1)
    ax.scatter(*x[0, :], color='g', s=50, zorder=2)
    ax.scatter(*x[-1, :], color='r', s=50, zorder=2)
# plt.axis('off')
plt.show()
plt.close()

#%% md

# One can see the two wings. The green/red dots are the initial/final states.

# Secondly we draw one series of observation.

#%%

fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(ys[0, :, :].T, aspect='auto')
# plt.axis('off')
plt.show()
plt.close()

#%% md

# Now we fit the model.

#%%

likelihood = 'gaussian'  # Gaussian observation
dynamics = 'rbf'  # RBF network dynamic model
recognizer = "mlp"  # MLP recognitiom model
rdim = 50  # number of RBFs
hdim = 100  # number of MLP hidden units

mdl = online.VJF(
    config=dict(
        resume=False,
        xdim=xdim,
        ydim=ydim,
        udim=udim,
        Ydim=udim,
        Udim=udim,
        rdim=rdim,
        hdim=hdim,
        lr=1e-3,
        clip_gradients=5.0,
        debug=True,
        likelihood=likelihood,  #
        system=dynamics,
        recognizer=recognizer,
        C=(None, True),  # loading matrix: (initial, estimate)
        b=(None, True),  # bias: (initial, estimate)
        A=(None, False),  # transition matrix if LDS
        B=(np.zeros((xdim, udim)), False),  # interaction matrix
        Q=(1.0, True),  # state noise
        R=(1.0, True),  # observation noise
    )
)

#%%

# We feed the data multiple times to help the training
# This may take some time
# n_epoch = 10
#
# for i in range(n_epoch):
#     mu, logvar, losses = mdl.filter(ys, us)
#
# mu = torch.detach(mu).numpy().squeeze()
# mu = mu.transpose(1, 0, 2)

# pseudo offline training
print(ys.shape)
mu, logvar, loss = mdl.fit(ys, us, max_iter=5)  # posterior mean, variance and loss (negative ELBO)
mu = mu.detach().numpy().squeeze()  # convert to numpy array

#%% md

# Then we draw the estimated states. You can see the manifold. Note that the states are subject to an arbitrary affine transformation.

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')
for x in mu[::50, ...]:
    ax.plot(*x.T, color='b', alpha=0.1, zorder=1)
    ax.scatter(*x[0, :], color='g', s=50, zorder=2)
    ax.scatter(*x[-1, :], color='r', s=50, zorder=2)
# plt.axis('off')
plt.show()
plt.close()

#%%


