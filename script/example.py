# %%

import matplotlib as mpl
# mpl.use("pgf")
# %matplotlib inline
# %matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%

import time

import numpy as np
import scipy as sp
from sklearn import decomposition
from sklearn.linear_model import LinearRegression
from functools import partial

# %% md

# Let's load the packages

# %%

import torch
from torch.nn import functional
import vjf
from vjf.model import VJF, RBFLDS

# %% md

# We have synthesized some data from Lorenz attractor, 3D state, 200D observation, 216 realizations, each lasts 1500 steps. Let's load the data first. ([download here](https://doi.org/10.6084/m9.figshare.14588469))

# %%
from vjf.module import RBF, bLinReg

data = np.load('lorenz_216_1500_10_200_gaussian_s0.npz')

# %%

xs = data['x']  # state
ys = data['y']  # observation
# ys = np.exp(ys)
xdim = xs.shape[-1]
ydim = ys.shape[-1]

# xs = xs[:5, ...]
# ys = ys[:5, ...]

# %% md

# Firstly we draw some of the latent trajectories.

# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')
for x in xs[::50, ...]:
    # ax.plot(*x.T, color='b', alpha=0.1, zorder=1)
    ax.scatter(*x[0, :], color='g', s=50, zorder=2)
    ax.scatter(*x[-1, :], color='r', s=50, zorder=2)

# %%
print(xs.shape)
x0 = torch.as_tensor(np.reshape(xs[:, :-1, :], (-1, 3)), dtype=torch.get_default_dtype())
x1 = torch.as_tensor(np.reshape(xs[:, 1:, :], (-1, 3)), dtype=torch.get_default_dtype())
dx = x1 - x0
print(x0.shape, dx.shape)
# blr = bLinReg(RBF(3, 10), 3)

lds = RBFLDS(100, 3, 0)

# p = x + blr(x)
# p = lds(x0)
# p = p.detach().numpy().reshape(216, -1, 3)
#
# for xi in p[::50, ...]:
#     ax.plot(*xi.T, color='C1', alpha=0.5, zorder=1)

# blr.update(dx, x0, 1.)
# p = x + blr(x)
for i in range(5):
    lds.update(x0, x1 + torch.randn_like(x1))
    print(lds.linreg.w_precision.diagonal().mean())

p = lds(x0, sampling=True)
p = p.detach().numpy().reshape(216, -1, 3)

for xi in p[::50, ...]:
    ax.plot(*xi.T, color='C2', alpha=0.5, zorder=1)


p = lds(x0 + 0.1 * torch.randn_like(x0), sampling=True)
p = p.detach().numpy().reshape(216, -1, 3)

for xi in p[::50, ...]:
    ax.plot(*xi.T, color='C3', alpha=0.5, zorder=1)

# p = lds.simulate(torch.tensor(xs[:, 0, :], dtype=torch.get_default_dtype()),
#                  500)
# print(p.shape)
# p = p.detach().numpy().transpose(1, 0, 2)
# print(p.shape)
#
# for xi in p:
#     ax.plot(*xi.T, color='C4', alpha=0.1, zorder=1)

# plt.axis('off')
plt.show()
plt.close()

# %% md

# One can see the two wings. The green/red dots are the initial/final states.

# Secondly we draw one series of observation.

# %%

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.matshow(ys[0, :, :].T, aspect='auto')
# # plt.axis('off')
# plt.show()
# plt.close()

# %% md

# Now we fit the model.

# # %%
# udim = 0
# model = VJF.make_model(ydim, xdim, udim, n_rbf=100, likelihood='gaussian', hidden_sizes=[100])
#
# # %%
# ys = np.transpose(ys, (1, 0, 2))
#
# print(ys.shape)
# qs = model.fit(ys, max_iter=20)
# mu = torch.stack([q[0] for q in qs])
# mu = mu.detach().numpy()
# # %% Then we draw the estimated states. You can see the manifold. Note that the states are subject to an arbitrary
# # affine transformation.
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.set_aspect('equal')
# for x in mu[::50, ...]:
#     ax.plot(*x.T, color='b', alpha=0.1, zorder=1)
#     ax.scatter(*x[0, :], color='g', s=50, zorder=2)
#     ax.scatter(*x[-1, :], color='r', s=50, zorder=2)
# # plt.axis('off')
# plt.show()
# plt.close()

# %% Sample future trajectory
# x_future, y_future = mdl.forecast(x0=torch.zeros(5, 3), step=10)
# print(x_future.shape, y_future.shape)
