# %%
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from vjf.model import VJF

# %%
# using double precision
torch.set_default_dtype(torch.double)
np.random.seed(0)
torch.manual_seed(0)

# %% Generate data
T = 100.  # length
dt = 1e-2 * math.pi  # time step
xdim = 2  # state dimensionality
ydim = 20  # obsetvation dimensionality
udim = 0  # size of input

C = torch.randn(xdim, ydim)  # loading matrix
d = torch.randn(ydim)  # bias

t = torch.arange(0, T, step=dt)  # time point evaluated
print(t.shape)
x = 2 * torch.column_stack((torch.sin(t), torch.cos(t)))  # limit cycle
x = x + torch.randn_like(x) * 0.1  # add some noise

# observation
# y = torch.poisson(torch.exp(x @ C + d))
y = x @ C + d
y = y + torch.randn_like(y) * 0.1  # add noise

fig = plt.figure()
ax = fig.add_subplot(221)
ax.plot(*x.numpy().T)
plt.title('True state')

# %% Fit VJF
n_rbf = 100  # number of radial basis functions
hidden_sizes = [20]  # structure of MLP
likelihood = 'gaussian'  # gaussian or poisson
# likelihood = 'poisson'  # gaussian or poisson

model = VJF.make_model(ydim, xdim, udim=udim, n_rbf=n_rbf, hidden_sizes=hidden_sizes, likelihood=likelihood)
ax.scatter(*model.transition.velocity.feature.centroid.detach().numpy().T, s=10, c='r')

c0 = model.transition.velocity.feature.centroid.clone()

m, logvar, _ = model.fit(y, max_iter=100, warm_up=True, clip_value=5., gamma=0.99, update=True)  # return list of state posterior tuples (mean, log variance)

c1 = model.transition.velocity.feature.centroid.clone()

print(torch.norm(c0 - c1).item())

m = m.detach().numpy().squeeze()

# %% draw the latent trajectory
ax = fig.add_subplot(222)
ax.plot(m)
plt.title('Posterior mean')

# Draw velocity field
# make mesh for velocity field
def grid(n, lims):
    xedges = np.linspace(*lims, n)
    yedges = np.linspace(*lims, n)
    X, Y = np.meshgrid(xedges, yedges)
    grids = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    return X, Y, grids

ax = fig.add_subplot(223)
r = np.mean(np.abs(m).max())  # determine the limits of plot

Xm, Ym, XYm = grid(51, [-1.5*r, 1.5*r])
Um, Vm = model.transition.velocity(torch.tensor(XYm)).detach().numpy().T  # get velocity
Um = np.reshape(Um, Xm.shape)
Vm = np.reshape(Vm, Ym.shape)
plt.streamplot(Xm, Ym, Um, Vm)
plt.scatter(*model.transition.velocity.feature.centroid.detach().numpy().T, s=10, c='r')
plt.plot(*m.T, color='C1', alpha=0.5, zorder=5)
plt.title('Velocity field')

# %% Forecast state and observation
fig.add_subplot(224)
x, y = model.forecast(x0=m[9, ...], n_step=int(100 / dt), noise=False)
x = x.detach().numpy().squeeze()
plt.plot(x)
plt.title('Forecast')

plt.tight_layout()
plt.show()
plt.close()

# %%
