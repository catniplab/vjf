# %%
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from vjf.driven import VJF, train

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

t = torch.arange(0, T, step=dt) * .5  # time point evaluated
print(t.shape)
x = torch.column_stack((torch.sin(t), torch.cos(t)))  # limit cycle
x = x + torch.randn_like(x) * 0.1  # add some noise

# observation
# y = torch.poisson(torch.exp(x @ C + d))
y = x @ C + d
y = y + torch.randn_like(y) * 0.1  # add noise

u = [torch.zeros(y.shape[0], udim)]

fig = plt.figure()
ax = fig.add_subplot(221)
ax.plot(x.numpy())
plt.title('True state')

# %% Fit VJF
n_rbf = 20  # number of radial basis functions
n_layer = 2
edim = 20  # encoder
likelihood = 'gaussian'  # gaussian or poisson
normalized_rbfn = True
# likelihood = 'poisson'  # gaussian or poisson

model = VJF.make_model(ydim,
                       xdim,
                       udim,
                       n_rbf=n_rbf,
                       n_layer=n_layer,
                       ds_bias=False,
                       normalized_rbfn=normalized_rbfn,
                       edim=edim,
                       likelihood=likelihood,
                       ds='rbf',
                       state_logvar=-5.)
# %%
y = [y]

losses, mu, logvar = train(model, y, u, max_iter=1000, lr=1e-3)

mu = mu[0].detach().numpy().squeeze()

# m = m.detach().numpy().squeeze()

# %% draw the latent trajectory
ax = fig.add_subplot(222)
ax.plot(mu)
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
r = np.mean(np.abs(mu).max())  # determine the limits of plot

Xm, Ym, XYm = grid(51, [-1.1 * r, 1.1 * r])
Um, Vm = model.transition.velocity(
    torch.tensor(XYm), torch.zeros(XYm.shape[0],
                                   udim)).detach().numpy().T  # get velocity
Um = np.reshape(Um, Xm.shape)
Vm = np.reshape(Vm, Ym.shape)
plt.streamplot(Xm, Ym, Um, Vm)
# if hasattr(model.transition.predict, 'center'):
#     center = model.transition.predict.center.detach().numpy()
#     print(center.shape)
#     plt.scatter(*center.T, s=10, c='r')
plt.plot(*mu.T, color='C1', alpha=0.5, zorder=5)
plt.title('Velocity field')

# %% Forecast state and observation
ax = fig.add_subplot(224)
x, y = model.forecast(x0=mu[9, ...],
                      u=torch.zeros(1, int(100 / dt), udim),
                      n_step=int(100 / dt),
                      noise=True)
x = x.detach().numpy().squeeze()
plt.plot(x)
plt.title('Forecast')

plt.tight_layout()
plt.show()
plt.close()

# %%
