import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from vjf.model import VJF


def grid(n, lims):
    xedges = np.linspace(*lims, n)
    yedges = np.linspace(*lims, n)
    X, Y = np.meshgrid(xedges, yedges)
    grids = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    return X, Y, grids


torch.set_default_dtype(torch.double)
np.random.seed(0)
torch.manual_seed(0)

T = 100.
dt = 1e-2 * math.pi
xdim = 2  # state dimensionality
ydim = 20  # obsetvation dimensionality
udim = 0  # size of input
n_rbf = 100  # number of radial basis functions
hidden_sizes = [20]  # structure of MLP

C = torch.randn(xdim, ydim)  # loading matrix
d = torch.randn(ydim)  # bias

t = torch.arange(0, T, step=dt)
x = torch.column_stack((torch.sin(t), torch.cos(t)))
x = x + torch.randn_like(x) * 0.1  # add noise

y = x @ C + d
y = y + torch.randn_like(y) * 0.1  # add noise

fig = plt.figure()
ax = fig.add_subplot(221)
ax.plot(*x.numpy().T)
plt.title('True state')


model = VJF.make_model(ydim, xdim, udim=udim, n_rbf=n_rbf, hidden_sizes=hidden_sizes, likelihood='gaussian')

q_seq = model.fit(y, max_iter=150)

m = torch.stack([q.mean for q in q_seq])  # collect posterior mean
m = m.detach().numpy().squeeze()

ax = fig.add_subplot(222)
ax.plot(m)
plt.title('Posterior mean')

ax = fig.add_subplot(223)
r = np.mean(np.abs(m).max())

Xm, Ym, XYm = grid(51, [-1.5*r, 1.5*r])
Um, Vm = model.transition.velocity(torch.tensor(XYm, dtype=torch.get_default_dtype())).detach().numpy().T
Um = np.reshape(Um, Xm.shape)
Vm = np.reshape(Vm, Ym.shape)
plt.streamplot(Xm, Ym, Um, Vm)
plt.scatter(*model.transition.velocity.feature.centroid.detach().numpy().T, s=10, c='r')
plt.plot(*m.T, color='C1', alpha=0.5, zorder=5)
plt.title('Velocity field')

ax = fig.add_subplot(224)
x, _ = model.forecast(x0=m[9, ...], n_step=int(100 / dt), noise=False)
x = x.detach().numpy().squeeze()
plt.plot(x)
plt.title('Forecast')

plt.tight_layout()
plt.show()
plt.close()
