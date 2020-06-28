import numpy as np
import torch


def test_rbfn():
    from vjf.module import RBFN

    np.random.seed(0)
    batch_size = 5
    x_dim = 3
    r_dim = 4
    x_np = np.random.randn(batch_size, x_dim).astype(np.float32)
    c_np = np.random.randn(r_dim, x_dim).astype(np.float32)
    gamma_np = np.random.randn(r_dim).astype(np.float32)

    gamma = np.exp(gamma_np)

    phi_np = np.exp(-0.5 * gamma * np.sum(np.square(x_np[:, None, :] - c_np), axis=2))

    # torch
    with torch.no_grad():
        rbfn = RBFN(x_dim, r_dim, c_np, gamma_np)
        phi = rbfn(torch.as_tensor(x_np))

    assert np.allclose(c_np, rbfn.c.data)
    assert np.allclose(gamma_np, rbfn.logwidth.data)
    assert np.allclose(phi_np, phi)


def test_velocity():
    from vjf.dynamics import RBFS

    np.random.seed(0)
    batch_size = 5
    x_dim = 3
    r_dim = 4
    udim = 1
    x_np = np.random.randn(batch_size, x_dim).astype(np.float32)
    c_np = np.random.randn(r_dim, x_dim).astype(np.float32)
    gamma_np = np.random.randn(r_dim).astype(np.float32)

    gamma = np.exp(gamma_np)

    phi_np = np.exp(-0.5 * gamma * np.sum(np.square(x_np[:, None, :] - c_np), axis=2))

    W_np = np.random.randn(r_dim, x_dim)
    f_np = phi_np @ W_np

    # torch
    B = (np.zeros((udim, x_dim)), False)
    Q = (1.0, False)
    with torch.no_grad():
        rbfs = RBFS(x_dim, r_dim, udim, B, Q, c_np, gamma_np)
        rbfs.transition.weight.data = torch.tensor(W_np.T, dtype=torch.float)
        f = rbfs.velocity(torch.as_tensor(x_np))

    assert np.allclose(f_np, f)
