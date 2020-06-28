import torch


def gaussian_entropy(logvar):
    return 0.5 * torch.mean(torch.sum(logvar, dim=-1))


def elbo(q0, q1, target, decoder, system, sample, regularize):
    y, u = target
    ll_reconstruction = -decoder.loss(q1, y)

    ll_dynamics = -system.loss(q0, q1, u, sample, regularize)

    entropy = gaussian_entropy(q1[1])

    return ll_reconstruction, ll_dynamics, entropy
