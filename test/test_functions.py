import torch

from vjf.distributions import Gaussian
from vjf.functions import GaussianLoss


def test_gaussian():
    loss = GaussianLoss()
    print(loss.compute)
    print(
        loss.compute(x=torch.tensor(0.0, dtype=torch.float),
                     mean=torch.tensor(0.0, dtype=torch.float),
                     logvar=torch.tensor(0.0, dtype=torch.float))
    )
    mean = Gaussian(mean=torch.tensor(0.0, dtype=torch.float), logvar=torch.tensor(0.0, dtype=torch.float))
    marginal = loss.integrate("mean", mean)
    print(
        marginal.compute(x=torch.tensor(0.0, dtype=torch.float), logvar=torch.tensor(0.0, dtype=torch.float))
    )
    print(loss.compute)
    print(marginal.compute)
    loss = GaussianLoss()
    print(loss.compute)

