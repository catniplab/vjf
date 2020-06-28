import pytest
import torch


@pytest.mark.parametrize("n, m, xdim, ydim", [(10, 5, 3, 2)])
def test_sgp(n, m, xdim, ydim):
    import numpy as np
    from vjf.gp import SGP
    from vjf.gp.covfun import SquaredExponential

    torch.set_default_dtype(torch.double)

    A = np.random.randn(xdim, ydim)
    x = np.random.randn(n, xdim)
    y = x @ A
    inducing = np.random.randn(m, xdim)
    covfun = SquaredExponential(1.0, 0.1)
    sgp = SGP(xdim, ydim, 0, covfun, noise_var=0.0, f_cov="I", inducing=inducing)
    sgp.initialize()
    sgp.predict(x)
    sgp.fit(x, y)
    sgp.predict(x)
