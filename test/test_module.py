import pytest
import torch
from torch import nn

from vjf.module import RBF, LinearRegression, RBFN


class IdentityFeature(nn.Module):
    n_feature = 2

    def forward(self, x):
        return x


@pytest.mark.parametrize('precision', [
    torch.tensor([[1., 2.], [2., 1.]]),
    torch.diag(torch.tensor([0., 1.])),
])
def test_RLS_recovers_positive_definite_precision(precision):
    blr = LinearRegression(IdentityFeature(), n_output=1)
    blr.w_precision = precision

    with pytest.warns(UserWarning, match='added diagonal jitter'):
        blr.rls(torch.zeros(1, 2), torch.zeros(1, 1), 1.)

    recovered_precision = blr.w_pchol.mm(blr.w_pchol.t())
    assert torch.all(torch.linalg.eigvalsh(blr.w_precision) > 0.)
    assert torch.allclose(blr.w_precision, recovered_precision)


def test_RBF():
    n_dim, n_basis = 3, 10
    rbf = RBF(n_dim, n_basis)
    blr = LinearRegression(rbf, n_dim)

    N = 20
    x = torch.randn(N, n_dim)
    y = torch.randn(N, n_dim)
    blr(x)
    blr.kalman(y, x, 1.)


def test_RBFN():
    n_dim, n_basis = 3, 10
    rbfn = RBFN(n_dim, n_dim, n_basis)

    N = 20
    x = torch.randn(N, n_dim)
    y = torch.randn(N, n_dim)

    rbfn(x)


def test_init_srls_uses_preset_centers():
    torch.manual_seed(0)
    feat = RBF(2, 5)
    lr = LinearRegression(feat, 2, bayes=True)
    centers = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 2.]])
    logw = torch.log(torch.full((5,), 0.3))
    x = torch.randn(20, 2)
    y = torch.randn(20, 2)
    lr.init_srls(x, y, centers=centers, logwidths=logw)
    assert torch.allclose(lr.feature.centroid.data, centers)
    assert torch.allclose(lr.feature.logwidth.data, logw)
