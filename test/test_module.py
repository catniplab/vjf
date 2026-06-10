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


def test_grow_basis_shapes_and_prediction():
    import math
    torch.manual_seed(0)
    lr = LinearRegression(RBF(2, 5), 2, bayes=True)
    x, y = torch.randn(30, 2), torch.randn(30, 2)
    lr.init_srls(x, y)
    probe = torch.randn(7, 2)
    pred_before = lr(probe, sampling=False).mean.clone()
    lr.grow_basis(torch.tensor([[3.0, -2.0]]), logwidth=math.log(0.3))
    assert lr.feature.centroid.shape == (6, 2)
    assert lr.feature.logwidth.shape == (6,)
    assert lr.feature.n_basis == 6
    assert lr.w_mean.shape == (6, 2) and lr.w_chol.shape == (6, 6)
    # the new center has a zero weight row -> the velocity prediction is unchanged everywhere
    assert torch.allclose(pred_before, lr(probe, sampling=False).mean, atol=1e-5)
    # srls still runs on the grown basis (correct shapes) and moves the weights
    w0 = lr.w_mean.clone()
    lr.srls(torch.randn(4, 2), torch.randn(4, 2), 0.1)
    assert lr.w_mean.shape == (6, 2) and not torch.allclose(w0, lr.w_mean)


def test_rbfds_grow_triggers_on_far_state_only():
    from vjf.model import RBFDS
    torch.manual_seed(0)
    ds = RBFDS(n_rbf=4, xdim=2, udim=0, flow_learner='srrls')
    xs = torch.randn(50, 2) * 0.05
    ds.initialize(xs + torch.randn(50, 2) * 0.01, xs)         # centers on a small cloud
    ds.grow_rbf, ds.grow_thresh, ds.grow_min_gap, ds.max_rbf = True, 0.5, 1, 50
    n0 = ds.velocity.feature.n_basis
    # a state far from every center -> uncovered -> add one center
    ds.update(torch.tensor([[5.01, -4.99]]), torch.tensor([[5.0, -5.0]]))
    assert ds.velocity.feature.n_basis == n0 + 1
    # a state right on an existing center -> covered -> no growth
    n1 = ds.velocity.feature.n_basis
    c0 = ds.velocity.feature.centroid[0:1].clone()
    ds.update(c0 + 0.001, c0)
    assert ds.velocity.feature.n_basis == n1
    # cap is respected
    ds.max_rbf = ds.velocity.feature.n_basis
    ds.update(torch.tensor([[-9.0, 9.0]]), torch.tensor([[-9.01, 9.01]]))
    assert ds.velocity.feature.n_basis == n1
