"""Kalman filter and smoother
x: state
y: observation
x = Ax + w(Q)
y = Hx + v(R)
"""
from typing import Tuple

import torch
from torch import Tensor, linalg

from .numerical import positivize


@torch.no_grad()
def predict(
        x: Tensor,
        V: Tensor,
        A: Tensor,
        Q: Tensor,
        H: Tensor,
        R: Tensor,
        cholesky=True,
) -> Tuple:
    """
    x(t) | x(t-1), P(t-1), u(t), A, B, Q
    :param x: previous state, (xdim, batch)
    :param V: previous posteriori covariance, (xdim, xdim)
    :param A: transition matrix, (xdim, xdim)
    :param Q: state noise covariance, (xdim, xdim)
    :param H: observation matrix, (ydim, xdim)
    :param R: observation noise covariance, (ydim, ydim)
    :param cholesky: True if V is Cholesky form, default=True
    :return:
        yhat: predicted observation, (ydim, batch)
        xhat: predicted mean, (xdim, batch)
        Vhat: predicted covariance or its Choleksy, (xdim, xdim)
    """
    n_sample = H.shape[0]
    xhat = A.mm(x)  # Ax
    if cholesky:
        L = V
    else:
        L = linalg.cholesky(V)
    AL = A.mm(L)
    Vhat = AL.mm(AL.t()) + Q  # APA' + Q, n samples one step equivalent to one sample n steps
    yhat = H.mm(xhat)
    if cholesky:
        Vhat = linalg.cholesky(Vhat)
    return yhat, xhat, Vhat


@torch.no_grad()
def update(y: Tensor,
           yhat: Tensor,
           xhat: Tensor,
           Vhat: Tensor,
           H: Tensor,
           R: Tensor,
           cholesky=True,
           ) -> Tuple:
    """
    :param y: measurement, (ydim,)
    :param yhat: predicted measurement, (ydim,)
    :param xhat: predicted state, (xdim,)
    :param Vhat: predicted covariance, (xdim, xdim)
    :param H: measurement matrix, (ydim, xdim)
    :param R: measurement noise covariance, (ydim, ydim)
    :param cholesky: True if Vhat is Cholesky form, default=True
    :return:
        x: posterior mean, (xdim, batch)
        V: posterior covariance or its Cholesky, (xdim, xdim)
    """
    e = y - yhat
    if cholesky:
        Lhat = Vhat
        Vhat = Lhat.mm(Lhat.t())
    else:
        Lhat = linalg.cholesky(Vhat)
    HL = H.mm(Lhat)
    S = HL.mm(HL.t()) + R  # HVH' + R

    L = linalg.cholesky(S)
    # K = H.cholesky_solve(L).mm(Vhat)  # L^{-1}HV
    # K = H.mm(Vhat).cholesky_solve(L)  # L^{-1}HV
    G = H.mm(Vhat).triangular_solve(L, upper=False).solution.t()
    # G' = L^{-1}HV, gain K = VH'S^{-1} = VH'(LL')^{-1} = VH'L'^{-1}L^{-1} = G L^{-1}

    x = xhat + G.mm(e.triangular_solve(L, upper=False).solution)
    V = Vhat - G.mm(G.t())  # minus is dangerous
    # V = positivize(V)
    # assert symmetric(V), 'V is asymmetric'
    if cholesky:
        try:
            V = linalg.cholesky(V)
        except RuntimeError:
            print('Singular covariance', torch.linalg.eigvalsh(V))

    return x, V


@torch.no_grad()
def joseph_update(y: Tensor,
                  yhat: Tensor,
                  xhat: Tensor,
                  Vhat: Tensor,
                  H: Tensor,
                  R: Tensor,
                  cholesky=True,
                  ) -> Tuple:
    """
    :param y: measurement, (ydim,)
    :param yhat: predicted measurement, (ydim,)
    :param xhat: predicted state, (xdim,)
    :param Vhat: predicted covariance, (xdim, xdim)
    :param H: measurement matrix, (ydim, xdim)
    :param R: measurement noise covariance, (ydim, ydim)
    :param cholesky: True if Vhat is Cholesky form, default=True
    :return:
        x: posterior mean, (xdim, batch)
        V: posterior covariance or its Cholesky, (xdim, xdim)
    """
    e = y - yhat
    if cholesky:
        Lhat = Vhat
        Vhat = Lhat.mm(Lhat.t())
    else:
        Lhat = linalg.cholesky(Vhat)
    HL = H.mm(Lhat)
    S = HL.mm(HL.t()) + R  # HVH' + R

    L = linalg.cholesky(S)
    G = H.mm(Vhat).cholesky_solve(L).t()  # L^{-1}HV, gain K = VH'S^{-1} = VH'(LL')^{-1} = VH'L'^{-1}L^{-1} = G L^{-1}
    x = xhat + G.mm(e.cholesky_solve(L))
    # V = (I - KH) Vhat (I - KH)' + K R K'
    eye = torch.eye(Vhat.shape[0])
    IminusKH = eye - G.mm(H.cholesky_solve(L))
    IminusKHLhat = IminusKH.mm(Lhat)
    KR = G.mm(R.sqrt().cholesky_solve(L))  # R's supposed to be diagonal
    V = IminusKHLhat.mm(IminusKHLhat.t()) + KR.mm(KR.t())
    if cholesky:
        V = linalg.cholesky(V)
        # print(torch.linalg.eigvalsh(V))

    return x, V
