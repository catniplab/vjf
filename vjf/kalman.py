"""Kalman filter and smoother
x: state
y: observation
x = Ax + w(Q)
y = Hx + v(R)
"""
from typing import Tuple

import torch
from torch import Tensor

from .numerical import positivize
from .util import symmetric


@torch.no_grad()
def predict(
        x: Tensor,
        P: Tensor,
        A: Tensor,
        Q: Tensor,
        H: Tensor,
        R: Tensor) -> Tuple:
    """
    x(t) | x(t-1), P(t-1), u(t), A, B, Q
    :param x: previous state, (xdim, batch)
    :param P: previous posteriori covariance, (xdim, xdim)
    :param A: transition matrix, (xdim, xdim)
    :param Q: state noise covariance, (xdim, xdim)
    :param H: observation matrix, (ydim, xdim)
    :param R: observation noise covariance, (ydim, ydim)
    :return:
        yhat: predicted observation, (ydim, batch)
        xhat: predicted mean, (xdim, batch)
        Phat: predicted covariance, (xdim, xdim)  # only depends on A
    """
    xhat = A.mm(x)  # Ax
    L = torch.linalg.cholesky(P)
    AL = A.mm(L)
    Phat = AL.mm(AL.t()) + Q  # APA' + Q
    # assert symmetric(Phat), 'Phat is asymmetric'
    # Phat = positivize(Phat)
    yhat = H.mm(xhat)
    return yhat, xhat, Phat


@torch.no_grad()
def update(y: Tensor,
           yhat: Tensor,
           xhat: Tensor,
           Phat: Tensor,
           H: Tensor,
           R: Tensor) -> Tuple:
    """
    :param y: measurement, (ydim,)
    :param yhat: predicted measurement, (ydim,)
    :param xhat: predicted state, (xdim,)
    :param Phat: predicted covariance, (xdim, xdim)
    :param H: measurement matrix, (ydim, xdim)
    :param R: measurement noise covariance, (ydim, ydim)
    :return:
        x: posterior mean, (xdim, batch)
        P: posterior covariance, (xdim, xdim)
    """
    # eye = torch.eye(Phat.shape[0])
    e = y - yhat
    L = torch.linalg.cholesky(Phat)
    HL = H.mm(L)
    # S = torch.linalg.multi_dot((H, Phat, H.t())) + R  # HPH' + R
    S = HL.mm(HL.t()) + R
    # assert symmetric(S), 'S is asymmetric'
    # S = positivize(S)
    # L = torch.linalg.cholesky(S)
    # K = Phat.mm(H.cholesky_solve(L).t())  # filter gain, PH'S^{-1}
    # x = xhat + K.mm(e)  # x + Ke
    # P = (eye - K.mm(H)).mm(Phat)  # (I - KH)P
    # P = positivize(P)

    L = torch.linalg.cholesky(S)
    K = H.cholesky_solve(L).mm(Phat)  # L^{-1}HP
    x = xhat + K.t().mm(e.cholesky_solve(L))
    P = Phat - K.t().mm(K)  # minus is dangerous
    P = positivize(P)
    # assert symmetric(P), 'P is asymmetric'
    return x, P
