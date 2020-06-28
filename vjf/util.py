import numpy as np
from scipy.ndimage import gaussian_filter1d


def make_xy(x):
    """
    Prepare time series
    Args:
        x: raw dynamics
    Returns:

    """

    if x.ndim == 1:
        x3d = np.atleast_3d(x)
    elif x.ndim == 2:
        x3d = x[None, ...]
    else:
        x3d = x

    x0 = x3d[:, :-1, :]
    x1 = x3d[:, 1:, :]
    return x0, x1


def make_xyu(x, u):
    if x.ndim == 1:
        x3d = x[None, :, None]
    elif x.ndim == 2:
        x3d = x[None, :, :]
    else:
        x3d = x

    if u.ndim == 1:
        u3d = u[None, :, None]
    elif u.ndim == 2:
        u3d = u[None, :, :]
    else:
        u3d = u

    x0 = x3d[:, :-1, :].reshape((-1, x3d.shape[-1]))
    x1 = x3d[:, 1:, :].reshape((-1, x3d.shape[-1]))
    u0 = u3d[:, :-1, :].reshape((-1, u3d.shape[-1]))

    return x0, x1, u0


def embed(x, dim=1, delay=1):
    if x.ndim == 1:
        x3d = np.atleast_3d(x)
    elif x.ndim == 2:
        x3d = x[None, ...]
    else:
        x3d = x
    return np.concatenate(
        [np.roll(x3d, shift=-d * delay, axis=1) for d in range(dim)], axis=2
    )[:, : -(dim - 1) * delay, :]


def smooth_1d(x, sigma=10):
    assert x.ndim == 1
    y = gaussian_filter1d(x, sigma=sigma, mode="constant", cval=0.0)
    return y


def smooth(x, sigma=10):
    return np.stack([smooth_1d(row, sigma) for row in x])


def cut(a, i):
    return a[:i], a[i:]


def assert_numeric(param):
    assert np.all(np.isfinite(param))
