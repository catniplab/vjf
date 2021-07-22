import torch


def chol_solve(A, B):
    L = torch.linalg.cholesky(A)  # .cholesky()  # A = LL'
    # X = B.potrs(L, upper=False)  # gradient not implemented
    # A^{-1} B = (LL')^{-1} B = (L')^{-1} L^{-1} B = (L')^{-1} (L^{-1} B)
    # C = B.trtrs(L, upper=False, transpose=False)[0]
    # X = C.trtrs(L, upper=False, transpose=True)[0]
    X = B.cholesky_solve(L)
    return X


def qr_solve(A, B):
    # QRx = b => x = R^{-1} Q'b
    # Q, R = torch.qr(A)
    # X = (Q.t() @ B).trtrs(R)[0]
    X = torch.linalg.solve(A, B)
    return X


def solve(A, B, method="qr"):
    A = torch.as_tensor(A)
    B = torch.as_tensor(B)

    # b is required to be 2D
    if B.dim() == 1:
        B = B.unsqueeze(1)

    if method.lower() == "qr":
        return qr_solve(A, B)
    elif method.lower() == "chol":
        return chol_solve(A, B)
    else:
        raise ValueError(f"Unknown method {method}")


def sqrt(x, eps=1e-12):
    """Safe-gradient square root"""
    return torch.sqrt(x + eps)


def squared_scaled_dist(a, b, gamma):
    if a.shape[1] != b.shape[1]:
        raise ValueError("Inconsistent dimensions")

    a = a * gamma
    b = b * gamma
    b = b.t()  # final outcome in shape of x row * z col
    a2 = torch.sum(a ** 2, dim=1, keepdim=True)
    b2 = torch.sum(b ** 2, dim=0, keepdim=True)
    ab = a.mm(b)
    d2 = a2 - 2 * ab + b2
    return torch.clamp(d2, min=0)


def scaled_dist(a, b, gamma):
    return sqrt(squared_scaled_dist(a, b, gamma))


def kron(a, b):
    """Kronecker product"""
    ra, ca = a.shape
    rb, cb = b.shape
    return torch.reshape(
        a.reshape(ra, 1, ca, 1) * b.reshape(1, rb, 1, cb), (ra * rb, ca * cb)
    )
