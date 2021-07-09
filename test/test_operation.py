def test_solve():
    import numpy as np
    from vjf.gp.operation import solve

    m = 10
    p = 5
    A = np.random.randn(m, m)
    A = np.dot(A.T, A)
    b = np.random.randn(m, p)

    x = solve(A, b, "chol").numpy()

    assert np.allclose(A @ x, b)

    x = solve(A, b, "qr").numpy()

    assert np.allclose(A @ x, b)
