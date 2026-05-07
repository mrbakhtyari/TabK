import numpy as np

from tabk.synthesis.sampling import lhs


def test_lhs_shape_and_range():
    rng = np.random.default_rng(123)
    n, d = 50, 3
    U = lhs(n, d, rng)

    assert U.shape == (n, d)

    assert (U >= 0).all() and (U < 1).all()


def test_lhs_stratification_property():
    rng = np.random.default_rng(42)
    n, d = 100, 2
    U = lhs(n, d, rng)

    for j in range(d):
        bins = (U[:, j] * n).astype(int)
        counts = np.bincount(bins, minlength=n)
        assert (counts == 1).all()


def test_lhs_reproducibility():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    U1 = lhs(20, 2, rng1)
    U2 = lhs(20, 2, rng2)
    assert np.allclose(U1, U2)
