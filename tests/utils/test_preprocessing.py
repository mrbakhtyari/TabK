import numpy as np
import numpy.testing as npt
import pytest

from tabk.utils import apply_standard_scaling


def test_standard_scaling_centers_and_scales_each_feature():
    X = np.random.default_rng(0).normal(loc=5.0, scale=3.0, size=(200, 4))

    scaled = apply_standard_scaling(X)

    assert scaled.dtype == np.float32
    npt.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-5)
    npt.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-5)


def test_standard_scaling_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D"):
        apply_standard_scaling(np.zeros(5))
