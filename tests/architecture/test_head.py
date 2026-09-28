import numpy as np
import numpy.testing as npt
import pytest
import torch

from tabk.architecture import AppConfig, HeadConfig, KHead


@pytest.mark.parametrize("k", [2, 7, 15])
def test_target_is_a_distribution_peaked_at_k(k):
    config = HeadConfig()
    target = config.target_transform(k)

    assert target.shape == (config.num_classes,)
    npt.assert_allclose(target.sum(), 1.0, rtol=1e-6)
    assert int(target.argmax()) + config.min_k == k


def test_target_clips_k_outside_supported_range():
    config = HeadConfig(min_k=2, max_k=15)
    npt.assert_array_equal(config.target_transform(40), config.target_transform(15))


def test_predict_k_and_metrics_match_targets():
    config = HeadConfig()
    head = KHead(d_model=8, config=config)
    targets = torch.from_numpy(np.stack([config.target_transform(k) for k in (3, 9)]))
    log_probs = torch.log(targets)

    npt.assert_array_equal(head.predict_k(log_probs).numpy(), [3, 9])
    metrics = head.compute_metrics(log_probs, targets)
    assert metrics == {"acc": 1.0, "mae": 0.0, "mse": 0.0, "off1": 1.0}


def test_from_dict_ignores_keys_from_older_configs():
    legacy = {
        "head_config": {
            "head_type": "k_estimator",
            "min_k": 2,
            "max_k": 15,
            "mode": "distribution",
            "sigma": 0.5,
            "class_weights": None,
        },
        "model": {"d_model": 256, "n_head": 8, "n_layers": 4, "model_type": "default"},
        "max_rows": 2500,
    }
    config = AppConfig.from_dict(legacy)

    assert config.head_config == HeadConfig(min_k=2, max_k=15, sigma=0.5)
    assert (config.model.d_model, config.model.n_head, config.model.n_layers) == (256, 8, 4)
    assert config.max_rows == 2500
