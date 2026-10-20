import math

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
    center = k - config.min_k
    neighbor = center + 1 if k < config.max_k else center - 1
    assert target[neighbor] / target[center] == pytest.approx(math.exp(-2), rel=1e-6)


@pytest.mark.parametrize("k, clipped", [(-3, 2), (40, 15)])
def test_target_clips_k_outside_supported_range(k, clipped):
    config = HeadConfig(min_k=2, max_k=15)
    npt.assert_array_equal(config.target_transform(k), config.target_transform(clipped))


@pytest.mark.parametrize(
    "sigma, weights",
    [
        (0.5, [math.exp(-8), math.exp(-2), 1, math.exp(-2), math.exp(-8)]),
        (1.0, [math.exp(-2), math.exp(-0.5), 1, math.exp(-0.5), math.exp(-2)]),
    ],
)
def test_target_gaussian_mass_depends_on_sigma(sigma, weights):
    config = HeadConfig(min_k=4, max_k=8, sigma=sigma)
    expected = np.asarray(weights) / sum(weights)

    npt.assert_allclose(config.target_transform(6), expected, rtol=1e-6)


def test_kl_loss_value_and_logit_gradients():
    head = KHead(d_model=8, config=HeadConfig(min_k=2, max_k=3))
    logits = torch.tensor([[0, 0], [math.log(1.5), 0]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=torch.float64)
    expected = (
        0.75 * math.log(0.75 / 0.5)
        + 0.25 * math.log(0.25 / 0.5)
        + 0.25 * math.log(0.25 / 0.6)
        + 0.75 * math.log(0.75 / 0.4)
    ) / 2

    loss = head.loss_fn()(logits.log_softmax(dim=-1), targets)

    assert loss.item() == pytest.approx(expected, abs=1e-12)
    loss.backward()
    torch.testing.assert_close(
        logits.grad, torch.tensor([[-0.125, 0.125], [0.175, -0.175]], dtype=torch.float64)
    )


def test_predict_k_and_metrics_match_targets():
    config = HeadConfig()
    head = KHead(d_model=8, config=config)
    targets = torch.from_numpy(np.stack([config.target_transform(k) for k in (3, 9)]))
    log_probs = torch.log(targets)

    npt.assert_array_equal(head.predict_k(log_probs).numpy(), [3, 9])
    metrics = head.compute_metrics(log_probs, targets)
    assert metrics == {"acc": 1.0, "mae": 0.0, "mse": 0.0, "off1": 1.0}


def test_metrics_include_errors_larger_than_one():
    config = HeadConfig()
    head = KHead(d_model=8, config=config)
    predicted = torch.tensor([3, 6, 10, 15]) - config.min_k
    actual = torch.tensor([3, 5, 7, 13]) - config.min_k
    logits = torch.zeros(4, config.num_classes).scatter_(1, predicted[:, None], 5)
    targets = torch.zeros_like(logits).scatter_(1, actual[:, None], 1)

    assert head.compute_metrics(logits.log_softmax(dim=-1), targets) == {
        "acc": 0.25,
        "mae": 1.5,
        "mse": 3.5,
        "off1": 0.5,
    }


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
