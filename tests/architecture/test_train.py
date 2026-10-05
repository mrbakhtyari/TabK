import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tabk.architecture import AppConfig, HeadConfig, KHead, ModelConfig, TrainingConfig
from tabk.architecture.dataset import collate_batch
from tabk.architecture.train import train_one_epoch, validate
from tabk.architecture.utils import create_model


class _LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Small inputs and zero initial weights keep gradients below the clipping threshold.
        self.weight = nn.Parameter(torch.zeros(2, 3, dtype=torch.float64))

    def forward(self, x, row_mask=None, col_mask=None):
        return (x.flatten(1) @ self.weight).log_softmax(dim=-1)


def _classification_data(n_samples):
    rng = torch.Generator().manual_seed(11)
    x = 0.1 * torch.randn(n_samples, 1, 2, generator=rng, dtype=torch.float64)
    targets = torch.zeros(n_samples, 3, dtype=torch.float64)
    targets.scatter_(1, (torch.arange(n_samples) % 3)[:, None], 1)
    return TensorDataset(
        x,
        targets,
        torch.zeros(n_samples, 1, dtype=torch.bool),
        torch.zeros(n_samples, 2, dtype=torch.bool),
    )


@pytest.mark.parametrize("n_batches, accum_steps", [(6, 3), (5, 3), (2, 3), (3, 1)])
def test_accumulation_matches_equivalent_large_batch_updates(n_batches, accum_steps):
    data = _classification_data(n_batches * 2)
    model = _LinearClassifier()
    reference = _LinearClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(
        reference_optimizer, step_size=1, gamma=0.8
    )
    criterion = KHead(8, HeadConfig(min_k=2, max_k=4)).loss_fn()
    config = AppConfig(training=TrainingConfig(device="cpu", accum_steps=accum_steps))

    loss = train_one_epoch(
        model,
        DataLoader(data, batch_size=2),
        optimizer,
        criterion,
        epoch_idx=0,
        fold_idx=0,
        config=config,
        scheduler=scheduler,
    )

    # Independent reference: each concatenated accumulation window is one ordinary batch.
    expected_loss_sum = 0.0
    for x, targets, row_mask, col_mask in DataLoader(data, batch_size=2 * accum_steps):
        reference_optimizer.zero_grad(set_to_none=True)
        batch_loss = criterion(reference(x, row_mask, col_mask), targets)
        batch_loss.backward()
        assert reference.weight.grad.norm() < 1
        reference_optimizer.step()
        reference_scheduler.step()
        expected_loss_sum += batch_loss.item() * len(x)

    assert math.isfinite(loss)
    assert loss == pytest.approx(expected_loss_sum / len(data), abs=1e-12)
    assert torch.count_nonzero(model.weight) > 0
    torch.testing.assert_close(model.weight, reference.weight, rtol=1e-10, atol=1e-12)
    assert scheduler.get_last_lr() == reference_scheduler.get_last_lr()


def test_training_updates_backbone_and_head_with_finite_weights():
    config = AppConfig(
        model=ModelConfig(d_model=8, n_head=2, n_layers=1, num_bins=4, dropout=0),
        training=TrainingConfig(device="cpu", accum_steps=1),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = create_model(config)
        targets = [torch.from_numpy(config.head_config.target_transform(k)) for k in (3, 8)]
        batch = collate_batch([(torch.randn(5, 3), targets[0]), (torch.randn(8, 4), targets[1])])
        backbone_before = model.col_processor.val_encoder.weight.detach().clone()
        head_before = model.head.fc[-1].weight.detach().clone()

        loss = train_one_epoch(
            model,
            [batch],
            torch.optim.SGD(model.parameters(), lr=0.1),
            model.head.loss_fn(),
            epoch_idx=0,
            fold_idx=0,
            config=config,
        )

    assert math.isfinite(loss) and loss > 0
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert not torch.equal(model.col_processor.val_encoder.weight, backbone_before)
    assert not torch.equal(model.head.fc[-1].weight, head_before)


class _FixedPredictions(nn.Module):
    def __init__(self, head_config):
        super().__init__()
        self.head = KHead(8, head_config)

    def forward(self, x, row_mask=None, col_mask=None):
        return x[:, 0, :]


def test_validation_weights_loss_and_metrics_by_sample_count():
    head_config = HeadConfig(min_k=2, max_k=4)
    config = AppConfig(head_config=head_config, training=TrainingConfig(device="cpu"))
    probabilities = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    data = TensorDataset(
        probabilities.log().repeat(3, 1).unsqueeze(1),
        torch.eye(3, dtype=torch.float64),
        torch.zeros(3, 1, dtype=torch.bool),
        torch.zeros(3, 3, dtype=torch.bool),
    )
    model = _FixedPredictions(head_config)

    metrics = validate(model, DataLoader(data, batch_size=2), model.head.loss_fn(), config)

    assert metrics == pytest.approx(
        {
            "val_loss": -(math.log(0.6) + math.log(0.3) + math.log(0.1)) / 3,
            "val_acc": 1 / 3,
            "val_mae": 1,
            "val_mse": 5 / 3,
            "val_off1": 2 / 3,
            "val_rmse": math.sqrt(5 / 3),
        }
    )
