import pytest
import torch

from tabk.architecture import AppConfig, ModelConfig
from tabk.architecture.dataset import collate_batch
from tabk.architecture.utils import create_model


@pytest.mark.parametrize("larger_shape", [(12, 3), (5, 8), (12, 8)])
def test_padding_in_a_mixed_size_batch_does_not_change_table_output(larger_shape):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = create_model(
            AppConfig(model=ModelConfig(d_model=16, n_head=2, n_layers=1, num_bins=4))
        ).eval()

    rng = torch.Generator().manual_seed(3)
    table = torch.randn(5, 3, generator=rng)
    larger = torch.randn(*larger_shape, generator=rng)
    target = torch.zeros(14)
    x, _, row_mask, col_mask = collate_batch([(table, target), (larger, target)])

    with torch.no_grad():
        alone = model(table.unsqueeze(0))[0]
        batched = model(x, row_mask, col_mask)[0]
        # Masked cells must be irrelevant even when their values are nonzero.
        padded_cells = row_mask.unsqueeze(-1) | col_mask.unsqueeze(1)
        altered = x.masked_fill(padded_cells, 100)
        altered_output = model(altered, row_mask, col_mask)[0]

    torch.testing.assert_close(batched, alone, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(altered_output, alone, rtol=1e-5, atol=1e-6)
