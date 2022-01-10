import os

import pytest
import torch

from src.models.train_model import build_model

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname + "/..")
# Get model struct

model, model_conf = build_model()
model.train()


@pytest.mark.parametrize("batch", [20, 10, 90])
def test_dim_output(batch):
    log_ps = model(torch.randn(batch, 28, 28))
    assert log_ps.shape[0] == batch, "Batch not correct size"
    assert log_ps.shape[1] == 10, "Output dimension per sample is not correct"


def test_dim_input():
    with pytest.raises(ValueError, match="Expected input to a 3D tensor"):
        model(torch.randn(1, 28, 28, 4))

    with pytest.raises(ValueError, match=r"Expected each sample to have shape \[28, 28\]"):
        model(torch.randn(1, 2, 3))
