import torch
import pdb
from hydra import compose, initialize
from omegaconf import OmegaConf,DictConfig
from hydra.core.global_hydra import GlobalHydra
import pytest

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname + '/..')

from unittest import mock
# Importing source_code.py
import sys
#sys.set
#sys.path.append("src")
from src.models import train_model
from src.models.model import ConvolutionModel_v1, CNNModuleVar
from src.models.train_model import build_model

# Get model struct
model, model_conf = build_model()
model.train()



@pytest.mark.parametrize("batch", [20,10,90])
def test_dim_output(batch):
    # content of test_sample.py
    Train = torch.load("data/processed/train_dataset.pt")
    train_set = torch.utils.data.DataLoader(Train, batch_size=batch, shuffle=True)
    # Get sample batch
    images, labels = next(iter(train_set))

    log_ps = model(images)
    assert log_ps.shape[0] == batch, "Batch not correct size"
    assert log_ps.shape[1] == 10, "Output dimension per sample is not correct"



def test_dim_input():
    # content of test_sample.py
    Train = torch.load("data/processed/train_dataset.pt")
    train_set = torch.utils.data.DataLoader(Train, batch_size=20, shuffle=True)
    # Get sample batch
    images, labels = next(iter(train_set))

    with pytest.raises(ValueError, match='Expected input to a 3D tensor'):
        model(torch.randn(1,28,28,4))

    with pytest.raises(ValueError, match=r'Expected each sample to have shape \[28, 28\]'):
        model(torch.randn(1,2,3))

