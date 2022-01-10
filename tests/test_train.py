import torch
import pdb
from hydra import compose, initialize
from omegaconf import OmegaConf,DictConfig
from hydra.core.global_hydra import GlobalHydra
import pytest

from unittest import mock
# Importing source_code.py
import sys
sys.path.append("src/models/")
from src.models import train_model
from src.models.model import ConvolutionModel_v1, CNNModuleVar
from src.models.train_model import build_model


@mock.patch("src.models.train_model.torch.optim.zero_grad")
def test_run_it(method1_mock, my_fixture):
    resp = _import.train_model(req)
    method1_mock.assert_called_once()