import torch
import pdb
from hydra import compose, initialize
from omegaconf import OmegaConf,DictConfig
from hydra.core.global_hydra import GlobalHydra
import pytest

from unittest import mock
# Importing source_code.py
import sys
sys.path.append("src")
from src.models import train_model
from src.models.model import ConvolutionModel_v1, CNNModuleVar
from src.models.train_model import build_model
