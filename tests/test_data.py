import os.path

import pytest
import torch


@pytest.mark.skipif(
    not (
        os.path.exists("data/processed/train_dataset.pt")
        or os.path.exists("data/processed/test_dataset.pt")
    ),
    reason="Data files not found",
)
def test_num_obs():
    # content of test_sample.py
    Train = torch.load("data/processed/train_dataset.pt")
    Test = torch.load("data/processed/test_dataset.pt")
    N_Train = 40000
    N_Test = 5000

    assert len(Train) == N_Train
    assert len(Test) == N_Test


@pytest.mark.skipif(
    not (
        os.path.exists("data/processed/train_dataset.pt")
        or os.path.exists("data/processed/test_dataset.pt")
    ),
    reason="Data files not found",
)
def test_image_shape():
    # content of test_sample.py
    Train = torch.load("data/processed/train_dataset.pt")
    Test = torch.load("data/processed/test_dataset.pt")

    for i in range(len(Train)):
        assert Train.__getitem__(i)[0].shape[0] == 28
        assert Train.__getitem__(i)[0].shape[1] == 28

    for i in range(len(Test)):
        assert Test.__getitem__(i)[0].shape[0] == 28
        assert Test.__getitem__(i)[0].shape[1] == 28


@pytest.mark.skipif(
    not (
        os.path.exists("data/processed/train_dataset.pt")
        or os.path.exists("data/processed/test_dataset.pt")
    ),
    reason="Data files not found",
)
def test_labels():
    # content of test_sample.py
    Train = torch.load("data/processed/train_dataset.pt")
    Test = torch.load("data/processed/test_dataset.pt")

    for i in range(len(Train)):
        assert type(Train.__getitem__(i)[1]) == torch.Tensor
        assert type(Train.__getitem__(i)[1].item()) == int

    for i in range(len(Test)):
        assert type(Train.__getitem__(i)[1]) == torch.Tensor
        assert type(Test.__getitem__(i)[1].item()) == int
