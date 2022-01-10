import torch
import pdb
import os.path
import pytest

# content of test_sample.py
Train = torch.load("data/processed/train_dataset.pt")
Test = torch.load("data/processed/test_dataset.pt")
N_Train = 40000
N_Test = 5000

@pytest.mark.skipif(not (os.path.exists("data/processed/train_dataset.pt") or 
                         os.path.exists("data/processed/test_dataset.pt")), reason="Data files not found")
def test_num_obs():
    assert len(Train) == N_Train 
    assert len(Test) == N_Test

def test_image_shape():
    for i in range(len(Train)):
        assert Train.__getitem__(i)[0].shape[0] == 28
        assert Train.__getitem__(i)[0].shape[1] == 28

    for i in range(len(Test)):
        assert Test.__getitem__(i)[0].shape[0] == 28
        assert Test.__getitem__(i)[0].shape[1] == 28


def test_labels():
    for i in range(len(Train)):
        assert type(Train.__getitem__(i)[1]) == torch.Tensor
        assert type(Train.__getitem__(i)[1].item()) == int

    for i in range(len(Test)):
        assert type(Train.__getitem__(i)[1]) == torch.Tensor
        assert type(Test.__getitem__(i)[1].item()) == int

