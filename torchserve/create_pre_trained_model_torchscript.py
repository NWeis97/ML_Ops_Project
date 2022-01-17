import torch
from torchvision import models
import os

model = models.resnet18(pretrained=True)
script_model = torch.jit.script(model)
script_model.save(os.path.dirname(os.path.realpath(__file__))+'/../deployable_models/resnet18_pretrained.pt')

