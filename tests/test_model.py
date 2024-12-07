import pytest
import torch
from models.mnist_net import Net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_batch_norm():
    model = Net()
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model must use Batch Normalization"

def test_dropout():
    model = Net()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model must use Dropout"

def test_fully_connected():
    model = Net()
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert has_fc, "Model must have Fully Connected layers"

def test_parameter_count():
    model = Net()
    param_count = count_parameters(model)
    print(f"Total parameters: {param_count}")
    assert param_count <= 20000, "Model should have less than or equal to 20k parameters" 