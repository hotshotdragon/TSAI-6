import sys
import os
import pytest
import torch
from models.mnist_net import Net

# Add the root directory of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_model_structure():
    model = Net()
    # Check for Batch Normalization
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model must use Batch Normalization"
    
    # Check for Dropout
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model must use Dropout"
    
    # Check for Fully Connected layers
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert has_fc, "Model must have Fully Connected layers"

def test_model_forward():
    model = Net()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input for MNIST
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"

def test_model_parameters():
    model = Net()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count <= 20000, "Model should have less than or equal to 20k parameters"

def test_output_range():
    model = Net()
    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    assert torch.all(output <= 0), "All output values should be <= 0 for log_softmax"

def test_batch_processing():
    model = Net()
    input_tensor = torch.randn(32, 1, 28, 28)  # Batch of 32
    output = model(input_tensor)
    assert output.shape == (32, 10), "Output shape should be (32, 10) for a batch of 32"