import torch
from models.mnist_net import Net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_batch_norm(model):
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model must use Batch Normalization"

def check_dropout(model):
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model must use Dropout"

def check_fully_connected_or_gap(model):
    has_fc_or_gap = any(isinstance(m, (torch.nn.Linear, torch.nn.AdaptiveAvgPool2d)) for m in model.modules())
    assert has_fc_or_gap, "Model must have Fully Connected layers or Global Average Pooling"

def main():
    model = Net()
    
    # Check parameter count
    param_count = count_parameters(model)
    print(f"Total parameters: {param_count}")
    assert param_count > 10000, "Model should have more than 10k parameters"
    
    # Check for Batch Normalization
    check_batch_norm(model)
    print("✓ Model uses Batch Normalization")
    
    # Check for Dropout
    check_dropout(model)
    print("✓ Model uses Dropout")
    
    # Check for Fully Connected layers or GAP
    check_fully_connected_or_gap(model)
    print("✓ Model has Fully Connected layers or Global Average Pooling")

if __name__ == "__main__":
    main() 