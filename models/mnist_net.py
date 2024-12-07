import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initial Block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)  # 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        
        # Block 1
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)  # 28x28x16
        self.bn2 = nn.BatchNorm2d(16)
        
        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x16
        
        # Block 2
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, bias=False)  # 14x14x16
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1, bias=False)  # 14x14x16
        self.bn4 = nn.BatchNorm2d(16)
        
        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x16
        
        # Block 3
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1, bias=False)  # 7x7x16
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 32, 3, padding=1, bias=False)  # 7x7x32
        self.bn6 = nn.BatchNorm2d(32)
        
        # Final Block
        self.conv7 = nn.Conv2d(32, 10, 1, bias=False)  # 7x7x10
        
        self.dropout = nn.Dropout(0.1)
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(10, 10)  # Adding a small fully connected layer

    def forward(self, x):
        x = self.dropout(self.bn1(F.relu(self.conv1(x))))
        
        x = self.dropout(self.bn2(F.relu(self.conv2(x))))
        x = self.pool1(x)
        
        x = self.dropout(self.bn3(F.relu(self.conv3(x))))
        x = self.dropout(self.bn4(F.relu(self.conv4(x))))
        x = self.pool2(x)
        
        x = self.dropout(self.bn5(F.relu(self.conv5(x))))
        x = self.dropout(self.bn6(F.relu(self.conv6(x))))
        
        x = self.conv7(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 10)
        x = self.fc1(x)  # Pass through the fully connected layer
        return F.log_softmax(x, dim=1) 