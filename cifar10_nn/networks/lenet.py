# -*- coding: utf-8 -*-

# cifar10_nn
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Two Conv-2D followed by two Fully-Connected

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    '''This neural network consists of the following flow:
        -                             Input:  [batch_size,  1, 28, 28]
        - Layer    | Conv2D         | Output: [batch_size, 10, 24, 24]
        - Function | ReLU           | Output: [batch_size, 10, 12, 12]
        - Layer    | MaxPool2D      | Output: [batch_size, 10, 12, 12]
        - Layer    | Conv2D         | Output: [batch_size, 20, 8, 8]
        - Function | ReLU           | Output: [batch_size, 10, 12, 12]
        - Layer    | MaxPool2D      | Output: [batch_size, 10, 12, 12]
        - Function | Resize         | Output: [batch_size, 320]
        - Layer    | FullyConnected | Output: [batch_size, 50]
        - Function | ReLU           | Output: [batch_size, 50]
        - Layer    | FullyConnected | Output: [batch_size, 10]
        - Function | ReLU           | Output: [batch_size, 10]
        - Function | LogSoftMax     | Output: [batch_size, 10]'''

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.name = "LeNet"

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.conv1(x)))
        # Layer 2
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        # Layer 3
        x = F.relu(self.fc1(x))
        # Layer 4
        x = F.relu(self.fc2(x))
        # Layer 5
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
