import torch
from torch import nn


class Regression(nn.Module):
    def __init__(self, num_inputs, num_class):
        super(Regression, self).__init__()
        self.linear = nn.Linear(num_inputs, num_class)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear(out)
        out = self.softmax(out)
        return out
