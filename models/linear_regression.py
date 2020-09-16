import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_dimension):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dimension, 1)
        self.sig = nn.Sigmoid()
     
    def forward(self, x):
        x = self.linear(x)
        return self.sig(x)
