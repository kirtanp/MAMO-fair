import torch.nn as nn

class NN1(nn.Module):
    def __init__(self, input_dimension):
        super(NN1, self).__init__()
        self.linear1 = nn.Linear(input_dimension, 60)
        self.linear2 = nn.Linear(60, 25)
        self.linear3 = nn.Linear(25, 1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return self.sig(x)