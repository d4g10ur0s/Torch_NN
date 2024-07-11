import torch
from torch import nn

class Perceptron(nn.Module):
    # simple perceptron for binary classification
    def __init__(self, input_dim, output_dim):
        super(Perceptron, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        # Sigmoid for binary classification
        self.activation = nn.Sigmoid()
    # forward pass
    def forward(self, x):
        output = self.linear(x)
        output = self.activation(output)
        return output
