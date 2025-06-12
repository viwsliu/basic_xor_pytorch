import torch.nn as nn
import torch.nn.functional as F
import torch

class XORNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden = nn.Linear(2, 4)  # 2 inputs, 4 hidden units
    self.output = nn.Linear(4, 1)  # 4 hidden, 1 output

  def forward(self, x):
    x = F.relu(self.hidden(x))
    x = torch.sigmoid(self.output(x))
    return x



