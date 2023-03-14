import torch
import torch.nn as nn

class MLP_2L(nn.Module):
    def __init__(self, in_size):
        super(MLP_2L,self).__init__()
        self.f1 = nn.Flatten()
        self.l1=nn.Linear(24, 48)
        self.a1=nn.ReLU()
        self.l2=nn.Linear(48, 12)
        self.a2=nn.ReLU()
        self.l3=nn.Linear(12, 1)
  
    def forward(self, x):
        x = self.f1(x)
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        return x