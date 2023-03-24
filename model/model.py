import torch
import torch.nn as nn

class MLP_2L(nn.Module):
    def __init__(self, in_size):
        super(MLP_2L,self).__init__()
        self.f1 = nn.Flatten()
        self.l1=nn.Linear(int(in_size), int(2*in_size))
        self.a1=nn.ReLU()
        self.l2=nn.Linear(int(2*in_size), int(0.5 * in_size))
        self.a2=nn.ReLU()
        self.l3=nn.Linear(int(0.5 * in_size), 1)
  
    def forward(self, x):
        x = self.f1(x)
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        return x