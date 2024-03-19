import torch.nn as nn
import torch
from ptflops import get_model_complexity_info

class MLP_1L(nn.Module):
    def __init__(self, in_size=7, cluster_size=5):
        super(MLP_1L,self).__init__()
        self.f1 = nn.Flatten()
        self.l1=nn.Linear(in_size*in_size-cluster_size*cluster_size, cluster_size*cluster_size)
        self.input_size = in_size
        self.cluster_size = cluster_size
  
    def forward(self, x):
        # Calculate the starting indices for the center 5x5 region
        start_row = (self.input_size - self.cluster_size) // 2
        start_col = (self.input_size - self.cluster_size) // 2

        # Flatten the remaining pixels (pixels outside the center 5x5 region)
        flattened_pixels = torch.cat([
            self.f1(x[:,:,:start_row, :]),
            self.f1(x[:,:,start_row + self.cluster_size:, :]),
            self.f1(x[:,:,start_row:start_row + self.cluster_size, :start_col]),
            self.f1(x[:,:,start_row:start_row + self.cluster_size, start_col + self.cluster_size:])
        ], dim=1)
        x = self.l1(flattened_pixels)
        return x
    
class CNN(nn.Module):
    def __init__(self, cluster_size=5, in_size=15):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,stride=1),
            nn.BatchNorm2d(8),
            nn.ELU()
        )
        self.f1 = nn.Flatten()
        self.mlp = MLP_1L(in_size=7, cluster_size=cluster_size)
        self.out=nn.Linear(968+cluster_size*cluster_size, cluster_size*cluster_size)
        self.cluster_size = cluster_size
        self.input_size = in_size

    def forward(self,x):
        mlp_out = self.mlp(x[:,:,4:11,4:11]) # take 7x7 patch from center and pass it through mlp
        start_row = (self.input_size - self.cluster_size) // 2
        start_col = (self.input_size - self.cluster_size) // 2
        input = x.clone()
        input[:,:,start_row:start_row + self.cluster_size, start_col:start_col + self.cluster_size] = mlp_out.view(-1, self.cluster_size, self.cluster_size).unsqueeze(1) # mlp output to replace defective patch at the center
        y=self.conv1(input)
        y=self.conv2(y)
        y=self.f1(y)
        y = torch.cat((y, mlp_out), dim=1)
        y=self.out(y)
        return y.view(-1, self.mlp.cluster_size, self.mlp.cluster_size)
    
if __name__ == "__main__":
    model = CNN(in_size=15, cluster_size=5)
    # torch.save(model, 'vit.pt')
    x = torch.rand((1,1,15,15))
    # x = torch.rand((1,1,256,256))
    y = model(x)
    # x = x.numpy()
        
    # print(x.shape)
    # print(y.shape)
    flops, params = get_model_complexity_info(model, (1, 15, 15), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)