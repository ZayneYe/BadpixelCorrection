import torch

def del_tensor_ele(arr,index):
    arr1 = arr[:, 0:index]
    arr2 = arr[:, index+1:]
    return torch.cat((arr1,arr2),dim=1)

def prepocess(feature):
    N, H, W = feature.shape
    feature = feature.view(N, H*W)
    label = feature[:, 12]
    feature = del_tensor_ele(feature, 12)
    return feature.to(torch.float32), label.to(torch.float32)