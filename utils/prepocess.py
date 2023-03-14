import torch

def del_tensor_ele(arr,index):
    arr1 = arr[:, 0:index]
    arr2 = arr[:, index+1:]
    return torch.cat((arr1,arr2),dim=1)

def prepocess(feature, label):
    flatten = torch.nn.Flatten()
    feature = flatten(feature.to(torch.float32))
    feature = del_tensor_ele(feature, 12)
    label = label.to(torch.float32)
    label = flatten(label)[:, 12]
    return feature, label